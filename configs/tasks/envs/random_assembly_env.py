# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Random-shape FMB peg-in-hole assembly environment.

At every episode reset a new peg shape is drawn uniformly at random.
All 9 peg rigid bodies are present in every env; the active one is placed on
the table while the other 8 are parked below the scene floor.

Key design points
-----------------
* All 9 pegs are spawned once at scene creation — USD geometry cannot be
  swapped at runtime in Isaac Lab.
* Inactive pegs start at PARK_POS (env-local Z = -2 m, well below the
  ground plane at -1.05 m) and settle on the ground harmlessly.
* ``_active_shape``      (num_envs,)   long  — shape index 0..8
* ``_active_hole_pos_w`` (num_envs, 3) float — hole centre in world frame
  Both tensors are created lazily on the first call to ``_reset_idx`` so
  that ``super().__init__()`` (which triggers an initial reset) is safe.

MDP observation set includes a 9-dim one-hot shape identifier so the policy
can condition its behaviour on the current target shape.
"""

import os
from dataclasses import MISSING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import (
    CollisionPropertiesCfg,
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

from . import assembly_mdp as mdp

_FMB_ASSETS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "assets", "fmb_isaac"
)
_USD_DIR = os.path.join(_FMB_ASSETS, "usd")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ordered shape names — index here == index in _HOLE_POS_TABLE and _active_shape.
PEG_NAMES: list[str] = [
    "circle", "oval", "rectangle", "hexagon", "arch",
    "star", "3prong", "doublesquare", "squarecircle",
]

# Hole centres in env-local frame (board at y=0.55, top face at z=0.050 m).
# Order must match PEG_NAMES exactly.
_HOLE_POS_TABLE_LIST = [
    (-0.009,  0.554, 0.050),   # circle
    (-0.006,  0.620, 0.050),   # oval
    ( 0.066,  0.620, 0.050),   # rectangle
    (-0.073,  0.482, 0.050),   # hexagon
    ( 0.067,  0.473, 0.050),   # arch
    (-0.074,  0.551, 0.050),   # star
    (-0.007,  0.480, 0.050),   # 3prong
    ( 0.065,  0.553, 0.050),   # doublesquare
    (-0.073,  0.621, 0.050),   # squarecircle
]

# Env-local positions used at reset.
# Slot 0 is always the primary (active/target) peg; slots 1-8 are extras.
_TABLE_SLOTS = [
    ( 0.20,  0.05, 0.102),   # slot 0 — primary / active peg (right of robot base)
    (-0.10,  0.20, 0.102),   # slot 1
    ( 0.10,  0.20, 0.102),   # slot 2
    (-0.15,  0.10, 0.102),   # slot 3
    ( 0.15,  0.10, 0.102),   # slot 4
    (-0.10,  0.30, 0.102),   # slot 5
    ( 0.10,  0.30, 0.102),   # slot 6
    ( 0.20,  0.15, 0.102),   # slot 7
    (-0.20,  0.15, 0.102),   # slot 8
]

# Z=0.12: peg flipped 180° around X → actor frame is at peg top; body hangs
# 0.1 m downward, so Z must be peg_height (0.1m) + clearance (0.02m) = 0.12m.
_PARK_POSITIONS = [
    (-0.40, -0.9, 0.12),   # 0 — circle
    (-0.30, -0.9, 0.12),   # 1 — oval
    (-0.20, -0.9, 0.12),   # 2 — rectangle
    (-0.10, -0.9, 0.12),   # 3 — hexagon
    ( 0.00, -0.9, 0.12),   # 4 — arch
    ( 0.10, -0.9, 0.12),   # 5 — star
    ( 0.20, -0.9, 0.12),   # 6 — 3prong
    ( 0.30, -0.9, 0.12),   # 7 — doublesquare
    ( 0.40, -0.9, 0.12),   # 8 — squarecircle
]


def _make_peg_cfg(prim_name: str, usd_path: str, park_pos: tuple) -> RigidObjectCfg:
    """Build a RigidObjectCfg for one peg shape at its unique park position.

    """
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=park_pos,
            rot=(0.0, 1.0, 0.0, 0.0),  # 180° around X — peg upside-down
        ),
        spawn=UsdFileCfg(
            usd_path=usd_path,
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=1,
            ),
            mass_props=MassPropertiesCfg(mass=0.15),
            collision_props=CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------

@configclass
class RandomAssemblySceneCfg(InteractiveSceneCfg):
    """Scene with all 9 FMB peg shapes + hole board + table.

    Robot, EE frame and contact sensor are left as MISSING and filled by the
    concrete robot-specific config (e.g. KinovaRandomAssemblyEnvCfg).
    """

    # ---- Set by concrete cfg ----
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    contact_sensor: ContactSensorCfg = MISSING

    # ---- All 9 peg shapes — each at its own unique park position ----
    # (index in _PARK_POSITIONS must match index in PEG_NAMES)
    peg_circle:       RigidObjectCfg = _make_peg_cfg("Peg_circle",       os.path.join(_USD_DIR, "peg_circle.usd"),       _PARK_POSITIONS[0])
    peg_oval:         RigidObjectCfg = _make_peg_cfg("Peg_oval",         os.path.join(_USD_DIR, "peg_oval.usd"),         _PARK_POSITIONS[1])
    peg_rectangle:    RigidObjectCfg = _make_peg_cfg("Peg_rectangle",    os.path.join(_USD_DIR, "peg_rectangle.usd"),    _PARK_POSITIONS[2])
    peg_hexagon:      RigidObjectCfg = _make_peg_cfg("Peg_hexagon",      os.path.join(_USD_DIR, "peg_hexagon.usd"),      _PARK_POSITIONS[3])
    peg_arch:         RigidObjectCfg = _make_peg_cfg("Peg_arch",         os.path.join(_USD_DIR, "peg_arch.usd"),         _PARK_POSITIONS[4])
    peg_star:         RigidObjectCfg = _make_peg_cfg("Peg_star",         os.path.join(_USD_DIR, "peg_star.usd"),         _PARK_POSITIONS[5])
    peg_3prong:       RigidObjectCfg = _make_peg_cfg("Peg_3prong",       os.path.join(_USD_DIR, "peg_3prong.usd"),       _PARK_POSITIONS[6])
    peg_doublesquare: RigidObjectCfg = _make_peg_cfg("Peg_doublesquare", os.path.join(_USD_DIR, "peg_doublesquare.usd"), _PARK_POSITIONS[7])
    peg_squarecircle: RigidObjectCfg = _make_peg_cfg("Peg_squarecircle", os.path.join(_USD_DIR, "peg_squarecircle.usd"), _PARK_POSITIONS[8])

    # ---- Hole board ----
    fixture: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Fixture",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.55, 0.0)),
        spawn=UsdFileCfg(usd_path=os.path.join(_USD_DIR, "hole_board.usd")),
    )

    # ---- Table ----
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(usd_path=os.path.join(_USD_DIR, "table_top.usd")),
    )

    # ---- Lighting / ground ----
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=800.0),
    )
    plane: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )


# ---------------------------------------------------------------------------
# MDP: observations, rewards, terminations, events
# ---------------------------------------------------------------------------

@configclass
class RandomActionsCfg:
    arm_action:     mdp.JointPositionActionCfg       = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class RandomObservationsCfg:
    """Policy observations for the random-shape task."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Shape identifier — 9-dim one-hot so the policy knows what to grasp
        shape_id = ObsTerm(func=mdp.active_shape_onehot)

        # Peg pose in current hole frame (primary insertion signal)
        peg_pos_in_hole   = ObsTerm(func=mdp.active_peg_pos_in_hole_frame)
        peg_euler_in_hole = ObsTerm(func=mdp.active_peg_euler_in_hole_frame)

        # EE-to-active-peg vector (reaching / grasping phase)
        ee_to_peg = ObsTerm(
            func=mdp.active_ee_to_peg_distance,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )

        # EE pose in hole frame (fine alignment phase)
        ee_pos_in_hole = ObsTerm(
            func=mdp.active_ee_pos_in_hole_frame,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )
        ee_euler_in_hole = ObsTerm(
            func=mdp.active_ee_euler_in_hole_frame,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )

        # Contact force feedback
        contact_force = ObsTerm(
            func=mdp.contact_force_normalized,
            params={"sensor_cfg": SceneEntityCfg("contact_sensor"), "force_limit": 50.0},
        )

        # Proprioceptive
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions   = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RandomEventCfg:
    """Only the scene-level default reset is needed here.

    The per-env shape randomisation and peg repositioning are handled directly
    in ``RandomAssemblyEnv._reset_idx``.
    """
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RandomRewardsCfg:
    """Multi-stage dense rewards for pick-and-insert with random shape."""

    # Stage 1: drive EE toward the active peg
    reach_peg = RewTerm(
        func=mdp.reach_active_peg_reward,
        params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=2.0,
    )

    # Stage 2: drive active peg to its insertion goal
    peg_at_goal = RewTerm(
        func=mdp.active_peg_to_goal_distance,
        weight=10.0,
    )

    # Regularisation
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)
    joint_vel   = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class RandomTerminationsCfg:
    time_out     = DoneTerm(func=mdp.time_out, time_out=True)
    peg_inserted = DoneTerm(
        func=mdp.active_peg_inserted_success,
        params={"threshold": 0.008},
    )
    ee_out_of_workspace = DoneTerm(
        func=mdp.ee_outside_workspace,
        params={
            "workspace_radius": 1.0,           # home pose EE = 0.883 m; 1.0 m gives safe headroom
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "robot_cfg":    SceneEntityCfg("robot"),
        },
    )
    peg_fell_off_table = DoneTerm(
        func=mdp.active_peg_out_of_table,
        params={"min_z": -0.1},
    )


# ---------------------------------------------------------------------------
# Base environment configuration
# ---------------------------------------------------------------------------

@configclass
class RandomAssemblyEnvCfg(ManagerBasedRLEnvCfg):
    """Base config for the random-shape assembly task.

    Concrete robot-specific configs (e.g. KinovaRandomAssemblyEnvCfg) must
    fill in: scene.robot, scene.ee_frame, scene.contact_sensor,
    actions.arm_action, actions.gripper_action.
    """

    scene:        RandomAssemblySceneCfg  = RandomAssemblySceneCfg(num_envs=1024, env_spacing=2.5)
    observations: RandomObservationsCfg  = RandomObservationsCfg()
    actions:      RandomActionsCfg       = RandomActionsCfg()
    rewards:      RandomRewardsCfg       = RandomRewardsCfg()
    terminations: RandomTerminationsCfg  = RandomTerminationsCfg()
    events:       RandomEventCfg         = RandomEventCfg()

    # How many peg shapes to place on the table simultaneously (1–9).
    # The first shape (slot 0) is always the insertion target; the rest are
    # distractors.  All other shapes are parked below the ground plane.
    num_objects_on_table: int = 1

    # When True (training default) a new random shape is drawn at every episode
    # reset.  Set to False in play/eval configs so the shape stays fixed for the
    # whole session.
    change_shape_on_reset: bool = True

    def __post_init__(self):
        self.decimation        = 2
        self.episode_length_s  = 20.0
        self.sim.dt            = 0.01
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity              = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity      = 1024 * 1024   # 9 pegs need much more capacity
        self.sim.physx.friction_correlation_distance           = 0.00625
        self.sim.physx.gpu_max_num_partitions                  = 1  # critical for stable contact with many rigid bodies

        self.viewer.eye    = (1.0, -0.5, 0.8)
        self.viewer.lookat = (0.0,  0.30, 0.1)


# ---------------------------------------------------------------------------
# Custom env class — handles random shape selection at every reset
# ---------------------------------------------------------------------------

class RandomAssemblyEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv subclass that manages random peg shapes per episode.

    Extra per-env state
    -------------------
    _active_shape       (num_envs,)      long  — shape index 0..8 of the target peg
    _table_shapes       (num_envs, N)    long  — shape indices of all N pegs on table
    _active_hole_pos_w  (num_envs, 3)   float  — hole centre in world frame

    All tensors are created lazily on the first call to ``_reset_idx`` to avoid
    device/tensor-not-yet-available issues during ``super().__init__``.

    Config fields that drive behaviour
    -----------------------------------
    cfg.num_objects_on_table  (int, 1–9) — how many pegs sit on the table
    cfg.change_shape_on_reset (bool)     — if False the shape is fixed after
                                           the first assignment (useful for play)
    """

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        n   = len(env_ids)
        N   = max(1, min(self.cfg.num_objects_on_table, len(PEG_NAMES)))

        # ---- Lazy initialisation of per-env tensors -------------------------
        is_first_reset = not hasattr(self, "_active_shape")
        if is_first_reset:
            self._active_shape = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self._table_shapes = torch.zeros(
                self.num_envs, N, dtype=torch.long, device=self.device
            )
            self._active_hole_pos_w = torch.zeros(
                self.num_envs, 3, dtype=torch.float32, device=self.device
            )
            self._hole_pos_table = torch.tensor(
                _HOLE_POS_TABLE_LIST, dtype=torch.float32, device=self.device
            )   # (9, 3) env-local

        # ---- Default scene reset (robot joints, rigid-body default states) --
        super()._reset_idx(env_ids)

        # ---- Shape selection ------------------------------------------------
        # Always sample on the very first reset; then only if change_on_reset.
        if is_first_reset or self.cfg.change_shape_on_reset:
            # Sample N distinct shapes per env (without replacement).
            weights = torch.ones(n, len(PEG_NAMES), device=self.device)
            table_shapes_n = torch.multinomial(weights, num_samples=N, replacement=False)
            # (n, N) — column 0 is the target/active shape.
            self._table_shapes[env_ids] = table_shapes_n
            shape_ids = table_shapes_n[:, 0]          # (n,) active shape
            self._active_shape[env_ids] = shape_ids
            # Update hole world positions for the active shape.
            hole_pos_local = self._hole_pos_table[shape_ids]   # (n, 3)
            origins        = self.scene.env_origins[env_ids, :3]
            self._active_hole_pos_w[env_ids] = origins + hole_pos_local
        else:
            # Re-use existing assignments — just re-place the pegs (they were
            # sent back to PARK by reset_scene_to_default inside super()).
            table_shapes_n = self._table_shapes[env_ids]       # (n, N)

        # ---- Reposition pegs ------------------------------------------------
        # After super()._reset_idx all pegs are at their init_state (unique park
        # positions).  We move the N table pegs to their slots; the rest stay
        # parked at their own dedicated position.
        # Each peg has its own park_local so they never stack → no flashing.
        table_slots  = torch.tensor(
            _TABLE_SLOTS[:N], dtype=torch.float32, device=self.device
        )   # (N, 3) env-local
        park_positions = torch.tensor(
            _PARK_POSITIONS, dtype=torch.float32, device=self.device
        )   # (9, 3) env-local — one row per peg

        for peg_idx, peg_name in enumerate(PEG_NAMES):
            peg = self.scene[f"peg_{peg_name}"]

            # slot_mask[i, k] = True if env_ids[i] has this shape at slot k.
            slot_mask    = (table_shapes_n == peg_idx)   # (n, N) bool
            any_on_table = slot_mask.any(dim=1)          # (n,)   bool

            # Place at occupied slots.
            for k in range(N):
                at_k = slot_mask[:, k]
                if not at_k.any():
                    continue
                eids = env_ids[at_k]
                m    = eids.shape[0]
                pos_w = (
                    self.scene.env_origins[eids, :3]
                    + table_slots[k].unsqueeze(0).expand(m, -1)
                )
                root_state = torch.zeros(m, 13, dtype=torch.float32, device=self.device)
                root_state[:, :3] = pos_w
                root_state[:, 4]  = 1.0   # quat (w=0,x=1,y=0,z=0) — 180° around X, peg upside-down
                peg.write_root_state_to_sim(root_state, env_ids=eids)

            # Explicitly re-park pegs not on the table (they may have drifted
            # during the episode even with gravity disabled).
            parked = env_ids[~any_on_table]
            if parked.numel() > 0:
                m          = parked.shape[0]
                park_local = park_positions[peg_idx]   # unique per-peg position
                pos_w = (
                    self.scene.env_origins[parked, :3]
                    + park_local.unsqueeze(0).expand(m, -1)
                )
                root_state = torch.zeros(m, 13, dtype=torch.float32, device=self.device)
                root_state[:, :3] = pos_w
                root_state[:, 4]  = 1.0   # quat (w=0,x=1,y=0,z=0) — 180° around X, peg upside-down
                peg.write_root_state_to_sim(root_state, env_ids=parked)

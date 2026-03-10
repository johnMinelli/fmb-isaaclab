# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for the FMB peg-in-hole assembly environment.

Full pick-insert task:
  - Peg starts as a free rigid body on the table.
  - Robot grasps the peg, lifts it, carries it to the correct hole, and inserts it.
  - The hole board (peg_board.step, Board 1 medium holes) lies flat at y=0.55 m.
  - Holes face upward; insertion direction is -Z (downward).

Scene assets (from src/assets/fmb_isaac/usd/):
  - hole_board.usd      : FMB peg board with 9 shaped holes (230x230x50 mm, horizontal)
  - peg_<shape>.usd     : one of 9 FMB pegs (real geometry from peg.step, medium-short)

Hole positions in env-local frame (board placed at y=0.55, Z bottom=0):
  Board-local offsets (mm from board XY center, from wire analysis of peg_board.step):
    circle:       (-8.9,   +4.2)   oval:    (-6.1,  +70.4)
    rectangle:   (+66.3,  +70.3)   hexagon: (-73.3,  -68.1)
    arch:        (+67.1,  -77.4)   star:    (-73.7,   +1.4)
    doublesquare:(+64.8,   +3.2)   squarecircle: (-73.0, +71.4)
    3prong:       (-6.6,  -70.0)

Reward shaping:
  - reach_peg:         dense negative distance from EE to peg COM
  - peg_at_goal:       dense negative distance from peg COM to insertion goal
  - action_rate / joint_vel: regularisation penalties
"""

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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

_FMB_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "assets", "fmb_isaac")
_USD_DIR = os.path.join(_FMB_ASSETS, "usd")

##
# Peg shape catalogue  (used in concrete env cfgs to pick a shape)
##

PEG_USDS = {
    "circle":       os.path.join(_USD_DIR, "peg_circle.usd"),
    "rectangle":    os.path.join(_USD_DIR, "peg_rectangle.usd"),
    "arch":         os.path.join(_USD_DIR, "peg_arch.usd"),
    "oval":         os.path.join(_USD_DIR, "peg_oval.usd"),
    "hexagon":      os.path.join(_USD_DIR, "peg_hexagon.usd"),
    "doublesquare": os.path.join(_USD_DIR, "peg_doublesquare.usd"),
    "3prong":       os.path.join(_USD_DIR, "peg_3prong.usd"),
    "squarecircle": os.path.join(_USD_DIR, "peg_squarecircle.usd"),
    "star":         os.path.join(_USD_DIR, "peg_star.usd"),
}

##
# Hole / task parameters
##


@configclass
class HoleFrameCfg:
    """Parameters defining the hole frame and insertion goal.

    ``pos`` and ``rot`` specify the hole-opening centre in env-local frame.
    The hole frame has identity rotation (Z pointing up); insertion is in -Z.

    ``goal_pos_in_hole_frame``: desired peg COM in hole frame when fully inserted.
    With peg height=100 mm, 35 mm insertion depth → peg COM is 15 mm above hole surface:
        goal_z = -0.035 (bottom) + 0.050 (half peg height) = +0.015 m
    """

    pos: tuple[float, float, float] = (-0.009, 0.404, 0.050)
    """Hole opening centre in env-local frame (metres). Default: circle hole."""

    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Hole frame quaternion [w,x,y,z]. Identity: hole faces up, insertion in -Z."""

    goal_pos_in_hole_frame: tuple[float, float, float] = (0.0, 0.0, -0.035)
    """Desired peg actor-frame (bottom face) in hole frame (metres).
    -0.035 means peg bottom is 35 mm below the hole surface (fully inserted).
    Tracked via ``data.root_pos_w`` (actor frame = peg bottom in our centered USD)."""

    target_depth: float = 0.035
    """Insertion depth in metres (used for success threshold)."""


##
# Scene definition
##


@configclass
class AssemblySceneCfg(InteractiveSceneCfg):
    """Full FMB scene: robot + hole board + active peg.

    Fields that must be filled by concrete env cfg:
        robot, ee_frame, contact_sensor
    """

    # ---- Set by concrete cfg ----
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    contact_sensor: ContactSensorCfg = MISSING

    # ---- Active peg (free rigid body on table, grasped by robot) ----
    # Peg geometry from real FMB STEP files, centered with bottom face at Z=0.
    # Height = 100 mm; init_state.pos places COM (~50 mm above bottom) near table.
    # USD path is set per shape in concrete cfg: PEG_USDS["circle"], etc.
    peg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.20, 0.05, 0.102),  # right of robot base; Z raised for upside-down peg (height=0.1m)
            rot=(0.0, 1.0, 0.0, 0.0),  # 180° around X — peg upside-down
        ),
        spawn=UsdFileCfg(
            usd_path=MISSING,       # set in concrete cfg: PEG_USDS["circle"]
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

    # ---- Hole board: FMB peg_board (medium holes), lying flat at y=0.55 ----
    # Board is 230x230x50 mm, centered at XY=0, bottom face at Z=0 in USD.
    # Placed at (0, 0.55, 0): top face (with holes) at world Z=0.050 m.
    # Moved back from y=0.40 to give the robot more workspace between arm and board.
    fixture: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Fixture",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.55, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=UsdFileCfg(
            usd_path=os.path.join(_USD_DIR, "hole_board.usd"),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    # ---- Table ----
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=os.path.join(_USD_DIR, "table_top.usd"),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    # ---- Lighting ----
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=800.0),
    )

    # ---- Ground ----
    plane: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    arm_action: mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observations for full pick-insert task.

    Combines ARCH-style task observations (peg/EE relative to hole frame)
    with proximity observations needed for the reaching/grasping phase.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # Task: peg pose in hole frame (primary insertion signal)
        peg_pos_in_hole = ObsTerm(
            func=mdp.peg_pos_in_hole_frame,
            params={"peg_cfg": SceneEntityCfg("peg")},
        )
        peg_euler_in_hole = ObsTerm(
            func=mdp.peg_euler_in_hole_frame,
            params={"peg_cfg": SceneEntityCfg("peg")},
        )
        # Reach: EE-to-peg distance vector (for approach / grasp phase)
        ee_to_peg = ObsTerm(
            func=mdp.ee_to_peg_distance,
            params={
                "peg_cfg": SceneEntityCfg("peg"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )
        # EE pose in hole frame (useful after grasping for fine alignment)
        ee_pos_in_hole = ObsTerm(func=mdp.ee_pos_in_hole_frame)
        ee_euler_in_hole = ObsTerm(func=mdp.ee_euler_in_hole_frame)
        # Contact force feedback
        contact_force = ObsTerm(
            func=mdp.contact_force_normalized,
            params={"sensor_cfg": SceneEntityCfg("contact_sensor"), "force_limit": 50.0},
        )
        # Proprioceptive
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset scene to default poses at episode start."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    peg_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("peg", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    gripper_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper.*"),
            "static_friction_range": (1.5, 1.5),   # high friction to prevent peg sliding
            "dynamic_friction_range": (1.5, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )


@configclass
class RewardsCfg:
    """Multi-stage dense rewards for pick-and-insert task."""

    # Stage 1: drive EE toward peg (reaching / grasping)
    reach_peg = RewTerm(
        func=mdp.reach_peg_reward,
        params={
            "peg_cfg": SceneEntityCfg("peg"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=2.0,
    )

    # Stage 2: drive peg COM to insertion goal (transport + insertion)
    peg_at_goal = RewTerm(
        func=mdp.peg_to_goal_distance,
        params={"peg_cfg": SceneEntityCfg("peg")},
        weight=10.0,
    )

    # Penalize actions exceeding joint limits — prevents Q from assigning
    # value to out-of-range actions that the simulator would clamp anyway
    action_exceeds_limits = RewTerm(
        func=mdp.action_exceeds_limits,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Regularisation
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    peg_inserted = DoneTerm(
        func=mdp.peg_inserted_success,
        params={
            "threshold": 0.008,
            "peg_cfg": SceneEntityCfg("peg"),
        },
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
        func=mdp.peg_out_of_table,
        params={
            "min_z": -0.1,                     # table surface at Z=0; -0.1 m catches falls/tunnelling
            "peg_cfg": SceneEntityCfg("peg"),
        },
    )


##
# Environment configuration
##


@configclass
class AssemblyEnvCfg(ManagerBasedRLEnvCfg):
    """Base peg-in-hole assembly environment configuration."""

    scene: AssemblySceneCfg = AssemblySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Accessible from obs/reward/termination via env.cfg.hole
    hole: HoleFrameCfg = HoleFrameCfg()

    def __post_init__(self):
        self.decimation = 2           # 50 Hz policy
        self.episode_length_s = 20.0  # longer for full pick-insert task
        self.sim.dt = 0.01            # 100 Hz physics
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_num_partitions = 1  # critical for stable contact simulation

        self.viewer.eye = (1.0, -0.5, 0.8)
        self.viewer.lookat = (0.0, 0.35, 0.1)

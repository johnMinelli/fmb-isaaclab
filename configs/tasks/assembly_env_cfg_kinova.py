# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Concrete assembly env cfg: Kinova Gen3 + Robotiq 2F-85, full pick-insert task.

Scene:
    - Robot:   Kinova Gen3 7-DOF + Robotiq 2F-85 gripper
    - Fixture: FMB peg_board (medium holes, 230x230x50 mm), lying flat at (0, 0.40, 0).
               Top face (with holes) at world Z = 0.050 m.
    - Peg:     Free rigid body on table at (0, 0.25, 0.055), one of 9 FMB shapes.
    - Table:   FMB Isaac table top (src/assets/fmb_isaac/usd/table_top.usd)

Full task: robot grasps peg from table → lifts → aligns with hole → inserts.

Hole positions in env-local frame (board at y=0.40, top face at z=0.050):
    All positions computed from peg_board.step wire analysis (Board 1, medium holes).
    Board-local offsets (mm) measured from board XY center:

    circle:       ( -8.9,   +4.2)  →  env ( -0.009,  0.404, 0.050)
    oval:         ( -6.1,  +70.4)  →  env ( -0.006,  0.470, 0.050)
    rectangle:    (+66.3,  +70.3)  →  env ( +0.066,  0.470, 0.050)
    hexagon:      (-73.3,  -68.1)  →  env ( -0.073,  0.332, 0.050)
    arch:         (+67.1,  -77.4)  →  env ( +0.067,  0.323, 0.050)
    star:         (-73.7,   +1.4)  →  env ( -0.074,  0.401, 0.050)
    doublesquare: (+64.8,   +3.2)  →  env ( +0.065,  0.403, 0.050)
    squarecircle: (-73.0,  +71.4)  →  env ( -0.073,  0.471, 0.050)
    3prong:       ( -6.6,  -70.0)  →  env ( -0.007,  0.330, 0.050)

Hole frame: identity rotation (Z up). Insertion direction: -Z (downward).
Goal (peg bottom at 35 mm depth): hole_frame z = -0.035 m
    Tracked via data.root_pos_w (actor frame = peg bottom in centered USD).
    Peg bottom 35 mm into hole → z_world = 0.050 - 0.035 = 0.015; in hole frame: 0.015 - 0.050 = -0.035 m

Per-shape configs (train + play):
    KinovaAssemblyEnvCfg               — circle    (default)
    KinovaAssemblyEnvCfg_Rectangle     — rectangle
    KinovaAssemblyEnvCfg_Arch          — arch
    KinovaAssemblyEnvCfg_Oval          — oval
    KinovaAssemblyEnvCfg_Hexagon       — hexagon
    KinovaAssemblyEnvCfg_DoubleSquare  — double-square
    KinovaAssemblyEnvCfg_3Prong        — 3-prong
    KinovaAssemblyEnvCfg_SquareCircle  — square-circle
    KinovaAssemblyEnvCfg_Star          — star
"""

import os

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from .envs import assembly_mdp as mdp
from .envs.assembly_env import AssemblyEnvCfg, PEG_USDS

from assets.kinova.config.kinova_gripper import KINOVA_ROBOTIQ  # isort: skip
from .envs.hard_assembly_env import HardAssemblySceneCfg
from .envs.random_assembly_env import RandomAssemblyEnvCfg

# ---------------------------------------------------------------------------
# Hole positions in env-local frame (board at y=0.55, top face at z=0.050)
# ---------------------------------------------------------------------------
_HOLE_POS = {
    "circle":       (-0.009,  0.554, 0.050),
    "oval":         (-0.006,  0.620, 0.050),
    "rectangle":    ( 0.066,  0.620, 0.050),
    "hexagon":      (-0.073,  0.482, 0.050),
    "arch":         ( 0.067,  0.473, 0.050),
    "star":         (-0.074,  0.551, 0.050),
    "doublesquare": ( 0.065,  0.553, 0.050),
    "squarecircle": (-0.073,  0.621, 0.050),
    "3prong":       (-0.007,  0.480, 0.050),
}

# Insertion goal: peg actor frame (bottom face) at 35 mm depth (see docstring)
_GOAL_IN_HOLE = (0.0, 0.0, -0.035)
# Hole frame rotation: identity (holes face up, insertion in -Z)
_HOLE_ROT = (1.0, 0.0, 0.0, 0.0)


@configclass
class KinovaAssemblyEnvCfg(AssemblyEnvCfg):
    """Peg-in-hole with Kinova Gen3 + Robotiq 2F-85, full pick-insert task."""

    def __post_init__(self):
        super().__post_init__()

        # ---- Robot ----
        self.scene.robot = KINOVA_ROBOTIQ.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.activate_contact_sensors = True

        # ---- Actions ----
        self.actions.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-7]"],
            body_name="bracelet_link",
            body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.157)),
            controller=DifferentialIKControllerCfg(
                command_type="position",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=0.5,
        )
        # Binary gripper: open to release, close to grasp
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint", "right_outer_knuckle_joint"],
            open_command_expr={"finger_joint": 0.0, "right_outer_knuckle_joint": 0.0},
            close_command_expr={"finger_joint": 0.8, "right_outer_knuckle_joint": 0.8},
        )

        # ---- EE frame: gripper fingertip centre offset from bracelet_link ----
        # 0.157 m = approximate distance from bracelet_link to Robotiq 2F-85 fingertip centre.
        # Used for reach reward (EE-to-peg distance) and workspace safety check.
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/AssemblyEEFrame"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/bracelet_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.157)),
                ),
            ],
        )

        # ---- Contact sensor on gripper fingers ----
        self.scene.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*finger.*",
            update_period=0.0,
            history_length=10,
            debug_vis=False,
        )

        # ---- Peg shape: circle (default) ----
        self.scene.peg.spawn.usd_path = PEG_USDS["circle"]

        # ---- Hole parameters: circle hole ----
        self.hole.pos = _HOLE_POS["circle"]
        self.hole.rot = _HOLE_ROT
        self.hole.goal_pos_in_hole_frame = _GOAL_IN_HOLE

        # ---- Scene size ----
        self.scene.num_envs = 1024
        self.scene.env_spacing = 2.5


@configclass
class KinovaAssemblyEnvCfg_PLAY(KinovaAssemblyEnvCfg):
    """Circle peg — small scene for visualisation / evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# Per-shape configs
# ---------------------------------------------------------------------------

def _make_shape_cfg(peg_key: str):
    """Return a (train_cfg, play_cfg) pair for a given peg shape."""

    @configclass
    class _TrainCfg(KinovaAssemblyEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            self.scene.peg.spawn.usd_path = PEG_USDS[peg_key]
            self.hole.pos = _HOLE_POS[peg_key]
            self.hole.rot = _HOLE_ROT
            self.hole.goal_pos_in_hole_frame = _GOAL_IN_HOLE

    @configclass
    class _PlayCfg(_TrainCfg):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 16
            self.scene.env_spacing = 2.5
            self.observations.policy.enable_corruption = False

    return _TrainCfg, _PlayCfg


KinovaAssemblyEnvCfg_Rectangle, KinovaAssemblyEnvCfg_Rectangle_PLAY = _make_shape_cfg("rectangle")
KinovaAssemblyEnvCfg_Arch,      KinovaAssemblyEnvCfg_Arch_PLAY      = _make_shape_cfg("arch")
KinovaAssemblyEnvCfg_Oval,      KinovaAssemblyEnvCfg_Oval_PLAY      = _make_shape_cfg("oval")
KinovaAssemblyEnvCfg_Hexagon,   KinovaAssemblyEnvCfg_Hexagon_PLAY   = _make_shape_cfg("hexagon")
KinovaAssemblyEnvCfg_DoubleSquare, KinovaAssemblyEnvCfg_DoubleSquare_PLAY = _make_shape_cfg("doublesquare")
KinovaAssemblyEnvCfg_3Prong,    KinovaAssemblyEnvCfg_3Prong_PLAY    = _make_shape_cfg("3prong")
KinovaAssemblyEnvCfg_SquareCircle, KinovaAssemblyEnvCfg_SquareCircle_PLAY = _make_shape_cfg("squarecircle")
KinovaAssemblyEnvCfg_Star,      KinovaAssemblyEnvCfg_Star_PLAY      = _make_shape_cfg("star")



@configclass
class KinovaHardAssemblyEnvCfg(KinovaAssemblyEnvCfg):
    """Hard peg-in-hole with Kinova Gen3 + Robotiq 2F-85.

    Inherits all robot / EE-frame / contact-sensor / peg / hole setup from
    ``KinovaAssemblyEnvCfg`` (circle shape by default) and adds:
      - ``scene.reorient_fixture`` — FMB reorienting fixture in the workspace.
      - ``episode_length_s = 40`` — extra time for the reorientation phase.
      - Enlarged ``contact_offset`` on the peg so PhysX detects the table
        from the very first frame and the peg never falls through.
    """

    # Override scene to include the reorienting fixture.
    scene: HardAssemblySceneCfg = HardAssemblySceneCfg(num_envs=1024, env_spacing=2.5)

    def __post_init__(self):
        # Sets robot, EE frame, contact sensor, peg shape (circle), hole,
        # num_envs=1024, and all sim / physx parameters.
        super().__post_init__()
        self.episode_length_s = 40.0   # reorientation phase needs more time
        # Enlarge contact_offset so PhysX detects table contact on frame 0
        # when the peg spawns lying flat at z=0.030 m.
        self.scene.peg.spawn.collision_props.contact_offset = 0.05


@configclass
class KinovaHardAssemblyEnvCfg_PLAY(KinovaHardAssemblyEnvCfg):
    """Circle Hard — small scene for visualisation / evaluation (no noise)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


##
# Per-shape Hard configs
##


def _make_hard_shape_cfg(peg_key: str):
    """Return a (train_cfg, play_cfg) pair for the Hard variant of ``peg_key``."""

    @configclass
    class _HardTrainCfg(KinovaHardAssemblyEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            self.scene.peg.spawn.usd_path    = PEG_USDS[peg_key]
            self.hole.pos                    = _HOLE_POS[peg_key]
            self.hole.rot                    = _HOLE_ROT
            self.hole.goal_pos_in_hole_frame = _GOAL_IN_HOLE

    @configclass
    class _HardPlayCfg(_HardTrainCfg):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 16
            self.scene.env_spacing = 2.5
            self.observations.policy.enable_corruption = False

    return _HardTrainCfg, _HardPlayCfg


KinovaHardAssemblyEnvCfg_Rectangle,    KinovaHardAssemblyEnvCfg_Rectangle_PLAY    = _make_hard_shape_cfg("rectangle")
KinovaHardAssemblyEnvCfg_Arch,         KinovaHardAssemblyEnvCfg_Arch_PLAY         = _make_hard_shape_cfg("arch")
KinovaHardAssemblyEnvCfg_Oval,         KinovaHardAssemblyEnvCfg_Oval_PLAY         = _make_hard_shape_cfg("oval")
KinovaHardAssemblyEnvCfg_Hexagon,      KinovaHardAssemblyEnvCfg_Hexagon_PLAY      = _make_hard_shape_cfg("hexagon")
KinovaHardAssemblyEnvCfg_DoubleSquare, KinovaHardAssemblyEnvCfg_DoubleSquare_PLAY = _make_hard_shape_cfg("doublesquare")
KinovaHardAssemblyEnvCfg_3Prong,       KinovaHardAssemblyEnvCfg_3Prong_PLAY       = _make_hard_shape_cfg("3prong")
KinovaHardAssemblyEnvCfg_SquareCircle, KinovaHardAssemblyEnvCfg_SquareCircle_PLAY = _make_hard_shape_cfg("squarecircle")
KinovaHardAssemblyEnvCfg_Star,         KinovaHardAssemblyEnvCfg_Star_PLAY         = _make_hard_shape_cfg("star")


##
# Random-shape variant (all 9 pegs, shape drawn uniformly at each reset)
##


@configclass
class KinovaRandomAssemblyEnvCfg(RandomAssemblyEnvCfg):
    """Random-shape peg-in-hole with Kinova Gen3 + Robotiq 2F-85.

    At every episode reset a peg shape is drawn uniformly at random from the 9
    FMB shapes.  The robot must grasp it from the table and insert it into the
    corresponding hole on the board.

    Registered gym IDs
    ------------------
        Isaac-Assembly-Kinova-Random-v0        — training (1 024 envs)
        Isaac-Assembly-Kinova-Random-v0_PLAY   — evaluation (16 envs, no noise)
    """

    def __post_init__(self):
        super().__post_init__()

        # ---- Robot ----
        self.scene.robot = KINOVA_ROBOTIQ.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.activate_contact_sensors = True

        # ---- Actions ----
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-7]"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint", "right_outer_knuckle_joint"],
            open_command_expr={"finger_joint": 0.0, "right_outer_knuckle_joint": 0.0},
            close_command_expr={"finger_joint": 0.8, "right_outer_knuckle_joint": 0.8},
        )

        # ---- EE frame ----
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/RandomAssemblyEEFrame"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/bracelet_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.157)),
                ),
            ],
        )

        # ---- Contact sensor ----
        self.scene.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*finger.*",
            update_period=0.0,
            history_length=10,
            debug_vis=False,
        )

        # ---- Scene size ----
        self.scene.num_envs    = 1024
        self.scene.env_spacing = 2.5


@configclass
class KinovaRandomAssemblyEnvCfg_PLAY(KinovaRandomAssemblyEnvCfg):
    """Small scene for visualisation and evaluation (no observation noise).

    The peg shape is drawn once at startup and then kept fixed for the whole
    session (``change_shape_on_reset = False``).
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs    = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.change_shape_on_reset = False

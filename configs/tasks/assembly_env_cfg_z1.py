# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Concrete assembly env cfg: Unitree Z1 Air, full pick-insert task.

Scene:
    - Robot:   Unitree Z1 Air 6-DOF + integrated gripper
    - Fixture: FMB peg_board (medium holes, 230x230x50 mm), lying flat at (0, 0.40, 0).
               Top face (with holes) at world Z = 0.050 m.
    - Peg:     Free rigid body on table at (0, 0.25, 0.002), one of 9 FMB shapes.
    - Table:   FMB Isaac table top (src/assets/fmb_isaac/usd/table_top.usd)

EE control via DifferentialInverseKinematicsActionCfg:
    body_name = "gripperStator"  (last fixed arm link)
    body_offset = (0.1, 0.0, 0.0)  (centre of gripper fingers)

Per-shape configs (train + play):
    Z1AssemblyEnvCfg               — circle  (default)
    Z1AssemblyEnvCfg_Rectangle / _PLAY
    Z1AssemblyEnvCfg_Arch / _PLAY
    Z1AssemblyEnvCfg_Oval / _PLAY
    Z1AssemblyEnvCfg_Hexagon / _PLAY
    Z1AssemblyEnvCfg_DoubleSquare / _PLAY
    Z1AssemblyEnvCfg_3Prong / _PLAY
    Z1AssemblyEnvCfg_SquareCircle / _PLAY
    Z1AssemblyEnvCfg_Star / _PLAY

Hard variants (peg starts horizontal):
    Z1HardAssemblyEnvCfg / _PLAY  and per-shape equivalents
"""

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from .envs import assembly_mdp as mdp
from .envs.assembly_env import AssemblyEnvCfg, PEG_USDS
from .envs.hard_assembly_env import HardAssemblySceneCfg
from .envs.random_assembly_env import RandomAssemblyEnvCfg

from assets.z1.config.z1 import UNITREE_Z1_AIR  # isort: skip

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

_GOAL_IN_HOLE = (0.0, 0.0, -0.035)
_HOLE_ROT = (1.0, 0.0, 0.0, 0.0)


##
# Base Z1 assembly config
##


@configclass
class Z1AssemblyEnvCfg(AssemblyEnvCfg):
    """Peg-in-hole with Unitree Z1 Air + integrated gripper, full pick-insert task."""

    def __post_init__(self):
        super().__post_init__()

        # ---- Robot ----
        self.scene.robot = UNITREE_Z1_AIR.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.activate_contact_sensors = True

        # ---- Actions: DiffIK EE position control ----
        # self.actions.arm_action = mdp.JointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["arm_joint[1-6]"],
        #     scale=0.5,
        #     use_default_offset=True,
        # )
        self.actions.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["arm_joint[1-6]"],
            body_name="gripperStator",
            body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.1, 0.0, 0.0)),
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
            joint_names=["jointGripper"],
            open_command_expr={"jointGripper": -0.8},
            close_command_expr={"jointGripper": 0.0},
        )

        # ---- EE frame: centre of gripper fingers offset from gripperStator ----
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/AssemblyEEFrame"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/world",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripperStator",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.1, 0.0, 0.0)),
                ),
            ],
        )

        # ---- Contact sensor on gripper links ----
        self.scene.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper.*",
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
class Z1AssemblyEnvCfg_PLAY(Z1AssemblyEnvCfg):
    """Circle peg — small scene for visualisation / evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 4000.0
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# Per-shape configs
# ---------------------------------------------------------------------------

def _make_shape_cfg(peg_key: str):
    """Return a (train_cfg, play_cfg) pair for a given peg shape."""

    @configclass
    class _TrainCfg(Z1AssemblyEnvCfg):
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


Z1AssemblyEnvCfg_Rectangle,    Z1AssemblyEnvCfg_Rectangle_PLAY    = _make_shape_cfg("rectangle")
Z1AssemblyEnvCfg_Arch,         Z1AssemblyEnvCfg_Arch_PLAY         = _make_shape_cfg("arch")
Z1AssemblyEnvCfg_Oval,         Z1AssemblyEnvCfg_Oval_PLAY         = _make_shape_cfg("oval")
Z1AssemblyEnvCfg_Hexagon,      Z1AssemblyEnvCfg_Hexagon_PLAY      = _make_shape_cfg("hexagon")
Z1AssemblyEnvCfg_DoubleSquare, Z1AssemblyEnvCfg_DoubleSquare_PLAY = _make_shape_cfg("doublesquare")
Z1AssemblyEnvCfg_3Prong,       Z1AssemblyEnvCfg_3Prong_PLAY       = _make_shape_cfg("3prong")
Z1AssemblyEnvCfg_SquareCircle, Z1AssemblyEnvCfg_SquareCircle_PLAY = _make_shape_cfg("squarecircle")
Z1AssemblyEnvCfg_Star,         Z1AssemblyEnvCfg_Star_PLAY         = _make_shape_cfg("star")


##
# Hard variants (peg starts lying flat with random yaw)
##


@configclass
class Z1HardAssemblyEnvCfg(Z1AssemblyEnvCfg):
    """Hard peg-in-hole with Z1 Air — peg starts horizontal, reorienting fixture present."""

    scene: HardAssemblySceneCfg = HardAssemblySceneCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 40.0
        # Enlarge contact_offset so PhysX detects table contact on frame 0
        # when the peg spawns lying flat at z=0.030 m.
        self.scene.peg.spawn.collision_props.contact_offset = 0.05


@configclass
class Z1HardAssemblyEnvCfg_PLAY(Z1HardAssemblyEnvCfg):
    """Circle Hard — small scene for visualisation / evaluation (no noise)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False


def _make_hard_shape_cfg(peg_key: str):
    """Return a (train_cfg, play_cfg) pair for the Hard variant of ``peg_key``."""

    @configclass
    class _HardTrainCfg(Z1HardAssemblyEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            self.scene.peg.spawn.usd_path = PEG_USDS[peg_key]
            self.hole.pos = _HOLE_POS[peg_key]
            self.hole.rot = _HOLE_ROT
            self.hole.goal_pos_in_hole_frame = _GOAL_IN_HOLE

    @configclass
    class _HardPlayCfg(_HardTrainCfg):
        def __post_init__(self):
            super().__post_init__()
            self.observations.policy.enable_corruption = False

    return _HardTrainCfg, _HardPlayCfg


Z1HardAssemblyEnvCfg_Rectangle,    Z1HardAssemblyEnvCfg_Rectangle_PLAY    = _make_hard_shape_cfg("rectangle")
Z1HardAssemblyEnvCfg_Arch,         Z1HardAssemblyEnvCfg_Arch_PLAY         = _make_hard_shape_cfg("arch")
Z1HardAssemblyEnvCfg_Oval,         Z1HardAssemblyEnvCfg_Oval_PLAY         = _make_hard_shape_cfg("oval")
Z1HardAssemblyEnvCfg_Hexagon,      Z1HardAssemblyEnvCfg_Hexagon_PLAY      = _make_hard_shape_cfg("hexagon")
Z1HardAssemblyEnvCfg_DoubleSquare, Z1HardAssemblyEnvCfg_DoubleSquare_PLAY = _make_hard_shape_cfg("doublesquare")
Z1HardAssemblyEnvCfg_3Prong,       Z1HardAssemblyEnvCfg_3Prong_PLAY       = _make_hard_shape_cfg("3prong")
Z1HardAssemblyEnvCfg_SquareCircle, Z1HardAssemblyEnvCfg_SquareCircle_PLAY = _make_hard_shape_cfg("squarecircle")
Z1HardAssemblyEnvCfg_Star,         Z1HardAssemblyEnvCfg_Star_PLAY         = _make_hard_shape_cfg("star")


##
# Random-shape variant (all 9 pegs, shape drawn uniformly at each reset)
##


@configclass
class Z1RandomAssemblyEnvCfg(RandomAssemblyEnvCfg):
    """Random-shape peg-in-hole with Unitree Z1 Air + integrated gripper.

    At every episode reset a peg shape is drawn uniformly at random from the 9
    FMB shapes. The robot must grasp it from the table and insert it into the
    corresponding hole on the board.

    Registered gym IDs
    ------------------
        Isaac-Assembly-Z1-Random-v0
    """

    def __post_init__(self):
        super().__post_init__()

        # ---- Robot ----
        self.scene.robot = UNITREE_Z1_AIR.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.activate_contact_sensors = True

        # ---- Actions ----
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["arm_joint[1-6]"],
            scale=0.5,
            use_default_offset=True,
        )
        # self.actions.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        #     asset_name="robot",
        #     joint_names=["arm_joint[1-6]"],
        #     body_name="gripperStator",
        #     body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.1, 0.0, 0.0)),
        #     controller=DifferentialIKControllerCfg(
        #         command_type="position",
        #         use_relative_mode=True,
        #         ik_method="dls",
        #     ),
        #     scale=0.5,
        # )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["jointGripper"],
            open_command_expr={"jointGripper": -0.8},
            close_command_expr={"jointGripper": 0.0},
        )

        # ---- EE frame: centre of gripper fingers offset from gripperStator ----
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/RandomAssemblyEEFrame"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/world",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripperStator",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.1, 0.0, 0.0)),
                ),
            ],
        )

        # ---- Contact sensor on gripper links ----
        self.scene.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper.*",
            update_period=0.0,
            history_length=10,
            debug_vis=False,
        )

        # ---- Scene size ----
        self.scene.env_spacing = 2.5


@configclass
class Z1RandomAssemblyEnvCfg_PLAY(Z1RandomAssemblyEnvCfg):
    """Small scene for visualisation and evaluation (no observation noise).

    The peg shape is drawn once at startup and then kept fixed for the whole
    session (``change_shape_on_reset = False``).
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False
        self.change_shape_on_reset = False

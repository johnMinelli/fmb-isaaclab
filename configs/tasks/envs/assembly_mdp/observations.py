# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the FMB peg-in-hole assembly environment.

Full pick-insert task observation vector:
    peg_pos_in_hole_frame(3)    — peg COM relative to hole opening (task signal)
    peg_euler_in_hole_frame(3)  — peg orientation in hole frame
    ee_to_peg(3)                — EE-to-peg vector (approach/grasp signal)
    ee_pos_in_hole_frame(3)     — EE position relative to hole (fine alignment)
    ee_euler_in_hole_frame(3)   — EE orientation in hole frame
    contact_force(3)            — normalised gripper contact force
    joint_pos(N)                — joint positions
    joint_vel(N)                — joint velocities
    actions(M)                  — previous actions
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer
from isaaclab.utils.math import euler_xyz_from_quat, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _hole_pose_w(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (hole_pos_w, hole_rot_w) both shape (num_envs, 3/4)."""
    hole_pos_offset = torch.tensor(env.cfg.hole.pos, dtype=torch.float32, device=env.device)
    hole_rot = torch.tensor(env.cfg.hole.rot, dtype=torch.float32, device=env.device)
    hole_pos_w = env.scene.env_origins[:, :3] + hole_pos_offset.unsqueeze(0)
    hole_rot_w = hole_rot.unsqueeze(0).expand(env.num_envs, -1)
    return hole_pos_w, hole_rot_w


# ---------------------------------------------------------------------------
# Peg observations (from RigidObject)
# ---------------------------------------------------------------------------


def peg_pos_in_hole_frame(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """Peg COM position expressed in the hole frame.

    Zero when peg COM is at the hole opening; negative Z when inserted.

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    peg_pos_w = peg.data.root_pos_w  # (num_envs, 3)

    hole_pos_w, hole_rot_w = _hole_pose_w(env)
    peg_pos_in_hole, _ = subtract_frame_transforms(hole_pos_w, hole_rot_w, peg_pos_w)
    return peg_pos_in_hole  # (num_envs, 3)


def peg_euler_in_hole_frame(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """Euler angles (roll, pitch, yaw) of the peg expressed in the hole frame.

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    peg_quat_w = peg.data.root_quat_w  # (num_envs, 4) [w, x, y, z]

    _, hole_rot_w = _hole_pose_w(env)
    _, peg_quat_in_hole = subtract_frame_transforms(
        torch.zeros(env.num_envs, 3, device=env.device),
        hole_rot_w,
        torch.zeros(env.num_envs, 3, device=env.device),
        peg_quat_w,
    )
    roll, pitch, yaw = euler_xyz_from_quat(peg_quat_in_hole)
    return torch.stack([roll, pitch, yaw], dim=-1)  # (num_envs, 3)


def ee_to_peg_distance(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """3-D vector from the EE frame origin to the peg COM (world frame).

    Used to guide the robot toward the peg for reaching and grasping.

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    peg_com_w = peg.data.root_com_pos_w    # (num_envs, 3) — peg middle
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)
    return peg_com_w - ee_pos_w            # (num_envs, 3)


# ---------------------------------------------------------------------------
# EE observations (from FrameTransformer)
# ---------------------------------------------------------------------------


def ee_pos_in_hole_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """EE position expressed in the hole frame.

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)

    hole_pos_w, hole_rot_w = _hole_pose_w(env)
    ee_pos_in_hole, _ = subtract_frame_transforms(hole_pos_w, hole_rot_w, ee_pos_w)
    return ee_pos_in_hole  # (num_envs, 3)


def ee_euler_in_hole_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Euler angles (roll, pitch, yaw) of the EE expressed in the hole frame.

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]  # (num_envs, 4)

    _, hole_rot_w = _hole_pose_w(env)
    _, ee_quat_in_hole = subtract_frame_transforms(
        torch.zeros(env.num_envs, 3, device=env.device),
        hole_rot_w,
        torch.zeros(env.num_envs, 3, device=env.device),
        ee_quat_w,
    )
    roll, pitch, yaw = euler_xyz_from_quat(ee_quat_in_hole)
    return torch.stack([roll, pitch, yaw], dim=-1)  # (num_envs, 3)


# ---------------------------------------------------------------------------
# Contact / force observations
# ---------------------------------------------------------------------------


def contact_force_normalized(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    force_limit: float = 50.0,
    torque_limit: float = 1.0,
) -> torch.Tensor:
    """Normalised contact force (3D) on the gripper, clipped to [-1, 1].

    Uses a sliding-window average over history to de-noise the reading,
    matching ARCH's ``update_force`` function.

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    if contact_sensor.data.net_forces_w_history is not None:
        avg_forces = contact_sensor.data.net_forces_w_history.mean(dim=1).sum(dim=1)
    else:
        avg_forces = contact_sensor.data.net_forces_w.sum(dim=1)  # (num_envs, 3)

    return torch.clamp(avg_forces / force_limit, -1.0, 1.0)  # (num_envs, 3)
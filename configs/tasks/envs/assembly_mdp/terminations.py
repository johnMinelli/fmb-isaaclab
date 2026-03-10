# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for the FMB peg-in-hole assembly environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _hole_pose_w(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    hole_pos_offset = torch.tensor(env.cfg.hole.pos, dtype=torch.float32, device=env.device)
    hole_rot = torch.tensor(env.cfg.hole.rot, dtype=torch.float32, device=env.device)
    hole_pos_w = env.scene.env_origins[:, :3] + hole_pos_offset.unsqueeze(0)
    hole_rot_w = hole_rot.unsqueeze(0).expand(env.num_envs, -1)
    return hole_pos_w, hole_rot_w


def peg_inserted_success(
    env: ManagerBasedRLEnv,
    threshold: float = 0.008,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """Terminate when the peg actor-frame (bottom face) reaches the insertion goal within ``threshold`` metres.

    The goal is ``env.cfg.hole.goal_pos_in_hole_frame`` (default [0, 0, -0.035] m).
    With threshold=0.008 m (8 mm), the peg is considered inserted when its bottom
    is within 8 mm of the target depth (35 mm below hole surface).

    Returns:
        Bool tensor of shape ``(num_envs,)``.
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    peg_pos_w = peg.data.root_pos_w  # (num_envs, 3)

    hole_pos_w, hole_rot_w = _hole_pose_w(env)
    peg_pos_in_hole, _ = subtract_frame_transforms(hole_pos_w, hole_rot_w, peg_pos_w)

    goal = torch.tensor(
        env.cfg.hole.goal_pos_in_hole_frame, dtype=torch.float32, device=env.device
    )
    dist = torch.norm(peg_pos_in_hole - goal.unsqueeze(0), dim=-1)
    return dist < threshold


def ee_outside_workspace(
    env: ManagerBasedRLEnv,
    workspace_radius: float = 1.0,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the EE moves farther than ``workspace_radius`` from the robot base link.

    ``workspace_radius`` is measured from the robot's root body (base_link), not from
    the env origin.  Using the env origin is wrong when the robot is mounted on a table
    because the mounting height alone would exceed a small radius.

    Returns:
        Bool tensor of shape ``(num_envs,)``.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w      = ee_frame.data.target_pos_w[..., 0, :]          # (num_envs, 3)
    robot_base_w  = env.scene[robot_cfg.name].data.root_pos_w      # (num_envs, 3)
    dist = torch.norm(ee_pos_w - robot_base_w, dim=-1)
    return dist > workspace_radius


def peg_out_of_table(
    env: ManagerBasedRLEnv,
    min_z: float = -0.1,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """Terminate when the peg actor-frame (bottom face) drops below ``min_z`` in world Z.

    Catches pegs that fall off the table edge or are launched downward by
    physics depenetration forces.  The table surface is at Z = 0; during
    successful insertion the peg bottom is at Z ≈ 0.015 m (well above the
    default ``min_z = -0.1 m``).

    Returns:
        Bool tensor of shape ``(num_envs,)``.
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    peg_z = peg.data.root_pos_w[:, 2]   # (num_envs,)
    return peg_z < min_z
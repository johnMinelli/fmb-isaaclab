# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the FMB peg-in-hole assembly environment.

Full pick-insert task reward design:

    reach_peg_reward   — shaped reward: drives EE toward peg COM (reaching / grasping)
    peg_to_goal_distance — dense negative distance: peg COM to insertion goal

The goal position (in hole frame) is read from ``env.cfg.hole.goal_pos_in_hole_frame``
(default [0, 0, 0.015] m — peg COM 15 mm above hole surface = 35 mm insertion depth).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject, Articulation
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


def peg_to_goal_distance(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """Dense reward: negative L2 distance from peg COM to insertion goal.

    The goal is ``env.cfg.hole.goal_pos_in_hole_frame`` in the hole local frame.
    With default settings (goal = [0, 0, 0.015] m), the reward is -dist where
    dist=0 means the peg is fully inserted to 35 mm depth.

    Returns:
        Tensor of shape ``(num_envs,)`` ≤ 0.
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    peg_pos_w = peg.data.root_pos_w  # (num_envs, 3)

    hole_pos_w, hole_rot_w = _hole_pose_w(env)
    peg_pos_in_hole, _ = subtract_frame_transforms(hole_pos_w, hole_rot_w, peg_pos_w)

    goal = torch.tensor(
        env.cfg.hole.goal_pos_in_hole_frame, dtype=torch.float32, device=env.device
    )
    dist = torch.norm(peg_pos_in_hole - goal.unsqueeze(0), dim=-1)  # (num_envs,)
    return -dist


def reach_peg_reward(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Shaped reward: exponential approach from EE to peg COM.

    Returns exp(-5 * dist) — ranges from 0 (far) to 1 (at peg).
    This provides a smooth gradient for the reaching and grasping phase.

    Returns:
        Tensor of shape ``(num_envs,)`` in [0, 1].
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    peg_com_w = peg.data.root_com_pos_w                    # (num_envs, 3) — peg middle
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]       # (num_envs, 3)

    dist = torch.norm(peg_com_w - ee_pos_w, dim=-1)        # (num_envs,)
    return torch.exp(-5.0 * dist)


def action_exceeds_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize commanded actions that exceed the joint position limits.

    The simulator clamps actions to motor ranges, so the actual joint positions stay valid.
    But the Q-network sees the raw action output — if action=4 and action=400 both produce
    the same clamped result, Q may assign different values to them, causing unbounded drift.
    This penalty teaches Q that exceeding the range is undesirable.

    Returns the sum of squared excess beyond each joint's soft position limits.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    raw_actions = env.action_manager.action  # [n_envs, n_joints] — the raw network output

    lower = asset.data.soft_joint_pos_limits[:, :raw_actions.shape[-1], 0]
    upper = asset.data.soft_joint_pos_limits[:, :raw_actions.shape[-1], 1]

    below = (lower - raw_actions).clamp(min=0.0)
    above = (raw_actions - upper).clamp(min=0.0)

    return torch.sum(below.square() + above.square(), dim=1)

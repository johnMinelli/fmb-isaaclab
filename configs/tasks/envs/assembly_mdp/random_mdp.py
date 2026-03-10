# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP functions for the random-shape assembly environment.

At every reset the environment samples a new peg shape.  Two per-env tensors
are written by ``RandomAssemblyEnv._reset_idx`` and consumed here:

    env._active_shape       (num_envs,)    long  — index 0..8 into PEG_NAMES
    env._active_hole_pos_w  (num_envs, 3)  float — hole-centre in world frame

All functions prefixed ``active_`` operate on the currently active peg; the
rest are shared utilities (e.g. EE observations in hole frame) that also work
with the dynamic hole position.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import euler_xyz_from_quat, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Ordered list that must match RandomAssemblySceneCfg peg field order and
# _HOLE_POS_TABLE in random_assembly_env.py.
_PEG_NAMES = [
    "circle", "oval", "rectangle", "hexagon", "arch",
    "star", "3prong", "doublesquare", "squarecircle",
]

# Insertion goal in hole frame: peg bottom 35 mm below hole surface.
_GOAL_IN_HOLE = (0.0, 0.0, -0.035)


# ---------------------------------------------------------------------------
# Shared private helpers
# ---------------------------------------------------------------------------

def _active_hole_pose_w(env: ManagerBasedRLEnv):
    """(hole_pos_w, hole_rot_w) from the per-env tensor set at reset."""
    if not hasattr(env, "_active_hole_pos_w"):
        hole_pos_w = torch.zeros(env.num_envs, 3, dtype=torch.float32, device=env.device)
    else:
        hole_pos_w = env._active_hole_pos_w                             # (N, 3)
    hole_rot_w = (
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=env.device)
        .unsqueeze(0).expand(env.num_envs, -1)
    )                                                                    # (N, 4)
    return hole_pos_w, hole_rot_w


def _active_peg_state(
    env: ManagerBasedRLEnv,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(root_pos_w, root_com_pos_w, root_quat_w) for the active peg per env.

    Gathers across all 9 peg objects and selects the active one per row.
    Returns zero tensors when called before ``_reset_idx`` has initialised
    ``_active_shape`` (e.g. during ObservationManager shape inference).
    """
    if not hasattr(env, "_active_shape"):
        zeros3 = torch.zeros(env.num_envs, 3, dtype=torch.float32, device=env.device)
        id_q   = torch.zeros(env.num_envs, 4, dtype=torch.float32, device=env.device)
        id_q[:, 0] = 1.0
        return zeros3, zeros3, id_q

    all_pos  = torch.stack(
        [env.scene[f"peg_{n}"].data.root_pos_w     for n in _PEG_NAMES], dim=1
    )   # (N, 9, 3)
    all_com  = torch.stack(
        [env.scene[f"peg_{n}"].data.root_com_pos_w for n in _PEG_NAMES], dim=1
    )   # (N, 9, 3)
    all_quat = torch.stack(
        [env.scene[f"peg_{n}"].data.root_quat_w    for n in _PEG_NAMES], dim=1
    )   # (N, 9, 4)

    arange = torch.arange(env.num_envs, device=env.device)
    idx    = env._active_shape                                           # (N,)
    return all_pos[arange, idx], all_com[arange, idx], all_quat[arange, idx]


# ---------------------------------------------------------------------------
# Observation functions
# ---------------------------------------------------------------------------

def active_shape_onehot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """9-dimensional one-hot encoding of the current peg shape.

    Allows the policy to condition its behaviour on the target shape.
    Returns all-zeros during ObservationManager shape-inference (before first reset).

    Returns:
        Tensor of shape ``(num_envs, 9)``.
    """
    onehot = torch.zeros(env.num_envs, len(_PEG_NAMES), device=env.device)
    if not hasattr(env, "_active_shape"):
        return onehot
    onehot.scatter_(1, env._active_shape.unsqueeze(-1), 1.0)
    return onehot


def active_peg_pos_in_hole_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Active peg actor-frame (bottom) position expressed in the current hole frame.

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    pos_w, _, _ = _active_peg_state(env)
    hole_pos_w, hole_rot_w = _active_hole_pose_w(env)
    pos_in_hole, _ = subtract_frame_transforms(hole_pos_w, hole_rot_w, pos_w)
    return pos_in_hole


def active_peg_euler_in_hole_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Euler angles (roll, pitch, yaw) of the active peg in the current hole frame.

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    _, _, quat_w = _active_peg_state(env)
    _, hole_rot_w = _active_hole_pose_w(env)
    _, quat_in_hole = subtract_frame_transforms(
        torch.zeros(env.num_envs, 3, device=env.device), hole_rot_w,
        torch.zeros(env.num_envs, 3, device=env.device), quat_w,
    )
    r, p, y = euler_xyz_from_quat(quat_in_hole)
    return torch.stack([r, p, y], dim=-1)


def active_ee_to_peg_distance(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """3-D vector from EE to the active peg COM (world frame).

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    _, com_w, _ = _active_peg_state(env)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    return com_w - ee_pos_w


def active_ee_pos_in_hole_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """EE position expressed in the current (per-env) hole frame.

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    hole_pos_w, hole_rot_w = _active_hole_pose_w(env)
    pos_in_hole, _ = subtract_frame_transforms(hole_pos_w, hole_rot_w, ee_pos_w)
    return pos_in_hole


def active_ee_euler_in_hole_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Euler angles of the EE expressed in the current (per-env) hole frame.

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    _, hole_rot_w = _active_hole_pose_w(env)
    _, quat_in_hole = subtract_frame_transforms(
        torch.zeros(env.num_envs, 3, device=env.device), hole_rot_w,
        torch.zeros(env.num_envs, 3, device=env.device), ee_quat_w,
    )
    r, p, y = euler_xyz_from_quat(quat_in_hole)
    return torch.stack([r, p, y], dim=-1)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def active_peg_to_goal_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dense reward: negative L2 distance from active peg bottom to insertion goal.

    Goal = (0, 0, -0.035) in hole frame (peg bottom 35 mm below surface).

    Returns:
        Tensor of shape ``(num_envs,)`` ≤ 0.
    """
    pos_w, _, _ = _active_peg_state(env)
    hole_pos_w, hole_rot_w = _active_hole_pose_w(env)
    pos_in_hole, _ = subtract_frame_transforms(hole_pos_w, hole_rot_w, pos_w)

    goal = torch.tensor(_GOAL_IN_HOLE, dtype=torch.float32, device=env.device)
    dist = torch.norm(pos_in_hole - goal.unsqueeze(0), dim=-1)
    return -dist


def reach_active_peg_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Shaped reward: exponential approach from EE to active peg COM.

    Returns exp(-5 * dist) — 1 when at peg, → 0 when far.

    Returns:
        Tensor of shape ``(num_envs,)`` in [0, 1].
    """
    _, com_w, _ = _active_peg_state(env)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    dist = torch.norm(com_w - ee_pos_w, dim=-1)
    return torch.exp(-5.0 * dist)


# ---------------------------------------------------------------------------
# Termination functions
# ---------------------------------------------------------------------------

def active_peg_inserted_success(
    env: ManagerBasedRLEnv,
    threshold: float = 0.008,
) -> torch.Tensor:
    """Terminate when the active peg bottom reaches the insertion goal within ``threshold``.

    Returns:
        Bool tensor of shape ``(num_envs,)``.
    """
    pos_w, _, _ = _active_peg_state(env)
    hole_pos_w, hole_rot_w = _active_hole_pose_w(env)
    pos_in_hole, _ = subtract_frame_transforms(hole_pos_w, hole_rot_w, pos_w)

    goal = torch.tensor(_GOAL_IN_HOLE, dtype=torch.float32, device=env.device)
    dist = torch.norm(pos_in_hole - goal.unsqueeze(0), dim=-1)
    return dist < threshold


def active_peg_out_of_table(
    env: ManagerBasedRLEnv,
    min_z: float = -0.1,
) -> torch.Tensor:
    """Terminate when the active peg actor-frame drops below ``min_z`` in world Z.

    Catches the active peg falling off the table edge or being launched
    downward by physics depenetration forces.

    Returns:
        Bool tensor of shape ``(num_envs,)``.
    """
    pos_w, _, _ = _active_peg_state(env)
    return pos_w[:, 2] < min_z
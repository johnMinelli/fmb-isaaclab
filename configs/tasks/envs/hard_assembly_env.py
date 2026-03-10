# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hard variant of the FMB peg-in-hole assembly environment.

Differences from the standard single-object assembly task:
  - The peg starts lying flat on the table (pitch = 90° from vertical) in a
    uniformly random yaw direction, simulating a peg dropped at an unknown
    orientation.
  - The FMB environment reorienting fixture is present in the workspace so
    the robot can use it to flip the peg upright before inserting it.
  - Episode length is 40 s (vs. 20 s for the standard task).

Spawn strategy (avoids all falling / tunnelling physics):
  - Pitch is fixed at π/2 (peg fully horizontal).
  - Z is set to 0.05 m so the peg actor-frame (peg end) is well within the
    table's contact_offset zone (also set to 0.05 m) from the very first
    frame.  PhysX generates contact forces immediately; the peg never falls.
  - Because there is no meaningful fall, no CCD is required.
"""

import math
import os

import torch

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_angle_axis, quat_mul

from .assembly_env import AssemblySceneCfg

_USD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "assets", "fmb_isaac", "usd",
)

# Env-local position of the FMB reorienting fixture.
# Placed to the side at x=-0.15, y=0.30 — out of the peg grasp zone but
# reachable by the robot for peg reorientation before insertion.
_FIXTURE_POS = (-0.15, 0.30, 0.0)

# Env-local XY for the peg start (right of robot base, matching scene image).
_PEG_HARD_X = 0.20
_PEG_HARD_Y = 0.40   # board is at y=0.55; peg starts right of robot base
# Z = 0.030 m: the peg lies flat (pitch=90°) so its cross-section (~20–45 mm
# radius) extends into the table. Raised slightly above 0.025 to avoid mesh
# interpenetration at spawn; contact_offset=0.05 in HardAssemblySceneCfg
# ensures PhysX sees the table from frame 0.
_PEG_HARD_Z = 0.030


##
# Scene configuration
##


@configclass
class HardAssemblySceneCfg(AssemblySceneCfg):
    """Assembly scene with the FMB reorienting fixture added."""

    reorient_fixture: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ReorientFixture",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=_FIXTURE_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=UsdFileCfg(
            usd_path=os.path.join(_USD_DIR, "reorient_fixture.usd"),
            collision_props=CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.05,   # enlarged so flat peg contact is immediate at spawn
                rest_offset=0.0,
            ),
        ),
    )


##
# Custom environment class
##


class HardAssemblyEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv that spawns the peg lying flat with a random yaw.

    At each reset the peg is placed horizontally (pitch = 90°) at a random
    compass direction (yaw ∈ [0, 2π)).  The spawn height (_PEG_HARD_Z) and
    the enlarged contact_offset (set in the cfg) guarantee that PhysX sees
    the table contact from the very first simulation step, so the peg never
    falls through.
    """

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        # 1. Standard reset: robot joints + peg to its default upright init_state.
        super()._reset_idx(env_ids)

        n = len(env_ids)
        peg = self.scene["peg"]

        # 2. Uniformly random yaw; pitch fixed at π/2 (fully flat).
        yaw = torch.rand(n, device=self.device) * (2.0 * math.pi)

        axis_x = torch.zeros(n, 3, device=self.device)
        axis_x[:, 0] = 1.0
        axis_z = torch.zeros(n, 3, device=self.device)
        axis_z[:, 2] = 1.0

        pitch_val = torch.full((n,), math.pi / 2.0, device=self.device)
        q_pitch = quat_from_angle_axis(pitch_val, axis_x)
        q_yaw   = quat_from_angle_axis(yaw,       axis_z)
        q_rand  = quat_mul(q_yaw, q_pitch)

        # 3. Place peg at fixed height; contact is immediate (no falling).
        origins = self.scene.env_origins[env_ids, :3]   # (n, 3)
        pos_w = origins.clone()
        pos_w[:, 0] += _PEG_HARD_X
        pos_w[:, 1] += _PEG_HARD_Y
        pos_w[:, 2] += _PEG_HARD_Z

        # 4. Write state: horizontal orientation, zero velocity.
        root_state = torch.zeros(n, 13, dtype=torch.float32, device=self.device)
        root_state[:, :3]  = pos_w
        root_state[:, 3:7] = q_rand
        peg.write_root_state_to_sim(root_state, env_ids=env_ids)

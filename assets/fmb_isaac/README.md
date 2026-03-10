# FMB Assembly Assets

USD geometry assets for the peg-in-hole assembly task, derived from the
[Functional Manipulation Benchmark (FMB)](https://functional-manipulation-benchmark.github.io/) STEP files.

---

## Directory layout

```
assets/fmb_isaac/
├── README.md                   ← this file
├── usd/                        ← USD files used by the simulation
│   ├── peg_circle.usd
│   ├── peg_oval.usd
│   ├── peg_rectangle.usd
│   ├── peg_hexagon.usd
│   ├── peg_arch.usd
│   ├── peg_star.usd
│   ├── peg_3prong.usd
│   ├── peg_doublesquare.usd
│   ├── peg_squarecircle.usd
│   ├── hole_board.usd          ← medium-hole board (Board 1), used in sim
│   ├── hole_board_1/2/3.usd    ← all three board variants (reference)
│   ├── reorient_fixture.usd
│   └── table_top.usd           ← table mesh used by the simulation
└── scripts/
    ├── add_collision_api.py        ← one-time collision patch for static meshes
    ├── gen_usd.py              ← generate all USD files from STEP
    ├── patch_physics.py        ← add USD Physics schemas to peg files
    └── analyze_board.py        ← analyse board wire positions → hole coords
```

---

## Prerequisites

A Python virtual environment with **cadquery** and **usd-core** (`pxr`) is
required to run the scripts.  Isaac Sim's bundled Python is *not* suitable
here because its NumPy version conflicts with the cadquery dependency chain.

```bash
python -m venv ~/.virtualenvs/myenv
source ~/.virtualenvs/myenv/bin/activate
pip install cadquery "usd-core>=23" "trimesh==4.5.2"
```

> **trimesh pinned to 4.5.2** — Isaac Sim bundles trimesh 4.5.2 internally.
> Upgrading to 4.5.3+ causes a NumPy import error when Isaac Sim's bundled
> `sys.path` is active.  Keep the versions in sync.

---

## STEP source files

Obtain the FMB STEP files from the benchmark repository and place them in
`/tmp/` (or edit `STEP_DIR` in the scripts):

| File | Contents |
|------|----------|
| `peg.step` | All peg shapes — multiple solids in one file |
| `peg_board.step` | Three hole-board variants (large / medium / small holes) |
| `peg_fixture.step` | FMB reorienting fixture |

---

## Regeneration workflow

Run these three steps in order whenever the STEP files change.

### Step 1 — Generate USD meshes

```bash
source ~/.virtualenvs/myenv/bin/activate
cd src/assets/fmb_isaac/scripts
python gen_usd.py
```

Outputs 13 USD files into `usd/`.  Geometry conventions:
- **XY centered at origin** — the prim's local XY = 0 is at the mesh centroid.
- **Z-min = 0** — the bottom face of the object sits at local Z = 0, so
  `init_state.pos.z` in the Isaac Lab config maps directly to the height of
  the bottom face above the table.

### Step 2 — Patch physics schemas

```bash
python patch_physics.py
```

Isaac Lab's `UsdFileCfg` spawner calls `modify_rigid_body_properties()` and
`modify_collision_properties()`, which silently do nothing if the
`PhysicsRigidBodyAPI` / `PhysicsCollisionAPI` schemas are not already present.
This script applies them directly via `pxr`:

| Prim | Schemas added |
|------|---------------|
| `/Peg` (Xform) | `PhysicsRigidBodyAPI`, `PhysicsMassAPI` |
| `/Peg/Visual` (Mesh) | `PhysicsCollisionAPI`, `PhysicsMeshCollisionAPI` |

`physics:approximation = "convexDecomposition"` is used for all shapes to
handle the concave geometry of star and 3-prong pegs.

### Step 3 — (Optional) Re-derive hole positions

```bash
python analyze_board.py --board 1
```

Loads Board 1 (medium holes) from `peg_board.step`, extracts inner wires on
the top face, and prints each hole's centre in both board-local mm and
env-local metres.  Use the output to update `_HOLE_POS` in
`src/configs/tasks/assembly_env_cfg.py`.

---

## Cleanup policy

The active asset set for the assembly environments is:
- `usd/peg_*.usd`
- `usd/hole_board.usd`
- `usd/reorient_fixture.usd`
- `usd/table_top.usd`

Reference-only board variants (`usd/hole_board_1.usd`, `usd/hole_board_2.usd`,
`usd/hole_board_3.usd`) are retained for reproducibility.

---

## USD prim structure

### Peg files (`peg_*.usd`)

```
/Peg   (Xform, defaultPrim)
    APIs: PhysicsRigidBodyAPI, PhysicsMassAPI
/Peg/Visual   (Mesh)
    APIs: PhysicsCollisionAPI, PhysicsMeshCollisionAPI
    physics:approximation = "convexDecomposition"
```

### Board / fixture files

```
/Board   (Xform, defaultPrim)
/Board/Visual   (Mesh)
```

Static assets — no physics schemas (spawned via `AssetBaseCfg`).

---

## Peg shape → solid index mapping

The 9 medium-short pegs live inside `peg.step` as indexed solids.
The mapping below reflects the **corrected** ordering after verifying the
generated geometry (two pairs were swapped in the original FMB indexing).

| Shape key | Solid index | Note |
|-----------|-------------|------|
| circle | 10 | |
| oval | 3 | |
| rectangle | 9 | |
| hexagon | 11 | |
| arch | 19 | |
| doublesquare | 14 | index 14 = doublesquare (not squarecircle) |
| 3prong | 13 | index 13 = 3prong (not star) |
| squarecircle | 2 | index 2 = squarecircle (not doublesquare) |
| star | 17 | index 17 = star (not 3prong) |

---

## Hole positions (Board 1 — medium holes)

Board placed at `(0, 0.40, 0)` in the env frame; top face at Z = 0.050 m.

| Shape | env_x (m) | env_y (m) | env_z (m) |
|-------|-----------|-----------|-----------|
| circle | −0.009 | 0.404 | 0.050 |
| oval | −0.006 | 0.470 | 0.050 |
| rectangle | +0.066 | 0.470 | 0.050 |
| hexagon | −0.073 | 0.332 | 0.050 |
| arch | +0.067 | 0.323 | 0.050 |
| star | −0.074 | 0.401 | 0.050 |
| doublesquare | +0.065 | 0.403 | 0.050 |
| squarecircle | −0.073 | 0.471 | 0.050 |
| 3prong | −0.007 | 0.330 | 0.050 |

Insertion goal (used in rewards/terminations): peg bottom 35 mm below the
hole surface → `goal_pos_in_hole_frame = (0.0, 0.0, -0.035)`.

---

## Isaac Lab data API notes

| Attribute | Meaning |
|-----------|---------|
| `rigid_object.data.root_pos_w` | Actor-frame position = **peg bottom** (Z = 0 in centered USD) |
| `rigid_object.data.root_com_pos_w` | Centre-of-mass position = **peg middle** (~50 mm above bottom) |
| `rigid_object.data.root_quat_w` | Actor-frame orientation |

- Use `root_pos_w` for insertion depth tracking (peg bottom vs hole target).
- Use `root_com_pos_w` for reach/grasp reward (drive EE to peg middle).

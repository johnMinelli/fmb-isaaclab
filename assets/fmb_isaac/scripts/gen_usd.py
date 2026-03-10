"""
Generate USD files for FMB peg-in-hole assets.

Produces 9 peg USDs + 3 board USDs + reorient fixture USD from original FMB STEP files.
All geometry is centered: XY at origin, Z-min = 0 (bottom face sits on table at Z = 0).

Usage
-----
    source ~/.virtualenvs/myenv/bin/activate   # needs cadquery + usd-core (pxr)
    python gen_usd.py

Then run patch_physics.py to add USD Physics schemas to the peg files so
Isaac Lab's RigidObject spawner can find RigidBodyAPI on each peg.

Input STEP files (place in /tmp/ or edit STEP_DIR below)
---------------------------------------------------------
    peg.step           — all FMB peg shapes (multiple solids in one file)
    peg_board.step     — three board variants (small / medium / large holes)
    peg_fixture.step   — FMB reorienting fixture

Output
------
    src/assets/fmb_isaac/usd/
        peg_{shape}.usd          x9  (medium-short pegs)
        hole_board.usd               (Board index 1 = medium holes, used in sim)
        hole_board_1/2/3.usd     x3  (all three boards for reference)
        reorient_fixture.usd
"""

import os
import cadquery as cq
from pxr import Usd, UsdGeom, Gf, Vt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STEP_DIR  = "/tmp"
PEGS_DIR  = os.path.join(os.path.dirname(__file__), "..", "usd")
MM = 0.001   # mm → metres

# ---------------------------------------------------------------------------
# Solid index inside peg.step for each medium-short shape + display colour.
#
# Index mapping was derived by visual inspection of each solid extracted from
# peg.step.  Two pairs were found to be swapped in the original FMB ordering
# and are noted here for future reference:
#   solid 13 = 3prong  (NOT star)
#   solid 17 = star    (NOT 3prong)
#   solid  2 = squarecircle (NOT doublesquare)
#   solid 14 = doublesquare (NOT squarecircle)
# ---------------------------------------------------------------------------
SHAPE_INFO = {
    "circle":       (10, (0.55, 0.27, 0.07)),
    "oval":         ( 3, (0.18, 0.35, 0.65)),
    "rectangle":    ( 9, (0.18, 0.30, 0.60)),
    "hexagon":      (11, (0.10, 0.55, 0.15)),
    "arch":         (19, (1.00, 0.85, 0.00)),
    "doublesquare": (14, (0.50, 0.10, 0.65)),
    "3prong":       (13, (0.65, 0.15, 0.15)),
    "squarecircle": ( 2, (0.85, 0.10, 0.10)),
    "star":         (17, (0.05, 0.12, 0.50)),
}

# Board index inside peg_board.step that contains medium holes (used in sim).
# Board 0: large holes  (~62 mm max)
# Board 1: medium holes (~51 mm max)  ← used
# Board 2: small holes  (~42 mm max)
MEDIUM_BOARD_IDX = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_mesh_prim(stage, prim_path, solid, color_rgb):
    """Tessellate a CadQuery solid and write it as a UsdGeom.Mesh.

    Geometry is recentred: XY at origin, Z-min = 0 (bottom face on table).
    Returns the bounding-box size in metres as (dx, dy, dz).
    """
    bb = solid.BoundingBox()
    cx = (bb.xmin + bb.xmax) / 2.0
    cy = (bb.ymin + bb.ymax) / 2.0
    cz = bb.zmin                        # bottom face → z = 0

    tess  = solid.tessellate(0.3)       # 0.3 mm chord tolerance
    verts = [Gf.Vec3f((p.x - cx)*MM, (p.y - cy)*MM, (p.z - cz)*MM)
             for p in tess[0]]
    tris  = [idx for tri in tess[1] for idx in tri]
    n_tri = len(tris) // 3

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(verts))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(tris))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * n_tri))
    mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color_rgb)]))

    return ((bb.xmax - bb.xmin)*MM,
            (bb.ymax - bb.ymin)*MM,
            (bb.zmax - bb.zmin)*MM)


def write_peg_usd(path, solid, color_rgb):
    """Write a single peg to a USD file.  Prim hierarchy: /Peg (Xform) > /Peg/Visual (Mesh)."""
    stage = Usd.Stage.CreateNew(path)
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    root = UsdGeom.Xform.Define(stage, "/Peg")
    stage.SetDefaultPrim(root.GetPrim())
    bb = write_mesh_prim(stage, "/Peg/Visual", solid, color_rgb)
    stage.GetRootLayer().Save()
    return bb


def write_board_usd(path, solid, color_rgb):
    """Write a board to a USD file.  Prim hierarchy: /Board (Xform) > /Board/Visual (Mesh)."""
    stage = Usd.Stage.CreateNew(path)
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    root = UsdGeom.Xform.Define(stage, "/Board")
    stage.SetDefaultPrim(root.GetPrim())
    bb = write_mesh_prim(stage, "/Board/Visual", solid, color_rgb)
    stage.GetRootLayer().Save()
    return bb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("Loading STEP files …")
peg_cad     = cq.importers.importStep(os.path.join(STEP_DIR, "peg.step"))
board_cad   = cq.importers.importStep(os.path.join(STEP_DIR, "peg_board.step"))
fixture_cad = cq.importers.importStep(os.path.join(STEP_DIR, "peg_fixture.step"))

peg_solids   = peg_cad.solids().vals()
board_solids = board_cad.solids().vals()
fix_solids   = fixture_cad.solids().vals()

print(f"  peg.step:         {len(peg_solids)} solids")
print(f"  peg_board.step:   {len(board_solids)} solids")
print(f"  peg_fixture.step: {len(fix_solids)} solids")

os.makedirs(PEGS_DIR, exist_ok=True)

# ---- Peg USDs ---------------------------------------------------------------
print("\nGenerating peg USDs …")
peg_sizes = {}
for shape, (idx, color) in SHAPE_INFO.items():
    out = os.path.join(PEGS_DIR, f"peg_{shape}.usd")
    bb  = write_peg_usd(out, peg_solids[idx], color)
    peg_sizes[shape] = bb
    print(f"  {shape:15s}  solid[{idx:2d}]  "
          f"{bb[0]*1000:.1f}×{bb[1]*1000:.1f}×{bb[2]*1000:.1f} mm  →  {os.path.basename(out)}")

# ---- Board USDs -------------------------------------------------------------
print("\nGenerating board USDs …")
board_color = (0.15, 0.20, 0.50)

# All three for reference
for i, solid in enumerate(board_solids):
    out = os.path.join(PEGS_DIR, f"hole_board_{i+1}.usd")
    bb  = write_board_usd(out, solid, board_color)
    bb_raw = solid.BoundingBox()
    print(f"  Board {i} (hole_board_{i+1}.usd):  "
          f"{bb[0]*1000:.0f}×{bb[1]*1000:.0f}×{bb[2]*1000:.0f} mm")

# Primary board used in simulation
bb = write_board_usd(
    os.path.join(PEGS_DIR, "hole_board.usd"),
    board_solids[MEDIUM_BOARD_IDX],
    board_color,
)
print(f"\n  hole_board.usd (Board {MEDIUM_BOARD_IDX}, medium):  "
      f"{bb[0]*1000:.0f}×{bb[1]*1000:.0f}×{bb[2]*1000:.0f} mm")

# ---- Reorient fixture -------------------------------------------------------
print("\nGenerating reorient_fixture.usd …")
out = os.path.join(PEGS_DIR, "reorient_fixture.usd")
stage = Usd.Stage.CreateNew(out)
stage.SetMetadata("metersPerUnit", 1.0)
stage.SetMetadata("upAxis", "Z")
root = UsdGeom.Xform.Define(stage, "/Fixture")
stage.SetDefaultPrim(root.GetPrim())
write_mesh_prim(stage, "/Fixture/Visual", fix_solids[0], (0.80, 0.80, 0.80))
stage.GetRootLayer().Save()
print("  reorient_fixture.usd done")

print("\n✓  All USD files written.  Run patch_physics.py next to add physics schemas.")

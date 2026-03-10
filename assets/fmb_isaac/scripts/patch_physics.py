"""
Patch peg USD files with USD Physics schemas required by Isaac Lab.

Background
----------
Isaac Lab's UsdFileCfg spawner calls modify_rigid_body_properties() and
modify_collision_properties(), both of which return False silently if the
corresponding USD Physics API schema is not already present on the prim.
They do NOT auto-apply the schema — that must be done beforehand.

This script applies the following schemas to each peg USD:
    /Peg           ← PhysicsRigidBodyAPI  +  PhysicsMassAPI
    /Peg/Visual    ← PhysicsCollisionAPI  +  PhysicsMeshCollisionAPI
                      physics:approximation = "convexDecomposition"

convexDecomposition is required for concave shapes (star, 3prong); it is safe
to use for all shapes.

Usage
-----
    source ~/.virtualenvs/myenv/bin/activate   # needs usd-core (pxr)
    python patch_physics.py

Run this after gen_usd.py whenever the peg USD files are regenerated.
"""

import os
from pxr import Usd, UsdGeom, UsdPhysics

PEGS_DIR = os.path.join(os.path.dirname(__file__), "..", "usd")

SHAPES = [
    "circle",
    "oval",
    "rectangle",
    "hexagon",
    "arch",
    "star",
    "3prong",
    "doublesquare",
    "squarecircle",
]

for shape in SHAPES:
    path = os.path.join(PEGS_DIR, f"peg_{shape}.usd")
    if not os.path.exists(path):
        print(f"  SKIP (not found): {path}")
        continue

    stage = Usd.Stage.Open(path)

    # --- Root /Peg : rigid body + mass ---
    root = stage.GetPrimAtPath("/Peg")
    if not root.IsValid():
        print(f"  ERROR: /Peg not found in {os.path.basename(path)}")
        continue

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root)

    # --- /Peg/Visual : collision + mesh approximation ---
    visual = stage.GetPrimAtPath("/Peg/Visual")
    if not visual.IsValid():
        print(f"  ERROR: /Peg/Visual not found in {os.path.basename(path)}")
        continue

    UsdPhysics.CollisionAPI.Apply(visual)
    mesh_col = UsdPhysics.MeshCollisionAPI.Apply(visual)
    mesh_col.GetApproximationAttr().Set("convexDecomposition")

    stage.GetRootLayer().Save()
    print(f"  patched  peg_{shape}.usd")

print("\n✓  Physics schemas applied to all peg USDs.")

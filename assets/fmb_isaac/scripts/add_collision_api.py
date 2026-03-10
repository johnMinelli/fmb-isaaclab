#!/usr/bin/env python3
"""One-time script: bake UsdPhysics.CollisionAPI into the board and table USDs.

Run once from a pxr enabled env:
`python src/assets/fmb_isaac/scripts/add_collision_api.py`

After this, Isaac Lab's modify_collision_properties will find the CollisionAPI
on each mesh prim and enable physics collision for the board and table.
Static colliders (CollisionAPI without RigidBodyAPI) are treated as immovable
rigid static actors by PhysX — the robot and peg will collide with them.
"""

import os
from pxr import Usd, UsdGeom, UsdPhysics

_HERE = os.path.dirname(os.path.abspath(__file__))

TARGETS = [
    os.path.join(_HERE, "../usd/hole_board.usd"),
    os.path.join(_HERE, "../usd/reorient_fixture.usd"),
    os.path.join(_HERE, "../usd/table_top.usd"),
]


def add_collision_api(usd_path: str, mesh_approximation: str = "none") -> None:
    """Apply CollisionAPI + MeshCollisionAPI to every mesh prim in the USD.

    Args:
        usd_path: Path to the USD file.
        mesh_approximation: PhysX mesh approximation type.
            "none"              — exact triangle mesh (preserves holes, required for the board).
            "convexHull"        — single convex hull (fast, loses holes).
            "convexDecomposition" — multi-convex decomposition (approximate holes).
    """
    stage = Usd.Stage.Open(usd_path)
    count = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(True)

            # Explicit approximation so PhysX GPU doesn't fall back to a default
            # that silently skips complex meshes.
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                UsdPhysics.MeshCollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr(mesh_approximation)

            count += 1
    stage.GetRootLayer().Save()
    print(f"[OK] {os.path.basename(usd_path)}: set approximation='{mesh_approximation}' on {count} mesh(es)")


if __name__ == "__main__":
    for path in TARGETS:
        if not os.path.exists(path):
            print(f"[SKIP] not found: {path}")
            continue
        # Board has complex concave geometry with holes → must use exact triangle mesh.
        # Table is a simple box → "none" also works fine there.
        add_collision_api(path, mesh_approximation="none")

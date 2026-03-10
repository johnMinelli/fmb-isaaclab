"""
Analyze peg_board.step to identify board variants and hole positions.

This script:
  1. Loads the three board solids from peg_board.step.
  2. For each board, extracts the inner wires on the top face and measures
     their bounding-box size and centroid to identify hole shapes and positions.
  3. Prints the hole centre positions in the env-local frame
     (board at y = 0.40 m, top face at z = 0.050 m).

The output is used to populate _HOLE_POS in assembly_env_cfg.py.

Board classification (from the FMB benchmark):
    Board 0:  large  holes  (max wire span ~62 mm)
    Board 1:  medium holes  (max wire span ~51 mm)  ← used in simulation
    Board 2:  small  holes  (max wire span ~42 mm)

Usage
-----
    source ~/.virtualenvs/myenv/bin/activate   # needs cadquery
    python analyze_board.py [--board 1]

Wire → shape mapping for Board 1 (medium), 13 inner wires → 9 shapes:
    Single-edge wires  → circle (1 wire) or oval (1 wire)
    Rectangular wires  → rectangle, arch, doublesquare (2 wires), squarecircle (2 wires, circle+rect)
    Hexagonal wires    → hexagon
    Multi-circle wires → 3prong  (3 wires clustered together)
    Star outline       → star
"""

import argparse
import os
import cadquery as cq
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Ellipse
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_WIRE, TopAbs_FACE
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp

STEP_DIR = "/tmp"

# Env-local frame offset for the board placed at (0, 0.40, 0) with top face at Z=0.050
BOARD_WORLD_X = 0.0
BOARD_WORLD_Y = 0.40
BOARD_TOP_Z   = 0.050   # metres


def wire_bbox_and_centre(wire_shape):
    """Return (xmin, xmax, ymin, ymax, cx, cy) in mm for a wire's bounding box."""
    box = Bnd_Box()
    BRepBndLib.Add_s(wire_shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    return xmin, xmax, ymin, ymax, cx, cy


def wire_edge_count(wire_shape):
    """Count the number of edges in a wire."""
    exp = TopExp_Explorer(wire_shape, TopAbs_EDGE)
    n = 0
    while exp.More():
        n += 1
        exp.Next()
    return n


def analyse_board(board_solid, board_idx, board_world_x, board_world_y, board_top_z_m):
    """Print hole analysis for one board solid."""
    bb = board_solid.BoundingBox()
    board_cx_mm = (bb.xmin + bb.xmax) / 2.0
    board_cy_mm = (bb.ymin + bb.ymax) / 2.0

    print(f"\nBoard {board_idx}:  "
          f"{bb.xmax-bb.xmin:.0f}×{bb.ymax-bb.ymin:.0f}×{bb.zmax-bb.zmin:.0f} mm  "
          f"(XY centre: {board_cx_mm:.1f}, {board_cy_mm:.1f} mm)")

    # Select top face (max Z)
    top_z = bb.zmax
    faces = board_solid.faces(cq.selectors.DirectionMinMaxSelector(cq.Vector(0, 0, 1), True))
    top_face = faces.val()

    # Extract inner wires (holes)
    face_exp = TopExp_Explorer(top_face.wrapped, TopAbs_WIRE)
    wires = []
    while face_exp.More():
        w = face_exp.Current()
        xmin, xmax, ymin, ymax, cx, cy = wire_bbox_and_centre(w)
        span = max(xmax - xmin, ymax - ymin)
        n_edges = wire_edge_count(w)
        wires.append(dict(wire=w, span=span, cx=cx, cy=cy,
                          dx=xmax-xmin, dy=ymax-ymin, n_edges=n_edges))
        face_exp.Next()

    # Sort by span descending; skip the outer boundary wire (largest)
    wires.sort(key=lambda w: w["span"], reverse=True)
    inner = wires[1:]   # drop outer boundary

    print(f"  {len(inner)} inner wires (holes):")
    print(f"  {'#':>3}  {'span':>7}  {'dx':>7}  {'dy':>7}  {'edges':>6}  "
          f"{'board_cx mm':>11}  {'board_cy mm':>11}  env_x     env_y")
    print(f"  {'-'*3}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*11}  {'-'*11}  {'-'*8}  {'-'*8}")

    max_span = max(w["span"] for w in inner)
    print(f"  Max hole span: {max_span:.1f} mm")

    for i, w in enumerate(inner):
        # Convert board-local mm to env-local metres
        # board centre in env frame = (board_world_x, board_world_y)
        # but board usd is XY-centred so board_cx_mm in STEP = 0 in env
        offset_x_m = (w["cx"] - board_cx_mm) * 0.001
        offset_y_m = (w["cy"] - board_cy_mm) * 0.001
        env_x = board_world_x + offset_x_m
        env_y = board_world_y + offset_y_m
        print(f"  {i:>3}  {w['span']:>7.1f}  {w['dx']:>7.1f}  {w['dy']:>7.1f}  "
              f"{w['n_edges']:>6}  {w['cx']:>11.2f}  {w['cy']:>11.2f}  "
              f"{env_x:>8.4f}  {env_y:>8.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyse FMB board hole positions.")
    parser.add_argument("--board", type=int, default=None,
                        help="Board index to analyse (0/1/2). Default: all boards.")
    args = parser.parse_args()

    step_path = os.path.join(STEP_DIR, "peg_board.step")
    print(f"Loading {step_path} …")
    board_cad = cq.importers.importStep(step_path)
    board_solids = board_cad.solids().vals()
    print(f"  {len(board_solids)} board solids found")

    indices = [args.board] if args.board is not None else list(range(len(board_solids)))
    for idx in indices:
        if idx >= len(board_solids):
            print(f"ERROR: board index {idx} out of range")
            continue
        analyse_board(board_solids[idx], idx,
                      BOARD_WORLD_X, BOARD_WORLD_Y, BOARD_TOP_Z)


if __name__ == "__main__":
    main()
"""Mesh quality library demonstration.

Shows how to use tensorfem.mesh_quality to evaluate FEM mesh quality
for various element types, including cross-validation against analytical values.
"""
import math
import json
from pathlib import Path

from tensorfem.mesh_quality import (
    compute_aspect_ratio,
    compute_skew,
    compute_non_orthogonality,
    compute_jacobian,
    compute_volume,
    evaluate_mesh_quality,
    check_watertight_and_manifold,
    check_orientation,
    evaluate_modeldb_quality,
)
from tensorfem.modeldb import ElementBlock, ModelDB


def _xyz(*pts):
    return {i + 1: (pts[i][0], pts[i][1], pts[i][2]) for i in range(len(pts))}


def _xy(*pts):
    return {i + 1: (pts[i][0], pts[i][1]) for i in range(len(pts))}


def demo_perfect_elements():
    """Quality metrics for perfect (ideal) elements."""
    print("=" * 60)
    print("Perfect Element Quality Metrics")
    print("=" * 60)

    # Equilateral triangle
    a = 1.0
    nodes_tri = _xy((0, 0), (a, 0), (a / 2, a * math.sqrt(3) / 2))
    conn_tri = [1, 2, 3]
    print(f"\nEquilateral Triangle (side={a}):")
    print(f"  Aspect ratio:  {compute_aspect_ratio('CPS3', _coords(nodes_tri, conn_tri)):.6f}")
    print(f"  Skew:          {compute_skew('CPS3', _coords(nodes_tri, conn_tri)):.6f}°")
    print(f"  Jacobian:      {compute_jacobian('CPS3', _coords(nodes_tri, conn_tri))[0]:.6f}")
    print(f"  Area:          {compute_volume('CPS3', _coords(nodes_tri, conn_tri)):.6f}")

    # Unit square
    nodes_sq = _xy((0, 0), (1, 0), (1, 1), (0, 1))
    conn_sq = [1, 2, 3, 4]
    print(f"\nUnit Square:")
    print(f"  Aspect ratio:  {compute_aspect_ratio('CPS4', _coords(nodes_sq, conn_sq)):.6f}")
    print(f"  Skew:          {compute_skew('CPS4', _coords(nodes_sq, conn_sq)):.6f}°")
    j_min, j_max = compute_jacobian('CPS4', _coords(nodes_sq, conn_sq))
    print(f"  Jacobian:      [{j_min:.6f}, {j_max:.6f}]")
    print(f"  Area:          {compute_volume('CPS4', _coords(nodes_sq, conn_sq)):.6f}")

    # Regular tetrahedron
    nodes_tet = _xyz(
        (0, 0, 0), (a, 0, 0),
        (a / 2, a * math.sqrt(3) / 2, 0),
        (a / 2, a * math.sqrt(3) / 6, a * math.sqrt(6) / 3),
    )
    conn_tet = [1, 2, 3, 4]
    print(f"\nRegular Tetrahedron (edge={a}):")
    print(f"  Aspect ratio:  {compute_aspect_ratio('C3D4', _coords(nodes_tet, conn_tet)):.6f}")
    print(f"  Skew:          {compute_skew('C3D4', _coords(nodes_tet, conn_tet)):.6f}°")
    print(f"  Non-ortho:     {compute_non_orthogonality('C3D4', _coords(nodes_tet, conn_tet)):.6f}°")
    print(f"  Jacobian:      {compute_jacobian('C3D4', _coords(nodes_tet, conn_tet))[0]:.6f}")
    print(f"  Volume:        {compute_volume('C3D4', _coords(nodes_tet, conn_tet)):.6f}")

    # Unit cube
    nodes_hex = _xyz(
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    )
    conn_hex = [1, 2, 3, 4, 5, 6, 7, 8]
    print(f"\nUnit Cube:")
    print(f"  Aspect ratio:  {compute_aspect_ratio('C3D8', _coords(nodes_hex, conn_hex)):.6f}")
    print(f"  Skew:          {compute_skew('C3D8', _coords(nodes_hex, conn_hex)):.6f}°")
    print(f"  Non-ortho:     {compute_non_orthogonality('C3D8', _coords(nodes_hex, conn_hex)):.6f}°")
    j_min, j_max = compute_jacobian('C3D8', _coords(nodes_hex, conn_hex))
    print(f"  Jacobian:      [{j_min:.6f}, {j_max:.6f}]")
    print(f"  Volume:        {compute_volume('C3D8', _coords(nodes_hex, conn_hex)):.6f}")


def demo_poor_quality():
    """Quality metrics for poor-quality elements."""
    print("\n" + "=" * 60)
    print("Poor Quality Element Detection")
    print("=" * 60)

    # High aspect ratio triangle
    nodes = _xy((0, 0), (10, 0), (0, 0.1))
    conn = [1, 2, 3]
    ar = compute_aspect_ratio('CPS3', _coords(nodes, conn))
    sk = compute_skew('CPS3', _coords(nodes, conn))
    print(f"\nSliver Triangle (10×0.1):")
    print(f"  Aspect ratio:  {ar:.2f}  (ideal: 1.0)")
    print(f"  Skew:          {sk:.2f}°  (ideal: 0°)")

    # Skewed quad
    nodes = _xy((0, 0), (1, 0), (1.8, 0.2), (0.2, 1))
    conn = [1, 2, 3, 4]
    ar = compute_aspect_ratio('CPS4', _coords(nodes, conn))
    sk = compute_skew('CPS4', _coords(nodes, conn))
    print(f"\nSkewed Quad:")
    print(f"  Aspect ratio:  {ar:.2f}")
    print(f"  Skew:          {sk:.2f}°")

    # Inverted quad
    nodes = _xy((0, 0), (1, 0), (1, 1), (0, 1))
    conn = [1, 4, 3, 2]  # reversed
    j_min, _ = compute_jacobian('CPS4', _coords(nodes, conn))
    bad = check_orientation(nodes, [(1, 'CPS4', conn)])
    print(f"\nInverted Quad (reversed nodes):")
    print(f"  Min Jacobian:  {j_min:.6f}  (should be < 0)")
    print(f"  Bad elements:  {bad}")


def demo_full_report():
    """Full mesh quality report for a mixed mesh."""
    print("\n" + "=" * 60)
    print("Full Mesh Quality Report")
    print("=" * 60)

    # Mixed 2D mesh: 2 triangles + 1 quad
    nodes = _xy(
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
    )
    elements = [
        (1, "CPS3", [1, 2, 4]),
        (2, "CPS4", [2, 3, 6, 5]),
        (3, "CPS3", [1, 4, 5]),
    ]

    report = evaluate_mesh_quality(nodes, elements)
    d = report.to_dict()
    print(f"\nMesh: {d['n_elements']} elements, {d['n_nodes']} nodes")
    print(f"  Max aspect ratio:  {d['max_aspect_ratio']:.4f}")
    print(f"  Max skew:          {d['max_skew_deg']:.4f}°")
    print(f"  Min Jacobian:      {d['min_jacobian']:.6f}")
    print(f"  Mean Jacobian:     {d['mean_jacobian']:.6f}")
    print(f"  Total area:        {d['total_volume']:.6f}")
    print(f"  Orientation issues: {d['orientation_violations']}")

    # Per-element details
    print("\nPer-element quality:")
    for eq in report.elements:
        print(f"  Element {eq.element_id} ({eq.element_type}): "
              f"AR={eq.aspect_ratio:.3f}, skew={eq.skew:.2f}°, "
              f"J=[{eq.jacobian_min:.4f}, {eq.jacobian_max:.4f}], "
              f"vol={eq.volume:.4f}, orient={eq.orientation}")


def demo_3d_watertight():
    """3D mesh watertightness and manifoldness check."""
    print("\n" + "=" * 60)
    print("3D Mesh Watertightness & Manifoldness")
    print("=" * 60)

    # Two tets sharing a face
    nodes = _xyz(
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0),
    )
    elements = [
        (1, "C3D4", [1, 2, 3, 4]),
        (2, "C3D4", [2, 5, 3, 4]),
    ]

    wt, mf, viol = check_watertight_and_manifold(nodes, elements)
    print(f"\nTwo-tet mesh:")
    print(f"  Watertight:  {wt}")
    print(f"  Manifold:    {mf}")
    print(f"  Violations:  {len(viol)}")

    # Three tets sharing one face (non-manifold)
    nodes2 = _xyz(
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (1, 1, 0), (-1, 0, 0),
    )
    elements2 = [
        (1, "C3D4", [1, 2, 3, 4]),
        (2, "C3D4", [2, 5, 3, 4]),
        (3, "C3D4", [2, 6, 3, 4]),
    ]
    wt2, mf2, viol2 = check_watertight_and_manifold(nodes2, elements2)
    print(f"\nThree-tet mesh (shared face):")
    print(f"  Watertight:  {wt2}")
    print(f"  Manifold:    {mf2}")
    print(f"  Violations:  {len(viol2)}")
    if viol2:
        for face, owners in viol2:
            print(f"    Face {face} owned by {owners}")


def _coords(nodes, conn):
    import torch
    pts = []
    for nid in conn:
        p = nodes[nid]
        if len(p) == 2:
            pts.append([p[0], p[1], 0.0])
        else:
            pts.append([p[0], p[1], p[2]])
    return torch.tensor(pts, dtype=torch.float64)


if __name__ == "__main__":
    demo_perfect_elements()
    demo_poor_quality()
    demo_full_report()
    demo_3d_watertight()

    print("\n" + "=" * 60)
    print("Demo complete. All metrics computed successfully.")
    print("=" * 60)

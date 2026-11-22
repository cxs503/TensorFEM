"""Compare two face recovery rules on archived shell DOFs against five independent HEX20 solves."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

FIELDS = (
    Path(__file__).resolve().parents[1]
    / "docs/assets/benchmark-clouds/hemisphere-hole/results.json"
)
MESHES = ((20, 24, 1), (28, 36, 1), (34, 42, 1), (24, 24, 1), (24, 24, 2))


def recover(record, radial):
    import torch
    from tensorfem.hemisphere_benchmark_fields import surface_stress

    def tensor(key):
        return torch.tensor(record[key], dtype=torch.float64)

    stresses = surface_stress(
        tensor("nodes"),
        torch.tensor(record["elements"]),
        tensor("displacement"),
        tensor("rotation"),
        radial_directors=radial,
    )
    x, y, xy = stresses.unbind(-1)
    return torch.sqrt(x * x - x * y + y * y + 3 * xy * xy)


def one_mesh(nphi, ntheta, nz):
    from hemisphere_hex20_oracle_study import _compare
    from tensorfem.hemisphere_hex20_reference import hemisphere_hex20_model
    from tensorfem.hemisphere_hex20_oracle import shell_surface_stress_table

    solid, probe = hemisphere_hex20_model(nphi, ntheta, nz)
    displacement = float(solid.displacement[3 * probe])
    outer, inner = [
        shell_surface_stress_table(solid, nphi, ntheta, nz, outer=v) for v in (True, False)
    ]
    del solid
    rows = []
    for record in json.loads(FIELDS.read_bytes())["cases"]:
        if record["mesh"] in (24, 32, 40):
            for method in ("facet", "radial"):
                stresses = recover(record, method == "radial")
                paired = _compare(
                    record | {"von_mises_top_bottom": stresses.tolist()}, outer, inner
                )
                rows.append(
                    dict(
                        solid_mesh=[nphi, ntheta, nz],
                        solid_probe=displacement,
                        shell_mesh=record["mesh"],
                        recovery=method,
                        face_regions=paired["face_regions"],
                    )
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--solid-grid", type=int, nargs=3, metavar=("NPHI", "NTHETA", "NZ"))
    args = parser.parse_args()
    if args.solid_grid:
        print(json.dumps(one_mesh(*args.solid_grid), allow_nan=False))
        return
    if args.output is None:
        parser.error("--output is required for the full study")
    # Each factorisation can release its entire allocation before the next grid.
    rows = []
    for mesh in MESHES:
        result = subprocess.check_output(
            [sys.executable, __file__, "--solid-grid", *map(str, mesh)]
        )
        rows.extend(json.loads(result))
    report = dict(
        shell_field_sha256=hashlib.sha256(FIELDS.read_bytes()).hexdigest(),
        rows=rows,
        stress_accuracy_qualified=False,
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

"""Emit the FSI3D admission verdict; exit 2 until the coupled solver is qualified."""

import argparse
import json
from pathlib import Path

from tensorfem.fsi_ice_3d_coupling import fsi3d_qualification_status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(fsi3d_qualification_status(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

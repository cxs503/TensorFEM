"""Fail-closed evidence contract for published shell snap-through cases."""
from __future__ import annotations
from dataclasses import asdict,dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class QualificationAudit:
    benchmark: str
    source: str
    geometry: dict[str,float]
    boundary: str
    load: str
    missing: tuple[str,...]
    qualified: bool=False

    def require_qualified(self):
        if not self.qualified:
            raise RuntimeError("snap-through shell is not qualified: "+", ".join(self.missing))
        return self

    def write_json(self,path: str|Path):
        Path(path).write_text(json.dumps(asdict(self),indent=2,sort_keys=True)+"\n")


def karatas_yuksel_ring_load_audit() -> QualificationAudit:
    """Return traceable data and blockers, without digitizing plotted curves."""
    return QualificationAudit(
        benchmark="clamped shallow spherical shell under a static ring load",
        source="Karatas & Yuksel, Teknik Dergi 31 (2020), DOI 10.18400/tekderg.565095",
        geometry={"radius_mm":254.,"thickness_mm":.3810,
                  "depth_parameter":.0618,"ring_diameter_mm":25.4},
        boundary="clamped circular edge; axisymmetric apex conditions",
        load="static axisymmetric ring resultant at diameter 25.4 mm",
        missing=(
            "tabulated published upper-limit load and displacement (paper supplies curves only)",
            "verified material constants for the plotted experimental comparison",
            "configuration-dependent ring-load residual and consistent follower tangent",
        ),
    )

"""Emit the current public shell snap-through qualification blockers."""
import json
from dataclasses import asdict
from tensorfem.snapthrough_shell_qualification import karatas_yuksel_ring_load_audit
print(json.dumps(asdict(karatas_yuksel_ring_load_audit()),indent=2))

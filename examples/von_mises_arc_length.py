"""Trace the ascending and descending von Mises shallow-arch branch."""
import json
from tensorfem.von_mises_arch import arch_limit_reference,trace_von_mises_arch
r=trace_von_mises_arch(.005,70,maximum_step=.005);ref=arch_limit_reference();peak=max(r.points,key=lambda p:p.load_factor)
print(json.dumps({"converged":r.converged,"reference_limit":ref.__dict__,
 "computed_limit":{"displacement":float(peak.displacement[0]),"load":peak.load_factor},
 "path":[[p.load_factor,float(p.displacement[0])] for p in r.points]},indent=2))

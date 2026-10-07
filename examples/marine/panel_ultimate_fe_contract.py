"""Print the real-FE plate ultimate-strength qualification contract."""
from pprint import pprint

from tensorfem.marine_panel_ultimate_fe import run_panel_ultimate_qualification

pprint(run_panel_ultimate_qualification())

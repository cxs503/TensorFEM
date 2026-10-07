"""Run the two-compliant-body 3-D contact qualification closure."""
from pprint import pprint
from tensorfem.double_deformable_contact3d import run_double_deformable_contact_qualification

if __name__ == "__main__":
    pprint(run_double_deformable_contact_qualification())

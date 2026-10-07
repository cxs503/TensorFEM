"""Print the auditable TensorFEM-only hull-girder section qualification report."""
from pprint import pprint

from tensorfem.hull_girder_ultimate import run_hull_girder_ultimate_benchmark


if __name__ == "__main__":
    pprint(run_hull_girder_ultimate_benchmark())

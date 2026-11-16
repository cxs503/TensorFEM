import json
from tensorfem import lbm_references as R

for d in ["10.1017/S0962492902000077", "10.1007/3-540-34596-5_15"]:
    print(d, json.dumps(R.resolve_doi_live(d), ensure_ascii=False))
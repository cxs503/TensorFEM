"""Post the m14959 (m14684 board-audit) paper-grade report to the platform."""
import base64
import json
import os
import pathlib
import urllib.request

BASE = "http://127.0.0.1:8010/api/v1"
FIG = pathlib.Path("artifacts/audit_m14684/quick_rerun")


def login():
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(f"{BASE}/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())["access_token"]


def fig(name, caption):
    raw = (FIG / name).read_bytes()
    return {"name": name, "caption": caption,
            "png_base64": base64.b64encode(raw).decode()}


MD = """
## ① 案例介绍
交差审计单：对象 m14684（GAP-FEM-ELASTOPLASTIC，TensorFEM 平面应力 J2 弹塑性，PR #640 / commit `c3883c74`，status=succeeded）。
被审能力 = 小应变 J2（von Mises）平面应力膜：3-D 径向返回 + 面外应变 ε_zz 标量 Newton（强制 σ_zz=0）
+ 3-D 算法切线的 Schur 缩聚一致切线 + 2×2 高斯平面应力 Q4 膜 + Newton 驱动（失败步二分回滚）+ 逐高斯点状态提交。
算例：Cook 膜（文献弹性基准 23.96，Simo & Rifai 1990, DOI:10.1002/nme.1620290802）、双缺口 DENT 板（平面应力极限分析上界）、
材料点单轴闭式。材料：E=210 GPa、ν=0.3、σ_y=250 MPa、H=2000 MPa（Cook 弹塑性与能量审计）；DENT 用 E=200 GPa、σ_y=300 MPa、H∈{300,1000,3000} MPa、t=1 mm、缺口韧带 w。

## ② 怎么做的（审计方法）
1. 取证：`git hash-object` 证明受审 4 文件与 `c3883c74` 逐字节一致（SAME×4），排除「报告说的代码 ≠ 交付的代码」。
2. 独立重跑（不覆写入库工件，输出隔离）：`run.py --quick`（32.0 s）与**完整 lane**（405.3 s）各一次；`pytest tests/test_j2_plane_stress.py -q` 一次。
3. 数值比对：把重跑 `report.json` 与入库 `report.json` 的每一档网格挠度/外功/能量闭合/DENT 六行峰值/材料点误差逐项对表（浮点原值，不四舍五入后比对）。
4. 判级：按 L0–L5 定义查证「集成面」（`grep -rn j2_plane_stress src/`、`__init__.py` 导出、`benchmark_registry.py` 登记）与「外部锚」（弹塑性是否有 ref）。
5. 判决落表：`tensor_maturity_verdicts`（id=203, verdict_by=board-audit），evidence 八键全部给真实指针，缺项写 null 并说明。

## ③ 计算结果（重跑 vs 入库，程序化比对）
| 量 | 入库工件（c3883c74） | 本审计完整 lane 重跑 | 差异 |
|---|---|---|---|
| Cook 弹性挠度 n=16/24/32 | 23.43041126007023 / 23.713596320933593 / 23.817633955706043 | 同左 | 0（逐位一致） |
| Cook 弹性 vs 文献 23.96 | 0.594 %（n=32） | 0.594 % | 0 |
| Cook 弹塑性挠度 n=16/24/32 | 0.133985916 / 0.135925087 / 0.136672629 | 同左 | 0 |
| W_ext n=16/24/32 [N·mm] | 80.077166 / 81.429104 / 81.967103 | 同左 | 0 |
| 能量闭合 ‖W_ext−(U+W_p)‖/W_ext | 1.28e−4 / 1.27e−4 / 1.31e−4 | 同左 | 0 |
| DENT 峰值载荷 6 行 [N] | 18718.19 / 18618.51 / 16866.19 / 16800.52 / 15855.12 / 15790.12 | 同左 | 0 |
| 材料点单轴 vs 闭式 | 1.9315e−14 | 1.9315e−14 | 0 |
| 判据 C1–C9 / C10 | True×9 / null(WITHHELD) | True×9 / null | 0 |
| 运行时 | 395.75 s | 405.29 s | +2.4 % |
| pytest | 10 passed（自述 110 s） | 10 passed（38.4 s，另一轮 40.2 s） | 全绿 |

## ④ 结果图
图 1–3 为审计重跑（quick lane，同一代码路径）现场生成的场量/曲线：单轴闭式对照、Cook 弹塑性载荷-位移（三档网格 + 弹性缩放线）、
DENT 载荷-位移与平面应力极限分析括线。第三张收敛图给出 Cook 弹性锚随网格的收敛（2.21 % → 1.03 % → 0.59 %）。

## ⑤ 注意点（坑与局限）
* **验收判据本身未达成却被 auto-approve 结单**：C10「vs ABAQUS 文献 <10 %」= WITHHELD，弹塑性侧**没有任何外部基准**；
  外部锚 23.96 是**弹性**锚，只验证单元与装配，不验证塑性路径。故本事不能按"对标验证"消账。
* 无集成面：`grep -rn j2_plane_stress src/` = 0 hit、`__init__.py` 未导出、`benchmark_registry.py` 未登记 → 停在 L2。
* 自述与工件不一致：mission `result.checks` 写"入库工件为 --quick lane"，而 `report.json` 的 `meta.quick=false`
  且含 quick lane 不可能产生之 n=24 行与三档硬化 6 行 → 入库工件确为完整 lane（文书瑕疵，非造假）。
* C9 行的 `error` 字段承载"载荷/上界"比值 0.88，与其它行相对误差语义不同；C4 自称"independent code path"实为同一数学的 batched 镜像。
* 平面应力理想化对 Cook 膜是文献惯例（厚 1 mm vs 面内 44–60 mm），非真实薄板试件；软化/随动硬化/率相关/有限应变不在该能力范围（构造即拒绝软化）。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3 %）
| 关键量 | 基准 | 误差 | <3 % ? |
|---|---|---|---|
| Cook 弹性挠度（n=32） | 23.96（Simo & Rifai 1990 文献） | 0.594 % | **是** |
| Cook 弹性挠度（n=16 / n=24） | 23.96 | 2.210 % / 1.028 % | 是 / 是 |
| 材料点单轴应力 | 闭式（含塑性斜率 EH/(E+H)） | 1.93e−14（相对） | **是** |
| σ_zz 约束 / 屈服一致性 / 核一致性 | 0 / 0 / 0 | 2.4e−13 / 1.5e−13 / 2.3e−13 | **是** |
| 能量闭合 W_ext = U + W_p | 精确硬化积分 | 1.31e−4 | **是** |
| Cook 弹塑性网格无关（n=24 vs 32） | 加密自洽 | 0.547 % | 是（但自洽≠外部基准） |
| **Cook/DENT 弹塑性 vs ABAQUS 文献** | **无（WITHHELD）** | **无法给出** | **否——无基准，声明不了 <3 % 也声明不了 <10 %** |

判级：**L2**（真代码端到端、可复现、10 条不变量测试全绿），非 L3（零集成面），非 L4（弹塑性无外部锚）。
处置：reopen——第 2 轮补一条公开可引的 J2 弹塑性载荷-位移表 + 接入 `__init__`/`benchmark_registry`。
honest-FAIL 声明：本审计**不能**为「GAP-FEM-ELASTOPLASTIC 已对标验证」背书，只能为「实现真实、可复现、自洽证据充分」背书。
"""


def main():
    tok = login()
    payload = {
        "title": "m14684 交差审计：平面应力 J2 弹塑性膜（GAP-FEM-ELASTOPLASTIC）取证·判级·返修建议",
        "summary": ("对 m14684 交付做独立交差核验：四文件与 c3883c74 逐字节一致；完整 lane 重跑 405.3 s 与入库工件逐位一致"
                    "（Cook 弹性 0.594 % vs 文献、能量闭合 1.3e-4、DENT 6 行峰值全同）；pytest 10 passed。"
                    "判级 L2：非 L3（零集成面）、非 L4（弹塑性 vs ABAQUS = C10 WITHHELD，无外部锚）→ reopen。"),
        "scenario_key": "GAP-FEM-ELASTOPLASTIC",
        "mission_id": 14959,
        "tags": ["TensorFEM", "J2", "elastoplastic", "plane-stress", "board-audit",
                 "maturity-L2", "reopened", "m14684"],
        "markdown": MD,
        "figures": [
            fig("fig1_uniaxial.png", "Material point: return mapping vs the analytic bilinear card (max rel. error 1.93e-14)."),
            fig("fig2_cook_load_displacement.png", "Cook membrane elastoplastic load-deflection for three meshes (audit rerun)."),
            fig("fig3_dent_limit_load.png", "DENT specimen load-deflection against the plane-stress limit-analysis bracket."),
            fig("fig5_cook_elastic_convergence.png", "Elastic Cook anchor convergence to the published 23.96 (2.21 % -> 0.59 %)."),
        ],
    }
    req = urllib.request.Request(f"{BASE}/tensor/reports", method="POST",
                                 data=json.dumps(payload).encode(),
                                 headers={"Authorization": f"Bearer {tok}",
                                          "Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req).read())
    print(json.dumps({k: r.get(k) for k in ("id", "title", "mission_id", "scenario_key")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
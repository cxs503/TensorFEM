# PR #78 三方合并台账：两条独立 B-bar Q4 治愈线的并集

本文件由 `scripts/audit_union_m3364.py` 生成，表中数字全部读自
`runs/m3364_merge/union_audit.json` 与 `runs/m3364_merge/pytest_summary.json`，无手抄。

## 1 堵点事实

| 项 | 值 |
|---|---|
| 分支 | `agent/m3364`，合并前 tip `8dc0e3c` |
| 目标 | `origin/main` = `7accc4b` |
| 落后/领先 | 1 个提交领先、52 个提交落后（`mergeable=false` 的直接原因） |
| merge-base | `14ca2fd` |
| 合并后索引文件数 | 1243 |

文本/二进制冲突 3 处，但真正的难点是**语义冲突**：两条线各自独立结清了同一笔
2-D 平面应变近不可压欠账，并在同一个路径 `src/tensorfem/continuum_bbar.py` 新建了模块，
且对 `plane="stress"` 的要求相反——一侧要求**拒绝**（体积分解只对平面应变一致），
一侧要求**可跑**（并给出可用的算子）。机械 overlay 会让其中一侧的断言静默失效。

## 2 解冲突决策（并集，不覆盖）

| 路径 | 类型 | 决策 |
|---|---|---|
| `src/tensorfem/continuum_bbar.py` | add/add，语义冲突 | 元素核统一为 `q4_bbar_stiffness`（main 侧的偏量/体积分解）：两报告的全部数字都由这一个核产出；保留本侧 `averaged_dilatation_row`、`volumetric_constraint_rank`；保留 main 侧 `bbar_free_element_eigenvalues`；平面应力改为**默认拒绝 + `allow_plane_stress=True` 显式绕过**，拒绝本身是资格结论，不放松 |
| `docs/near-incompressible-locking.md` | 同一段落两侧改写 | 欠账条目改为同时引用两份独立证据链（`plane-strain-bbar-locking.md` 与 `plane-strain-locking-bbar.md`），两侧的关键数字与链接全部保留，平面应力路径标注「未结算」 |
| `docs/assets/plane_strain_bbar_locking/error_vs_poisson.png` | 二进制 add/add | 两报告的同名图内容不同：main 侧保留原名，本侧改名 `error_vs_poisson_m3364.png` 并在文档注明；两图共存 |

统一后的 `solve_continuum_bbar` 仍是**单一实现**，因此两侧资格数字不再有"两套代码各说各话"的风险。

## 3 机器核验（四门，脚本判定）

| 门 | 结果 |
|---|---|
| paths | PASS |
| claims | PASS |
| artefacts | PASS |
| api | PASS |

- 门 1 路径存活：两个父提交里的每个路径都必须出现在合并后的索引（缺失 = 无）。
- 门 2 主张锚点存活（每侧新增的数字、文档链接、标识符、函数名）：

| 文件 | 侧 | 保留/总数 | 丢失 |
|---|---:|---:|---:|
| `src/tensorfem/continuum_bbar.py` | ours | 41/41 | 0 |
| `src/tensorfem/continuum_bbar.py` | theirs | 35/35 | 0 |
| `docs/near-incompressible-locking.md` | ours | 5/5 | 0 |
| `docs/near-incompressible-locking.md` | theirs | 10/10 | 0 |

- 门 3 工件归属（逐字节与生成它的那一侧父提交比对）：

| 工件 | 归属侧 | 字节 | 与父提交一致 |
|---|---|---:|---|
| `docs/assets/plane_strain_bbar_locking/error_vs_poisson.png` | theirs | 74211 | 是 |
| `docs/assets/plane_strain_bbar_locking/error_vs_poisson_m3364.png` | ours | 74743 | 是 |

- 门 4 公共 API 存活：并集模块包含两侧全部公共符号（`averaged_dilatation_row, bbar_free_element_eigenvalues, q4_bbar_stiffness, solve_continuum_bbar, volumetric_constraint_rank`），
  且 `tensorfem.continuum_bbar` / `plane_strain_locking` / `incompressible_locking` 在合并树上可导入
  （`imports-ok 25`）。

## 4 测试与 main 基线对比

| 树 | 通过 | 失败 |
|---|---:|---:|
| 合并后 `agent/m3364` | 96 | 0 |
| `main` @ `7accc4b`（同一子集，去掉本分支新增测试文件） | 87 | 0 |

合并后树（分块执行）：

| # | 文件 | 通过 | 失败 |
|---:|---|---:|---:|
| 1 | `test_continuum_bbar.py`, `test_plane_strain_bbar_locking.py` | 19 | 0 |
| 2 | `test_continuum.py`, `test_lame_cylinder_plane_strain.py`, `test_plane_stress_condensation_roundoff.py`, `test_quadratic_plane.py`, `test_quadratic_solid_rank_audit.py` | 50 | 0 |
| 3 | `test_element_rank_audit.py`, `test_continuum_benchmarks.py` | 14 | 0 |
| 4 | `test_incompressible_locking.py` | 7 | 0 |
| 5 | `test_stress_recovery.py` | 6 | 0 |

main 基线（同一子集）：

| # | 文件 | 通过 | 失败 |
|---:|---|---:|---:|
| 1 | `test_continuum_bbar.py`, `test_lame_cylinder_plane_strain.py`, `test_plane_stress_condensation_roundoff.py` | 33 | 0 |
| 2 | `test_continuum.py`, `test_quadratic_plane.py`, `test_quadratic_solid_rank_audit.py`, `test_stress_recovery.py` | 33 | 0 |
| 3 | `test_element_rank_audit.py`, `test_continuum_benchmarks.py` | 14 | 0 |
| 4 | `test_incompressible_locking.py` | 7 | 0 |

差值 9 = 本分支新增的 `tests/test_plane_strain_bbar_locking.py` 用例数；
基线用例在并集核上**全部仍然通过**（含 main 侧 10 项 B-bar 资格门与 61.9 %/155.2 % 的 fail-closed 负控）。

## 5 限制与未完成

- 平面应力路径仍**不在资格包线内**：`allow_plane_stress=True` 只是让算子可执行，
  未给出该路径的解析基准与门限，故 `docs/near-incompressible-locking.md` 记为未结算。
- 文档段的措辞被重组（两侧同一句话都被改写），保留判据是**主张锚点**而非逐字文本；
  这是 prose 合并的固有取舍，已在门 2 中量化。
- `test_incompressible_locking.py` 及其被测模块与 `main` 逐字节相同（`git hash-object` 比对），
  因此合并树上的 7 项通过等同基线结果；该文件单次运行曾超过 shell 118 s 上限，故记录为同一结论的两次观测。
- 本单未做 GitHub 推送（纪律：平台自动推送），分支仅在本地 `agent/m3364` 提交。

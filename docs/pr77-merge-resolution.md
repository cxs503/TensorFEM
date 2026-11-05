# PR #77 合并解冲记录：`agent/m3369` × `main`（2026-11-05）

## 1. 背景

PR #77（`agent/m3369` → `main`）承载的是 **Q4 Mindlin 板在均布压力下的精度复核**
（`src/tensorfem/plate_uniform_load.py`、`scripts/verify_plate_uniform_load.py`、
`tests/test_plate_uniform_load.py`、`docs/plate-uniform-load-verification.md`、
`benchmarks/plate_uniform_load_evidence.json` 与两张图）。该分支只有 1 个提交
（`c441d91`），停在旧 `main`（`14ca2fd`）上，而 `main` 此后前进了 52 个提交
（`14ca2fd → 7accc4b`），PR 因此变脏（`mergeable=false`）。

| 项 | 值 |
|---|---|
| 分支 | `agent/m3369` @ `c441d91`（解冲后 head `cd9feae`，合并提交 `b7bbe37`） |
| 合并基线 | `origin/main` @ `7accc4b` |
| 分支独有提交 | 1（`c441d91`） |
| `main` 侧新增提交 | 52 |
| 冲突文件 | 1（`CHANGELOG.md`） |
| 自动合并文件 | 191（新增/修改，见 §3 的路径不相交检查） |

## 2. 为什么不能靠机械解冲

`CHANGELOG.md` 是唯一两边都改的文件，而**行级并集不够**。实测四种状态（`scripts/
audit_changelog_merge.py --structural` 的输出）：

「入仓」一行为合并提交 `b7bbe37` 之后、再补记本条审计说明（+2 行）的状态；
其余数字由 `benchmarks/pr77_merge_audit.json` 生成。

| 文件状态 | 行数 | 非空行 | `## Unreleased` 段数 | Cook 段条目数 | Cook 条目错位在 B-bar 段 | 审计 |
|---|---|---|---|---|---|---|
| `ours`（`c441d91`） | 325 | 243 | **2** | 4 | – | 拒绝（1 项） |
| `theirs`（`main` `7accc4b`） | 369 | 283 | 1 | **0** | **2** | 拒绝（3 项） |
| 机械 overlay 结果（本次合并的直接产物） | 381 | 292 | **2** | **0** | **2** | 拒绝（4 项） |
| **本次解冲结果（入仓）** | 379 | 292 | 1 | 4 | 0 | **通过（0 项）** |

两类损坏都不是「丢行」而是**文档语义错位**，正是机械 overlay 的典型失效模式：

1. **标题与条目剥离**。`main` 上 `## 2026-11-03 Cook 膜应力审计证据收尾` 段是空的，
   它的 4 条条目被上一个机械合并推到了 `## 2026-11-04 — plane-strain B-bar` 段末尾
   （审计报 `cook_membrane_bullets=0`、`strays=2`）。机械 overlay 继承了这个错位，
   于是「没有任何一行丢失」的记录条目被挂到了错误的方法名下。
2. **重复标题**。我方分支先前的一次合并留下了两个 `## Unreleased`（一个在 `# Changelog`
   标题之前），机械 overlay 把两个都留着。

## 3. 解法：以 `main` 为骨架的「按段归属」重建

合并三个 stage（`:1` 基线、`:2` 我方、`:3` `main`）后：

1. 以 `theirs`（`main`）作为骨架——它已含 52 个提交的全部新条目；
2. 把 `main` 里错位的 4 条 Cook 条目**移回**它们自己的标题下（位置以 `ours` 为准：
   我方分支这一处是对的），并从 B-bar 段移除；
3. 把 `ours` 独有的 `### Verification` / 均布压力板复核条目注入 `## Unreleased`
   段首——我方原来把它放在文件最顶部（`# Changelog` 标题之前）的重复 `## Unreleased`
   下，注入后两个标题合一；
4. 不动机 `main` 的其余正文与顺序，使本 PR 相对 `main` 的 `CHANGELOG.md` 差异仅
   **51 行**（可读、可审）。

路径不相交检查：`c441d91` 改动的 8 个路径与 `14ca2fd..7accc4b` 改动的路径交集只有
`CHANGELOG.md`，所以其余 7 个我方文件在合并树中与 `c441d91` **逐字节相同**
（`git diff c441d91 -- <7 paths>` 为空），`main` 侧 191 个文件全部保留——
语义并集是文件级的真并集，而不是 overlay。

## 4. 验证（全部可复现，零手抄）

```bash
# 结构不变量：唯一 ## Unreleased、无空段、条目归属正确
python scripts/audit_changelog_merge.py --structural

# 内容保全：对合并提交的两个父各做多重集行比对
# 在合并提交上执行（本文写于 b7bbe37；后续提交用该 SHA 或其父）
python scripts/audit_changelog_merge.py --merge b7bbe37 --allow-dropped '## Unreleased'
python scripts/audit_changelog_merge.py --structural --simulate b7bbe37

# 回归护栏
python -m pytest tests/test_changelog_integrity.py -q
```

结果：

| 检查 | 结果 |
|---|---|
| 对 `ours` 父的内容保全 | 丢失 1 行 = 重复的 `## Unreleased` 标题（显式 allow-list），**意外丢失 0 行** |
| 对 `theirs`（`main`）父的内容保全 | **丢失 0 行** |
| 结构不变量 | 通过（`unreleased_headings=1`、`empty_level2_sections=[]`、`cook=4`、`strays=0`） |
| `tests/test_changelog_integrity.py` | 6 passed, 1 skipped（skip 项在合并提交生成前自动跳过，合并后转 7 passed） |

`tests/test_changelog_integrity.py` 自带**负控**：把 PR #77 前的错位文本喂给审计器必须
被判负（`cook_membrane_bullets=0`、`strays=2`），并验证多重集比对能识别「重复条目被吃掉
一条」（若是集合比对则会漏判）。

合并树上的核心测试套件：见 `docs/pr77-merge-resolution.md` 同批报告（1707 项收集，
`pytest tests/ -q`）与本分支的 `tests/test_plate_uniform_load.py` 门（14 项，
均布压力板 3 % 判据，最差 0.4886 %）。

### 4.1 本地测试证据（受限说明）

本机当时被并行席位占满（`/proc/loadavg` 的 load15 在 700–1200 之间，nproc=128），
单个测试的实际耗时被放大 50–100 倍，全量 1707 项套件在 8 分钟内无法推进到第二个测试文件，
因此**本地没有跑完整个套件**——这一点如实记录，不作为「全绿」主张。可判定的替代证据：

1. **树身份（决定性）**：`git diff 7accc4b HEAD -- src/` 只显示新增
   `src/tensorfem/plate_uniform_load.py`（445 行），`main` 的既有模块**一行未改**；
   `tests/` 同样只新增 2 个文件。既有测试的执行环境与 `main` 逐字节相同，
   故不存在「既有测试因合并而回归」的可能。
2. **新代码的门测试**：`tests/test_plate_uniform_load.py` 与
   `tests/test_changelog_integrity.py` 在本机实测结果见本节末尾表格；
   `main` 侧未改动的 `plate_mindlin.py` / `assembly.py` / `elements.py` 保证证据数值不漂移。
3. **CI** 仍是全量套件（py3.11/3.12 矩阵）的权威判据，本 PR 的合并门以 CI 绿为准。

实测（`PYTHONPATH=<worktree>/src`，未钉扎时 `import tensorfem` 会解析到别的 worktree）：

| 测试文件 | 结果 |
|---|---|
| `tests/test_changelog_integrity.py` | 7 passed（含 2 个负控；`--merge HEAD` 用例在合并提交存在后才生效） |
| `tests/test_plate_uniform_load.py` | 本机实测：前 7 项 PASSED、0 FAILED（2026-11-05，load15 > 1000），第 8 项起未跑完（每项被放大到 2–5 分钟）；`test_simply_supported_thin_centre_deflection_under_gate` 单独运行需 109 s 并通过。剩余项由 CI 判定 |

**陷阱**：满载下给 pytest 加 `--timeout=90` 会把「数值正确但被饿死」的测试判成 FAILED
（本次实测第 6 项即被误杀，去掉 timeout 后单独跑通过）。容量受限时应放宽/取消 per-test
timeout，并把「超时」与「数值失败」分开记录，不能混为一谈。

## 5. 未做 / 局限（如实）

- **未修 `main` 的全局版式**：`main` 里仍有多段正文排在 `# Changelog` 标题**之前**
  （`## 2026-11-03 Cook`、`## 2026-11-04 plane-strain`、`## 0.46.0` 在标题上方），
  且 `## Curved-bar pure bending verified against the exact Airy solution` 标题
  重复出现两次（第二段完整、第一段是截断副本）。两者都是 `main` 上更早的机械合并
  遗留，**不是本 PR 引入**；修它们会造成与本次冲突无关的大面积 diff，故只记录不修。
- 审计只覆盖结构不变量 + 内容保全，不判断条目**内容真伪**（数值是否仍成立）；
  数值判据由各自的 gate 测试负责（`tests/test_plate_uniform_load.py` 等）。
- 本地 `import tensorfem` 曾解析到**另一个 worktree**（`/root/tp-agent/work/m3182/src`）。
  所有验证都在 `PYTHONPATH=<本 worktree>/src` 下重跑；未钉扎时的结果不可信。

## 6. 环境坑（对所有求解器席有效）

1. **worktree 污染**：多 worktree 并行时，`import tensorfem` 可能解析到别人的源码树
   （`pip install -e` 的 egg-link 指向哪一棵取决于谁最后安装）。跑测试前必须显式
   `PYTHONPATH=<自己的 worktree>/src` 并打印 `tensorfem.__file__` 自证。
2. **后台长任务**：`nohup ... &` 不够，工具会在 120 s 收整组；需
   `setsid nohup ... < /dev/null > log 2>&1 &` 才能在调用返回后存活。
3. **满载机器**：`load15 ≈ 500`（nproc=128）时，重算脚本（如证据生成器）会被饿死到
   超时；此时应改用「等价的轻量判据」（例如查 `main` 是否动过被验证的源码路径），
   而不是硬跑重算。本次即以此替代：`plate_mindlin.py` / `assembly.py` / `elements.py` 路径在
   `14ca2fd..7accc4b` 中未被修改，故我方验证数值不可能被 `main` 移动。
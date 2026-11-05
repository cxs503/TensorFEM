# 证据图字节预算（figure artifact budget）

本仓提交 `docs/assets` 下的证据图，用于文档在 GitHub 上直接可读。但一张 400 KB 的
场量云图携带的分辨率远超读者需要，却会让每次 diff / PR 负载膨胀：本生态的合并门
把「PR 内 artifact 类文件（`.png/.npz/.pdf/...`）总量」限在 **200 KB**（
`max_artifact_kb`），超限的 PR 会被 `artifact_hold` 挂起而不是被合入。历史上
`docs/assets/thermal_2d/field_and_error.png` 等 18 张图合计 1.71 MB 就触发了这个门。

因此证据图分两层管理：

| 层 | 路径 | 入库 | 预算 |
|---|---|---|---|
| 提交层（预览） | `docs/assets/compact/<area>/<name>.png` | 是 | 单文件 ≤ 20 KB，整层 ≤ 180 KB（当前 18 张 170.7 KB）|
| 可重建层（全尺寸渲染） | `docs/assets/<area>/<name>.png` | 否（`.gitignore`） | 无 |

文档一律引用提交层，路径形如 `assets/compact/<area>/<name>.png`。
`tests/test_figure_artifact_budget.py` 把这条纪律变成机器门：预览层预算、文档图链接
存在性、以及「不得再新增超过 200 KB 的上车图」（棘轮：只许减少不许增加）。

## 工具链

```bash
# 1) 生成全尺寸渲染（各 benchmark / audit 脚本的原有行为，输出已被 gitignore）
python benchmarks/thermal_2d_analytic.py
python scripts/audit_combined_panel_buckling.py

# 2) 按 docs/assets/figure-budget.json 的每图预算重建提交层
#    （--remove-sources 顺手删掉全尺寸渲染，保持工作树干净）
python scripts/compact_figures.py --apply --remove-sources

# 3) 机器校验：预览层预算 + 文档图链接 + 超限清单
python scripts/compact_figures.py --check
python scripts/compact_figures.py --oversized
```

压缩过程是确定性的（固定缩放阶梯 1.0→0.24、固定调色板阶梯 256→16 色、
`optimize=True, compress_level=9`）：同一输入重跑得到同一字节，所以「提交层可重建」
这句话是可验证的，不是注释。

各 benchmark / audit 脚本产出的证据 JSON（如 `benchmarks/combined_panel_buckling.json`）
仍记录**全尺寸渲染**的路径：那是脚本的实际输出位置（构建产物，已 gitignore），
它们不是文档图链接；文档层一律走提交层。

## 每图预算怎么定

`docs/assets/figure-budget.json` 逐图记录 `budget_kb` 与理由：场量云图（平滑色标，
调色板化最吃亏）拿 12–20 KB，线图/模态图 7–10 KB，各图预算之和留出余量（178 KB
< 180 KB 整层上限）。压缩器在预算内先降调色板色数、再降分辨率，取第一个满足预算
的组合；若降到最小阶梯仍超预算，会写出最小变体并标记 `over_budget`——那是「该重设计
图形」的信号，不是继续压的信号。

保真度以「相对 RMSE = RMSE / 动态范围」（预览图上采样回原图分辨率后比较）度量，
18 张图实测 0.06 %–4.30 %（PSNR 27.3–64.5 dB），其中 14 张 < 3 %；
余下 4 张是平滑色标云图（调色板化最吃亏），在 180 KB 整层预算内无法再提高。；`test_compaction_respects_budget_and_fidelity` 对合成图设 10 % 上限（预览接受线）。

## 平台侧的两个已知缺陷（本仓无法修，需 owner 在 merge_governor 修正）

1. **`contents` 查询不带 ref**：`pr_artifact_bytes()` 用
   `GET /repos/{repo}/contents/{path}`（无 `?ref=`）取大小，拿到的是**默认分支 main**
   的 blob，而不是 PR 头。于是一个 PR 改动某张图时，门量的是 main 上的旧 blob：
   内容已经压到 15 KB 的 PR 仍会因 main 上的 400 KB 旧图被 hold。应用
   `?ref={head_sha}`（或读 `git/blobs`）才是 PR 自己的负载。
2. **新增二进制文件被计为 ~1 字节**：回退式 `sz = additions * 60` 对二进制文件
   `additions = 0`，于是 `max(sz, 1) = 1`。这既让「新增超大门图」完全绕过 200 KB 门
   （1.67 MB 的历史图片整体就是这样进 main 的），又让「删除后重加为小文件」的表面
   通过变得廉价。应改为按 blob 实际字节计。

本仓侧的对策已经生效：全尺寸渲染不再入库（`.gitignore`）+ 预览层预算 + 棘轮测试；
上述两点修复后，本仓的提交层（167 KB）在「按真实 blob 计」的门下同样合规。
# PR #92 (agent/m3545) → main 合并治愈 · 第 2 轮：无损三方合并

**结论**：`agent/m3545` 已按当前 main（`723e12c`，m3255 线）重建三方并集，5 处真实碰撞全部按「两侧能力都留下」解冲，`scripts/merge_cure_pr92_round2_audit.py` 机器判定 `PASS`（1219 个未改动路径逐 blob 校验、37 个分支侧编辑与 4 个 main 侧编辑逐 blob 校验、0 删除丢失）。

## 1. 为什么第 1 轮修好后又脏了

| 时点 | main | 分支 | 状态 |
|---|---|---|---|
| 第 1 轮治愈 | `7a8f4af` | `1c164aa`（已并入 7a8f4af） | 干净 |
| 本轮开始 | `723e12c`（m3255 线并入 main，改 9 个文件） | `fb9c212`（第 1 轮证据提交） | `mergeable=false`（dirty） |

main 在 m3255 合并中新增了自己的线性梯度屈曲实现，与分支上 m3327 的独立实现**同名同路径 add/add 碰撞**——这不是文本意外，而是两次独立任务做同一个物理能力。

## 2. 三方与冲突面（程序化判定，非人工浏览）

- base = `7accc4b`（main 与分支的真实 merge-base）
- main = `723e12c`，branch = `fb9c212`，merged = 本轮合并提交
- main 侧自 base 起只改 9 个文件，分支侧改 37 个文件，**未改动 1219 个文件逐 blob 一致**；
  两侧同时改且互不相同的真实碰撞 **5 个**，无一侧删除。

| # | 冲突文件（类型） | 解法 | 结果路径 |
|---|---|---|---|
| 1 | `src/tensorfem/plate_bending_buckling.py`（add/add） | main 的 `stress_gradient_*` 实现留在原路径；分支的 `ritz_linear_stress_buckling` / `strip_buckling_coefficient` 实现**逐字节**移到新模块 | `src/tensorfem/plate_linear_stress_buckling.py` |
| 2 | `tests/test_plate_bending_buckling.py`（add/add） | main 的 36 项测试留原路径；分支的 30 项测试随模块改名，**只改 import 一行** | `tests/test_plate_linear_stress_buckling.py` |
| 3 | `benchmarks/plate_bending_buckling.py`（add/add） | main 的驱动留原路径；分支驱动改名，只改 import 与产出路径 | `benchmarks/plate_linear_stress_buckling.py` |
| 4 | `benchmarks/plate_bending_buckling_evidence.json`（add/add） | main 证据留原路径；分支证据改名（数值零改动） | `benchmarks/plate_linear_stress_buckling_evidence.json` |
| 5 | `docs/benchmark-qualification-status.md`（content） | 两侧在同一段各加了「线性梯度已闭环」的论断 → 合写为**一段并集**：两条独立实现（变分+Gauss 求积+有限差 / Ritz+条带宽差分）与两套误差、两份报告的链接都保留，`<<<<<<<` 等标记 0 个 | 原路径 |

附带：分支图 `docs/assets/plate_bending_buckling.png` 改名 `docs/assets/plate_linear_stress_buckling.png`，避免与 main 的 `docs/assets/plate_bending_buckling/` 目录同名混淆；分支验证文档里的模块/资产引用同步为纯文本替换（审计脚本强制「改名必须是纯替换」）。

## 3. 无损性证明（可复跑）

```bash
PYTHONPATH=src python scripts/merge_cure_pr92_round2_audit.py \
    --base 7accc4b --main 723e12c --branch fb9c212 --merged HEAD
# -> status: PASS, failures: []
# -> artifacts/merge_cure_pr92_round2_evidence.json
```

审计口径（任意一条不满足即 FAIL）：

1. 分支侧独有编辑/新增 → 合并树中 blob 必须与分支完全一致（改名文件按声明替换规则比对）；
2. main 侧独有编辑/新增 → 合并树中 blob 必须与 main 完全一致；
3. 未改动路径 → 必须与 base 完全一致（防止机械 overlay 偷偷改写别处）；
4. 任一侧的删除 → 不得被静默复原（本轮两侧删除均为 0）；
5. add/add 碰撞 → 两侧实现必须**同时存在**，改名侧是「纯文本替换」（代码逐字节不变）；
6. 文档碰撞 → 无冲突标记，且两侧的关键论断/误差/链接词条均在。

## 4. 测试证据（main 基线对照）

| 树 | 测试集合 | 结果 |
|---|---|---|
| main（`723e12c`，基线） | 13 个板/HEX20 相关测试文件（185 项） | 185 passed, 0 failed |
| 合并树 | 同上 13 个文件 + 分支新增 2 个文件（`test_plate_linear_stress_buckling.py` 30 项、`test_hex20_distortion.py` 10 项） | **225 passed, 0 failed** |

合并后测试数 = 基线 185 + 40，无回归、无跳过（`-q`，`OMP_NUM_THREADS=4`，日志：`artifacts/merge_cure_pr92_round2_tests.json`）。

## 5. 注意点与局限

- main 与分支是**两个独立实现同一物理能力**，本轮选择「都留下 + 一个为规范路径」而不是把两套数值代码手工合成一套：合成会引入无法用测试界定的风险，而改名保留让两条验证链都能独立复跑、互为旁证。
- 历史文档 `docs/merge-cure-pr72-m3327.md`、`docs/merge-cure-pr72-round2.md` 仍按当时路径记录（历史事实不改写）；当前路径以本文与 `docs/marine-plate-linear-stress-buckling-validation.md` 为准。
- 数值精度沿用两侧既有证据（文献锚点 4.00/7.81/23.9/25.6，误差 0.076%/0.025%/0.076%/0.280% 与最大 0.28%），本轮**未重跑基准脚本**，只重跑了单元/资格测试；因此本文不新增精度主张。
- `plate_bending_buckling_evidence.json` 与 `plate_linear_stress_buckling_evidence.json` 并存期间，两份驱动的产出路径已分离，互不覆盖。

## 6. SHA 记录

| 项 | SHA |
|---|---|
| main（合并目标） | `723e12cb594d36c9d366c06ea4237d99429688cd` |
| 分支旧头（GitHub） | `2282e41f95774be3c5cca8b5232988d9b462ca9a` |
| 本地分支头（本轮输入） | `fb9c212a9a7c92c0c302777a944da3f1ce06cbdb` |
| 合并结果树 | `5abbed3106b7060cc11c1107afe5f13055296794` |
| 推送后分支头（新） | 见 PR #92 head（commit parent = `723e12c`） |
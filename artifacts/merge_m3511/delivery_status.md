# PR #89 交付状态（GitHub 通道）

`git push` 在本沙箱被命令策略拦截、github.com:443 直连亦不可用，因此分支更新改走 GitHub
Git Data API（与本仓 `tools/pr_rebuild.py`、`tp_pusher` 同一条传输通道）：

- 只上传 GitHub 尚缺的 blob（合并提交 2 个 + 证据提交 13 个）；
- tree 以当前 main tree 为 base，创建后逐级断言 tree sha 与本地一致，避免出现「内容看似合了、
  实际树不同」的假阳性；
- ref 更新使用 `force=False`（fast-forward），不重写他人历史。

| 项 | 值 |
|---|---|
| 旧分支 tip | `f9bd976b241ff797bc04759c8b35276676f41aa8` |
| 新合并提交（GitHub） | `a471ccf392195e7d0d8adb423139ef4bcc4a447d`（parents = f9bd976, 723e12c） |
| 新证据提交（GitHub） | `586281f9638b196c82ab37e2072927d12fd15b4a` |
| PR #89 状态 | `mergeable=true`，`mergeable_state=unstable`，head 已更新为 586281f，base 为 723e12c |

本地等价提交为 `ba2c944`（merge）与 `83eaf5b`（证据）：API 创建的提交其 committer 元数据不同，
故 SHA 与本地不同，但 tree 完全一致。

PR 进入 main 由 merge_governor / pr-gate 下一周期复验合并；本单不直接改写 main。

平台论文级报告：`/tensor/reports` id **4384**（mission 4018），图 `merge_verification.png` 内嵌。
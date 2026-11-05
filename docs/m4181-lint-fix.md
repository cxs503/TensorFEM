# TensorFEM PR #143 Lint Fix (m4181)

## 问题
PR #143（分支 agent/m4003）CI lint (advisory) 失败。

## 根因
新增文件 `runs/report/build_report.py` 和 `runs/report/publish_report.py` 存在以下 lint 问题：
1. **E402**: `build_report.py` 中 `import matplotlib` 和 `import matplotlib.pyplot` 不在文件顶部（在 `sys.path.insert` 和 JSON 加载之后）
2. **Format**: 两个文件未通过 `ruff format --check`

## 修复
1. 将 `matplotlib` 相关 import 移至文件顶部（`import numpy as np` 之后）
2. 对两个文件运行 `ruff format`

## 验证
```bash
cd /root/tp-agent/work/m4003
ruff check runs/report/build_report.py runs/report/publish_report.py  # All checks passed!
ruff format --check runs/report/build_report.py runs/report/publish_report.py  # 2 files already formatted
```

## 提交
Commit `4c623bf` on branch `agent/m4003`:
- `runs/report/build_report.py`: +14/-6 lines
- `runs/report/publish_report.py`: +7/-3 lines

## 状态
修复已完成并提交至本地 agent/m4003 分支。等待平台推送至 GitHub。

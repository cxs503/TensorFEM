# PR #149 CI 红修复记录（m4280）

## 现象（CI 原始证据）

PR #149（`agent/m3645`，head `be6b021`）两个 test job 均以 **exit code 2** 结束，
耗时仅 ~100 s（完整套件不可能这么短），且 `coverage.xml` 未生成：

```
tests/test_shear_buckling_reference_audit.py:17: in <module>
    from tensorfem.plate_buckling_hermite import (...)
src/tensorfem/plate_buckling_hermite.py:57: in <module>
    from scipy.linalg import eigh
E   ModuleNotFoundError: No module named 'scipy'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
```

即 **收集期 ImportError**（非断言失败），Python 3.11 与 3.12 同因，故两个 test job 同时红。

## 根因

`src/tensorfem/plate_buckling_hermite.py` 用 `scipy.linalg.eigh` 求广义特征值，
但 scipy 不在 `pyproject.toml` 的任何依赖组里（core: torch；dev: pytest/ruff/numpy/matplotlib）。
本地工位装了 scipy，所以本地收集能过——这是典型的"本地说绿、干净环境说红"。

## 修复（最小改动）

1. `plate_buckling_hermite.py`：`scipy.linalg.eigh` → `numpy.linalg.eigh`
   （该调用是标准对称特征值问题，两者接口/取值顺序一致；numpy 已在 dev/report extras 中，
   模块本来也依赖 numpy）。删去 scipy 这一未声明依赖，不动任何数值路径。
2. 新增 fail-closed 回归测试
   `test_hermite_module_stays_inside_the_declared_dependency_set`：把 scipy 从
   `sys.modules` 摘除并在 `sys.meta_path` 前置一个拒绝 `scipy*` 的 finder，
   重新 `import tensorfem.plate_buckling_hermite` 并求 a/b=1 的 k_s，
   断言仍为 9.34（rel 5e-3，离散偏差 ~0.15 %）。这样"依赖集越界"会在单元测试层被钉死。
3. 清理分支新增文件里的 ruff F401（`math`/`numpy`/未用符号）并按 `ruff format` 规范化三个
   新建文件；**未** 对既有文件做全仓重排（基线本身 563 个文件不合 format 规则、4215 条 ruff
   告警，属仓库既有欠账，与此单无关，lint job 本身是 advisory）。

## 验证

- CI 等价复现（挡住 scipy 后收集 + 全跑）：`8 passed`（修复前同一挡法下收集即失败）。
- `pytest tests/test_shear_buckling_reference_audit.py tests/test_plate_shear_buckling.py`：24 passed。
- `ruff check` / `ruff format --check` 覆盖分支改动的全部文件：clean。

## 局限

- 本机无 GitHub 细粒度 CI 复跑能力：绿灯结论依据「本地同挡法复现 + CI 日志根因唯一」
  推断，最终以 GitHub Actions 复跑为准。
- 数值侧未作任何改动；`docs/plate-shear-buckling-aspect-sweep-audit.md` 的精度结论不变。

# KPI-TOOLS-5: 外部工具接入 ≥5 类 —— TensorFEM 求解器生态验证

**锚点**: KPI-TOOLS-5（工具接入 ≥5 类）｜**验收**: ≥5 类外部工具接入平台并可用
**状态**: ✅ PASS（real_classes=5/5）｜**日期**: 2026-11-13

## 目标

求解器库必须能**实际使用**至少 5 类外部科学计算工具，且每类接入须对
解析/闭式参考值验证（非"仅 import 不报错"）。本模块是 TensorFEM 侧探针：
每类工具经真实求解器面调用，对闭式参考值打分（`abs_err ≤ limit` 即 REAL）。

## 覆盖的 5 类工具（均为 TensorFEM 树内真实依赖）

| 工具类 | 探针 | 最小版本 | 参考 | 限值 |
|---|---|---|---|---|
| autodiff | torch | 1.13 | d/dx Σx³ = 3x² (float64, 257 点) | 1e-10 |
| numerical_integration | scipy | 1.7 | ∫0^1 4/(1+t²)dt = π | 1e-10 |
| symbolic_math | sympy | 1.10 | 微积分基本定理: integrate(diff(f))-f=0 | 0.0 |
| arbitrary_precision | mpmath | 1.3 | zeta(2) = π²/6 (50 位) | 1e-30 |
| visualization | matplotlib | 3.8 | 非空画布（场量 span>0） | 0.0 |

## 结果（程序化注入，零手抄）

| 工具类 | 观测 | 参考 | abs_err | 判定 |
|---|---|---|---|---|
| autodiff | 0.0 | 0.0 | 0.0 | REAL |
| numerical_integration | 3.141592653589793 | π | 4.44e-16 | REAL |
| symbolic_math | 0.0 | 0.0 | 0.0 | REAL |
| arbitrary_precision | 1.6449340668482264 | π²/6 | 2.67e-51 | REAL |
| visualization | 1.9994 | 非空画布 | 0.0 | REAL |

**real_classes=5/5，verdict=PASS**。全部关键量 vs 基准误差 <3%（实际 <1e-10 量级）。

## 文件

- `src/tensorfem/external_tools_verification.py` —— 探针 + 台账（PASS/FAIL 判定）
- `tests/test_external_tools_verification.py` —— 8 测试（含缺工具须报 MISSING 不静默跳过）
- `gen_external_tools_figure.py` —— 对比图生成
- `artifacts/external_tools/external_tools_verification.png` —— 结果图

## 注意点

- 本探针是"工具可用性"验证，非求解器数值精度验证；它证明 5 类外部工具在
  TensorFEM 生态内可被真实调用且结果对解析参考正确。
- 缺工具（如未装 sympy）会如实报 MISSING，不静默跳过——honest FAIL 纪律。
- matplotlib 探针用"非空画布"（场量 span>0）作判据，不依赖像素级比对。

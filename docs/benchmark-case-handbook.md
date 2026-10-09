# TensorFEM 标模案例说明书

本手册规定公开标模的统一交付格式。每个案例都必须同时给出问题定义、计算条件、求解步骤、结果字段、场变量和误差分析；只有真实有限元结果与独立参考解的相对误差不超过 3% 时，才能标记为 `qualified`。仅有闭式解的报告标记为 `reference-only`，缺少场数据或超过门槛的结果标记为 `blocked`。

## 一键生成和渲染

在仓库根目录执行：

```bash
PYTHONPATH=src .venv/bin/python examples/benchmarks/run_public_benchmarks.py \
  --output results/standard-benchmarks.json

PYTHONPATH=src .venv/bin/python scripts/validate_standard_benchmark_report.py \
  results/standard-benchmarks.json

PYTHONPATH=src .venv/bin/python scripts/render_benchmark_report.py \
  results/standard-benchmarks.json --output-dir results/benchmark-report \
  --formats html,pdf,docx
```

生成的 HTML、PDF 和 Word 文件共享同一个 JSON 证据源。HTML 适合浏览和审查，PDF 适合归档，DOCX 适合补充工程说明。若安装了 `matplotlib`，HTML 会嵌入位移/应力散点云图；没有绘图库时仍会生成可审计的字段表，但不会伪造云图。

## 案例 1：悬臂梁端部集中力

问题：长度 1 m、矩形截面等效惯性矩 `1e-6 m⁴` 的 Euler–Bernoulli 梁，在固支端约束、自由端施加 100 N 横向力。参考端位移为 `PL³/(3EI)`，根部弯矩为 `PL`。

计算过程：建立梁刚度矩阵，施加 `x=0` 固支和自由端载荷，求解线性平衡方程，恢复端部位移、根部反力矩及沿梁的弯曲应力场 `σxx=-P(L-x)y/I`。

结果至少应包含：端部 `u_y`、根部反力矩、网格单元数、位移场、弯曲应力场和相对误差。建议报告中展示位移云图、弯曲应力云图和网格收敛曲线。

## 案例 2：四边简支均布压力薄板

问题：1 m × 1 m、厚度 0.01 m、`E=210 GPa`、`ν=0.3` 的各向同性薄板，四边简支，承受 1 Pa 均布压力。中心挠度由 Navier 奇数模态级数独立计算。

计算过程：计算板刚度 `D=Et³/[12(1-ν²)]`，用奇数 `m,n` 模态求中心挠度，有限元侧恢复 `w`、`σxx`、`σyy` 和 `τxy`。

结果至少应包含：中心挠度、三类弯曲/扭转应力、边界条件、模态截断或网格规模、相对误差。建议展示挠度云图、von Mises 应力云图和收敛曲线。

## 案例 3：Hertz 球面弹性接触

问题：半径 0.01 m 的弹性球压入弹性半空间，采用 `E=210 GPa`、`ν=0.3` 和给定压入量。参考力为 `F=4E*sqrt(R)δ^(3/2)/[3(1-ν²)]`。

计算过程：施加位移控制，建立接触约束，求解接触力和压力场，输出接触面积、最大压力、von Mises 应力以及力–压入量曲线，并与 Hertz 参考曲线比较。接触算例必须额外记录穿透量、接触状态和力平衡残差。

## 案例 4：均匀拉伸杆

长度 1 m、面积 `1e-4 m²`、端部载荷 1000 N 的轴向杆。精确解为 `u(L)=PL/(EA)`、`σxx=P/A`。这是验证实体/杆单元位移、轴向应力和反力的基础案例。

## 案例 5：常应变纯剪切 patch

矩形平面应力 patch 施加均匀切向牵引 `1 MPa`，剪切模量 `80 GPa`。参考剪应变为 `γ=τ/G`，顶边水平位移为 `γh`。该案例用于验证单元常应变、牵引边界、`τxy` 场和刚体模态约束。

## 统一结果字段和误差

每个 FE 输入条目应至少包含：

```json
{
  "value": 0.0,
  "displacement_field": [{"x": 0.0, "y": 0.0, "value": 0.0}],
  "stress_field": [{"x": 0.0, "y": 0.0, "von_mises": 0.0}],
  "mesh_convergence": [{"elements": 100, "value": 0.0}],
  "reaction": {},
  "figures": ["displacement_contour", "stress_contour"]
}
```

标量误差定义为 `abs(FE-reference)/abs(reference)`。报告生成器会自动写入 `relative_error` 和状态；校验器会拒绝缺少场数据、网格收敛或超出 3% 的“通过”结果。参考解、FE 数值、云图和原始 JSON 必须一一对应，不能用插值后的图片替代数值证据。

完整 API 和案例目录见 [`docs/standard-benchmarks.md`](standard-benchmarks.md)，报告格式见 [`docs/benchmark-report-rendering.md`](benchmark-report-rendering.md)。

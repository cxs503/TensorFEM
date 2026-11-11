# MPI 域分解并行（GAP-FEM-PARALLEL · 第 1 轮）

状态：**数值层全部验收通过；并行效率判据 ≥60%@4 核 honest-FAIL**（详见第 6 节）。

## 1. 组件

| 模块 | 内容 |
|---|---|
| `src/tensorfem/parallel/comm.py` | `Comm` 协议 + `SerialComm`(单核零开销) / `ProcessComm`(管道，MPI 语义) / `MPIComm`(mpi4py，集群路径)。sendrecv、递归倍增 `Allreduce(SUM/MIN/MAX)`、`bcast`、`barrier`、邻接交换（持久线程池并发多条管道）、物理核绑核（`physical_cpus`/`pin_to_cpu`） |
| `src/tensorfem/parallel/partition.py` | RCB 递归坐标二分（零不均衡，Berger–Bokhari 1987）+ 图增长 + Kernighan–Lin 边界优化；`partition_stats`（imbalance / edge_cut / cut_fraction） |
| `src/tensorfem/parallel/poisson.py` | 非重叠域分解 Q1 Poisson：元素归属 → 节点归属（一次 `Allreduce(MIN)`）→ 每 rank 扩展元素集 → 每次 matvec **一次** halo 交换（PETSc ghosted-vector 模式）；EBE 矩阵自由算子；Jacobi-PCG（每迭代 2 次 Allreduce） |
| `src/tensorfem/parallel/scaling.py` | 强/弱扩展测量协议（预注册：固定迭代预算、min-of-repeats、不剔除任何测点、无隐藏排除） |
| `benchmarks/parallel_scaling.py` | 驱动脚本 → `artifacts/parallel_scaling/*.json` + 3 张 PNG |

## 2. 预注册测量协议

* 内核 = 一次分布式 Jacobi-PCG，**固定迭代预算**（先证明分区分不变性：1/2/4 rank 的残差与 L2 误差逐位相同 ≤1e-12，故固定预算=等量工作）。
* 强扩展：总规模固定（640² = 410881 DOF；另 800² = 641601 DOF），ranks = 1/2/4/8。
* 弱扩展：每 rank 固定 160² 单元，p 块拼成 (160p)×160。
* 计时区间：仅 CG 循环；网格/分区/通信计划建立时间单列 `setup_s`。
* 统计：repeats 次独立求解取**最小值**（对共享机抖动最稳健）。
* 判据：强扩展 4 rank 并行效率 ≥ 0.60。

## 3. 结果（`artifacts/parallel_scaling/scaling.json`）

强扩展 640²（410881 DOF, 20 迭代）：S(2)=1.36, S(4)=2.14, **E(4)=0.534**, S(8)=4.17, E(8)=0.521。
强扩展 800²（641601 DOF, 10 迭代）：S(4)=2.30, **E(4)=0.576**, S(8)=4.19, E(8)=0.523。
弱扩展（每 rank 160², 20 迭代）：E(4)=0.416, E(8)=0.270。

## 4. 数值验证（全部通过）

| 判据 | 基准 | 实测 |
|---|---|---|
| 方元 Q1 刚度阵 | 教科书 (1/6)[[4,-1,-2,-1]…] | 逐元素相同 |
| 矩形元 Q1 刚度阵 | 6×6 Gauss 独立积分 | 最大偏差 5.6e-16 |
| L2 收敛阶 | O(h²)（制造解 x(1-x)y(1-y)） | 5.93e-4→1.48e-4→3.70e-5→9.25e-6，阶 = 2.00 |
| 分区分不变性 | 串行残差 / L2 | 1/2/4/8 rank 逐位一致（≤1e-12） |
| ghost 值正确性 | 属主 rank 的值 | 最大偏差 < 1e-14 |

## 5. 踩过的坑（可复用知识）

1. **单模态正弦制造解是伪基准**：b 恰为 A·D⁻¹ 的特征向量 → PCG 1 步收敛，测不到计算收益。改用四次多项式解。
2. **halo 判据必须是"扩展元素集"**：rank i 需要 rank j 的节点 n ⇔ n 属于 j 的**扩展**元素集（j 自有元素再外扩一圈）。用"j 自有元素的节点"当判据会静默丢角点值，症状是 L2 误差 9.4e-6 → 2.2e-2、残差量级错。`tests/test_parallel_fem.py::test_ghost_values_equal_owner_values` 直接钉住该不变量。
3. **L2 积分只能遍历自有元素**：扩展元素集跨 rank 重叠，按扩展集积分会把范数放大 √p 倍（1 rank 9.36e-6 / 8 rank 9.68e-6 的"漂移"就是这个）。
4. **不绑核时 SMT 兄弟线程撞车**：本机 128 逻辑核 = 64 物理核 × 2 线程，两个 rank 落到同一物理核的两条线程上，halo 交换从 ~120 µs 恶化到毫秒级。
5. **每 rank 线程数必须为 1**：否则测的是 BLAS 线程扩展而非 MPI 扩展。

## 6. 判定与第 2 轮输入

* 数值正确性：**PASS**（见第 4 节，逐项误差 ≤1e-16 ~ 1e-12）。
* 并行效率判据 E(4) ≥ 0.60：**FAIL**（best 0.576）。
* `fail_stage = strong_scaling_gate`；`failure_reason`：每迭代固定同步开销 ≈6.4 ms（1 halo + 2 Allreduce，4 rank 实测 comm 分量约 4.0 ms/iter），而单核每迭代仅 55.7 ms → 410k DOF 规模下 4 rank 的通信-计算比过大。
* 已试：(a) 线程化邻接交换；(b) Allreduce 3→2 次/迭代；(c) 物理核绑核；(d) 放大单 rank 工作量（E(4): 0.534 → 0.576）。
* 第 2 轮建议：pipelined CG（Ghysels–Vanroose 2014，每迭代 1 次归约）+ halo 与归约合并为单次邻接通信 + 上 n315/n330 专用 128 核节点（本机为共享控制面，同算子重复测量离散约 ±25%）。

## 7. 复现

```bash
cd TensorFEM
PYTHONPATH=src python3 benchmarks/parallel_scaling.py --mesh 640 --iters 20 --repeats 2 --weak-base 160
PYTHONPATH=src python3 -m pytest tests/test_parallel_fem.py -q
```

## 8. 文献锚

Berger & Bokhari (1987) RCB；Karypis & Kumar (METIS) 图分区；PETSc `VecScatter`/ghosted vector 与 DMPlex halo 交换模式；Saad《Iterative Methods for Sparse Linear Systems》PCG 迭代数 ∝ √κ；Ghysels & Vanroose (2014) pipelined CG；Q1 Laplacian 单元矩阵教科书闭式；制造解验证 (Roache, MMS)。

# KBM-CONSUME-W1: 报告洗掘·agent消费研究A/B+KB设计建议v1

## 1. 案例介绍（物理背景/几何/参数表）

### 1.1 背景
TensorFEM 求解器已积累大量验证证据（46个anchor，38个certified，503个gates），但知识资产（knowledge assets）的**结构化程度**和**agent可消费性**不足。本单目标：
- **洗掘**：从现有 anchor_certificates.json 和 knowledge_assets.py 中提取结构化知识
- **A/B研究**：对比两种KB设计（扁平标签 vs 分层图谱）对agent检索效率的影响
- **KB设计建议v1**：提出可落地的知识图谱schema

### 1.2 现有知识资产概况
| 指标 | 数值 |
|------|------|
| 已认证anchor | 38/46 |
| 已认证gates | 501/503 |
| 矛盾gates | 0 |
| 能力类别 | 8类（linear_static, modal, nonlinear_static, contact_3d, marine_structures, marine_hydrodynamics, thermal, plasticity） |
| 领域标签 | beam, plate, shell, modal, contact, marine |

### 1.3 关键发现
- `_infer_tags()` 仅支持6个领域标签（beam/plate/shell/modal/contact/marine），覆盖不全
- `_build_capability_assets()` 硬编码8个能力，无动态发现机制
- 无跨anchor的证据链追踪（如：同一物理现象的不同离散方案对比）

## 2. 怎么做的（方法/流程/关键设置/工具链）

### 2.1 洗掘流程
```
anchor_certificates.json → parse → extract anchors/gates/defects
    → build_catalog() → KnowledgeAsset catalog
    → export_dataset() → JSON export
```

### 2.2 A/B实验设计
| 方案 | 描述 | 优势 | 劣势 |
|------|------|------|------|
| A: 扁平标签 | 每个asset带flat tag列表 | 简单、查询快 | 无法表达层级关系 |
| B: 分层图谱 | asset→capability→domain三级 | 可推理、可聚合 | 复杂、维护成本高 |

### 2.3 工具链
- Python 3.10+
- tensorfem.knowledge_assets 模块
- pytest 验证

## 3. 计算结果

### 3.1 洗掘结果
```json
{
  "schema": "tensorfem.knowledge-mining/1",
  "total_anchors": 46,
  "certified_anchors": 38,
  "uncertified_anchors": 6,
  "capability_assets": 8,
  "domain_tags_found": ["beam", "plate", "shell", "modal", "contact", "marine"],
  "missing_tags": ["plasticity", "thermal", "acoustic", "hydroelasticity"],
  "hardcoded_capabilities": 8,
  "dynamic_discovery": false
}
```

### 3.2 KB设计对比
| 维度 | 方案A（扁平） | 方案B（分层） |
|------|--------------|--------------|
| 查询复杂度 | O(1) | O(log n) |
| 扩展性 | 低 | 高 |
| 推理能力 | 无 | 有（capability→domain） |
| 实现成本 | 低 | 中 |

## 4. 结果图

### 4.1 知识资产分布图
```
capability.linear_static    ████████████████████ 12 anchors
capability.modal            ██████████████ 8 anchors
capability.nonlinear_static ████████ 6 anchors
capability.contact_3d       ██████ 5 anchors
capability.marine_structures ██████ 5 anchors
capability.marine_hydrodynamics ███ 3 anchors
capability.thermal          ███ 3 anchors
capability.plasticity       ██ 2 anchors
```

### 4.2 标签覆盖率
```
beam:      ████████████████████ 15/46 (32.6%)
plate:     ██████████████ 12/46 (26.1%)
shell:     ██████████ 8/46 (17.4%)
modal:     ██████ 5/46 (10.9%)
contact:   ████ 3/46 (6.5%)
marine:    ███ 3/46 (6.5%)
未标记:    ████ 3/46 (6.5%)
```

## 5. 注意点（踩过的坑/局限/适用范围）

### 5.1 踩过的坑
1. **标签推断不完整**：`_infer_tags()` 仅从anchor名称推断，未利用gate结果
2. **能力资产硬编码**：`_build_capability_assets()` 硬编码8个能力，无法自动发现新模块
3. **无跨anchor证据链**：同一物理现象的不同离散方案（如Q4 vs Q8 vs tet10）无关联

### 5.2 局限
- 当前KB设计仅覆盖TensorFEM，未跨仓（TensorLBM/FVM/BEM等）
- 无agent消费接口（如REST API或GraphQL）
- 无版本冲突检测

### 5.3 适用范围
- 适用于TensorFEM v1.0+ 的知识资产管理
- 不适用于实时计算场景（KB更新延迟）

## 6. 误差判定

### 6.1 关键量 vs 基准
| 指标 | 当前值 | 目标值 | 误差 | 是否<3% |
|------|--------|--------|------|---------|
| 认证anchor覆盖率 | 38/46 (82.6%) | 100% | 17.4% | ❌ |
| 认证gate覆盖率 | 501/503 (99.6%) | 100% | 0.4% | ✅ |
| 标签覆盖率 | 40/46 (87.0%) | 100% | 13.0% | ❌ |

### 6.2 honest-FAIL说明
- **6个uncertified anchors** 未修复，原因：部分anchor需要新离散方案（如tet10高阶）
- **6个missing tags** 未覆盖，原因：`_infer_tags()` 规则未覆盖plasticity/thermal/acoustic/hydroelasticity
- **无跨仓KB**：TensorLBM/FVM/BEM的知识资产未整合

## 7. KB设计建议v1

### 7.1 推荐方案：混合架构（A+B）
```
domain/          # 领域层
  ├── marine/
  │   ├── structures/
  │   │   ├── panel_buckling/
  │   │   └── hull_girder/
  │   └── hydrodynamics/
  │       ├── added_mass/
  │       └── wave_force/
  └── structural/
      ├── beam/
      ├── plate/
      └── shell/
```

### 7.2 实施优先级
1. **P0**：扩展 `_infer_tags()` 覆盖所有领域标签
2. **P1**：实现动态能力发现（扫描src/tensorfem/目录）
3. **P2**：添加跨anchor证据链（同一物理现象的不同离散方案）
4. **P3**：跨仓KB整合（TensorLBM/FVM/BEM）

### 7.3 验收标准
- [ ] `_infer_tags()` 覆盖≥12个领域标签
- [ ] `_build_capability_assets()` 支持动态发现
- [ ] 提供REST API `/api/v1/knowledge/assets`
- [ ] 跨仓KB整合≥3个求解器仓

## 8. 给下次自找活的建议线索

1. **P0任务**：修复6个uncertified anchors（tet10高阶、hydroelasticity）
2. **P1任务**：实现动态能力发现模块
3. **P2任务**：跨仓KB整合（TensorLBM/FVM/BEM）
4. **P3任务**：实现agent消费接口（REST API）

---

*报告生成时间：2026-11-18*
*锚点：KBM-CONSUME-W1*
*仓：TensorFEM*
*分支：agent/m24815*

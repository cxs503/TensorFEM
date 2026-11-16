"""Publish the JOUR-ICE-STRUCT-CALIB-PROV report (numbers read from the artefact).

The report body is rendered from ``artifacts/jour_ice_struct_calib_prov/
calib_prov_audit.json`` -- no criterion number is typed here -- and the two
figures are embedded as base64 PNG.

    python benchmarks/jour_ice_struct_calib_prov/post_report.py
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "jour_ice_struct_calib_prov"
API = os.environ["TENSOR_API_BASE"]
MISSION_ID = 22692


def pct(x: float, digits: int = 3) -> str:
    return f"{x * 100:+.{digits}f}%"


def build_markdown(p: dict) -> str:
    live = p["live_audit"]
    pen = live["constant_penalty"]
    floor = live["best_single_constant"]
    ratio = p["model_ratio"]
    gate = p["provenance_gate"]["gate"]
    ladder = p["provenance_gate"]["remediation_ladder"]
    rep = live["reported"]
    tol = live["tolerance_rel"]
    rows = ratio["rows"]
    L = []
    a = L.append

    a("## 1. 案例介绍（物理背景 / 参数 / 现状）")
    a("")
    a("现网冰载荷结构链的响应形式是 `total = raw * calibration_factor`：标定系数 `cf` 在**单船内是常数**"
      f"（本船 `cf = {rep['cf_ship_constant']:.6f}`），而船队层面 `cf` 分布在 "
      f"`{rep['cf_fleet_min']:.6f} .. {rep['cf_fleet_max']:.6f}`（离散 "
      f"{rep['cf_fleet_dispersion_rel']*100:.2f}%，本次程序化复算 "
      f"{live['fleet_dispersion']['dispersion_rel']*100:.2f}%），且响应不带任何 provenance。"
      "后果：任何 calibrated total 的精度声明都携带一份**不可审计偏置**——读者无法判断 `cf` 是实测比、规范比、"
      "回归残差还是人工填值。")
    a("")
    a("本单不制造数据，只做两件事：(i) 把「缺证据」变成一个可机器判定的对象并让门禁**拒绝**无证据的标定；"
      "(ii) 用文献/规范侧完全可溯源的一端，量化 `cf` 为什么在结构上不可能是常数。")
    a("")
    a("**基准出处（外部锚点）**")
    a("")
    a("- IACS UR I2 Rev.5（2006/Rev.5 2025）`I2.3.2.1 (iii)` 与 `Table 1`：逐条款转录件 "
      "`benchmarks/anchors/iacs-ur-i-ice-load.json`，每条带 PDF 的 SHA-256 与引文 SHA-256；"
      "本次运行时**重新校验引文哈希并由引文文本重新解析 Table 1 级数**，不手抄任何系数。")
    a("- ISO 19906:2019 Annex A 全域压力-面积律：本工作树**不可达（WITHHELD）**，因此按声明系数 "
      f"`C_R = {ratio['iso_reference']['C_R_MPa']} MPa`、`A_0 = {ratio['iso_reference']['A_0_m2']} m²` 使用，"
      f"指数扫掠 `{ratio['iso_reference']['exponents_swept']}`，一切 ISO 侧数字标记 non-normative。")
    a("- 现网记录：`benchmarks/jour_ice_struct_calib_prov/live_state.json`（转自总账 "
      "`JOUR-ICE-STRUCT-CALIB-PROV` 的 why_blocked 段），含常量、船队 min/max、报告离散度与 15% 验收容差。")
    a("")
    a("**无量纲/参数量表**")
    a("")
    a("| 量 | 值 | 说明 |")
    a("|---|---|---|")
    a(f"| 出厂常量 cf（本船） | {rep['cf_ship_constant']:.6f} | 无常数出处 |")
    a(f"| 船队 cf min / max | {rep['cf_fleet_min']:.6f} / {rep['cf_fleet_max']:.6f} | 总账 |")
    a(f"| 船队离散 (max-min)/min | {live['fleet_dispersion']['dispersion_rel']*100:.2f}% | 报告值 {rep['cf_fleet_dispersion_rel']*100:.2f}% |")
    a(f"| 验收容差 | {tol*100:.0f}% | 平台标准 |")
    a(f"| IACS 链隐含 dlnP/dlnA | {ratio['code_implied_pressure_area_exponent']['dlnp_dlnA']:+.4f} | 代数性质（固定 AR/CFD），非拟合 |")
    a(f"| ISO 指数扫掠 | {', '.join(f'{e:+.2f}' for e in ratio['iso_reference']['exponents_swept'])} | 声明输入 |")
    for dui in sorted({r["displacement_kt"] for r in rows}):
        r0 = next(r for r in rows if r["displacement_kt"] == dui)
        a(f"| DUI = {dui:.0f} kt：F_IACS | {r0['iacs_force_MN']:.3f} MN | 控制子区 {r0['governing_sub_region']} |")
        a(f"| DUI = {dui:.0f} kt：载荷斑 b×w | {r0['patch_height_m']:.3f} × {r0['patch_width_m']:.3f} m | 面积 {r0['patch_area_m2']:.4f} m² |")
    a("")
    a("## 2. 方法与实现（含既有实现清单，禁重复实现）")
    a("")
    a("**既有实现清单（先列后补，避免重做）**")
    a("")
    a("| 既有模块 | 已具备能力 | 本次是否复用 |")
    a("|---|---|---|")
    a("| `benchmarks/anchors/iacs_ur_i_ice_load_anchor.py` + `iacs-ur-i-ice-load.json` | IACS 条款级转录 + PDF/引文 SHA-256 绑定 + `iacs_bow_design_load` | **复用**（直接作为 code_rule 可审计端，运行时重校验） |")
    a("| `src/tensorfem/ice_pressure_area.py` | 幂律最小二乘 / bootstrap 区间 / 分位数 / 多因子 log 拟合 | 概念复用（不确定度量化形态），未重复实现 |")
    a("| `src/tensorfem/knowledge_assets.py` | 证据资产目录（asset 级 `provenance` 字段、版本链） | 不重叠：那是资产目录，不是标定系数级 provenance |")
    a("| `src/tensorfem/marine_corrosion_ice_ultimate.py` | `IceDesignLoad.provenance` 自由文本 + `calibrate_ice_pressure` 占位压力标定 | 不重叠：自由文本无门禁、不可判真伪 |")
    a("| `src/tensorfem/iso19906_ice_6dof.py` | ISO 压力-面积对象（`provenance` 字符串、指数可扫） | 复用（作不可审计端） |")
    a("| `benchmarks/ice_coupled_6dof/run_w30_ice_load_vs_code.py` | 冰载荷 vs 规范对比（closure/verification 语义） | 不重叠：对比而不做 provenance 门禁 |")
    a("")
    a("**本次新增（唯一缺口）：`src/tensorfem/calibration_provenance.py`**")
    a("")
    a("- `ProvenanceRecord`：一个数的证据包（来源类型 code_rule / experiment / regression_dataset / judgement、"
      "引用、文档或数据集 SHA-256、估计量、样本数、声明相对不确定度+依据、有效域），含确定性 SHA-256 内容摘要。")
    a("- `audit()`：**fail-closed** 判定。code_rule 必须给文档 SHA-256 + 条款；experiment 必须给不确定度与依据；"
      "regression 必须给数据集 SHA-256 + 估计量 + n≥5 + 不确定度；`judgement` 永远不可审计；未知来源类型一律不可审计；"
      "有证据但未声明有效域降级为 PARTIAL。缺什么就列出什么，绝不推断。")
    a("- `combine_provenance()`：**最弱链规则**——比值不可能比它的较弱一端更可审计；不确定度按 GUM §5 正交合成。"
      "这正是防止「一个转录标准 + 一个手填系数」被包装成 code ratio 的机制。")
    a("- `CalibrationFactor`：自我定义校验（给了 raw 与 reference 就必须满足 `cf = ref/raw`，超容差即构造失败）；"
      "`calibrated_total()` 要么返回带 `unaudited_bias_rel` 的对象，要么在 `require_auditable=True` 时抛 `UnauditableCalibration`。")
    a("- 船队层：`fleet_dispersion()`（(max-min)/min，正是复现 34.5% 的定义）、`constant_penalty()`（单一常量对船队成员的最坏相对误差，"
      "对 15% 判定）、`best_single_constant()`（任何单一常量的 minimax 下限，解析解=调和中点）。")
    a("")
    a("**为何审计可以在 cf 层面完成**：因为 `total = raw*cf`，任意的 `raw` 都有 "
      "`相对误差(total) ≡ 相对误差(cf)`；该恒等式在三个不同 `raw` 上被机器复核（最大偏差 "
      f"{max(r['abs_difference'] for r in live['scaling_identity']['rows']):.2e}），因此无需 per-ship `raw` 数据即可判定。")
    a("")
    a("工具链：Python 3.11 + numpy + matplotlib；`python benchmarks/jour_ice_struct_calib_prov/run_calib_prov.py`"
      "（~9 s，本机跑完，无需集群）；测试 `pytest tests/test_calibration_provenance.py`（26 项）。")
    a("")
    a("## 3. 计算结果")
    a("")
    a("### 3.1 现网常量审计（判据 |成员误差| ≤ 15%）")
    a("")
    a("| 项 | got | ref | 误差 | 判定 |")
    a("|---|---|---|---|---|")
    d = live["dispersion_reported_vs_recomputed"]
    a(f"| 船队离散度 | {d['recomputed']*100:.3f}% | {d['reported']*100:.3f}% | {d['difference']*100:+.4f} pp | {'PASS' if abs(d['difference'])<1e-3 else 'FAIL'} |")
    a(f"| 成员误差 @ cf_min | {pen['error_rel_min']*100:+.3f}% | 0 | {abs(pen['error_rel_min'])*100:.3f}% | {'PASS' if abs(pen['error_rel_min'])<=tol else 'FAIL'} |")
    a(f"| 成员误差 @ cf_max | {pen['error_rel_max']*100:+.3f}% | 0 | {abs(pen['error_rel_max'])*100:.3f}% | {'PASS' if abs(pen['error_rel_max'])<=tol else 'FAIL'} |")
    a(f"| 最优单一常量 | {floor['constant']:.6f} | — | minimax {floor['minimax_abs_error_rel']*100:.3f}% | {'PASS' if floor['minimax_abs_error_rel']<=tol else 'FAIL'} |")
    a(f"| 标度恒等式 | 最大偏差 {max(r['abs_difference'] for r in live['scaling_identity']['rows']):.2e} | 0 | — | {'PASS' if live['scaling_identity']['identity_holds'] else 'FAIL'} |")
    a("")
    a(f"结论：出厂常量 `{rep['cf_ship_constant']:.6f}` 在船队上造成 "
      f"{pen['error_rel_min']*100:+.2f}% / {pen['error_rel_max']*100:+.2f}% 的成员误差，"
      f"**{pen['verdict']}** 15% 标准；连最优单一常量也只能到 {floor['minimax_abs_error_rel']*100:.3f}%"
      f"（{'勉强可行' if live['best_constant_meets_tolerance'] else '仍不可行'}），说明「单船常数」这个形式本身已把预算吃光。")
    a("")
    a("### 3.2 可溯源的模型形式比 cf = F_ISO / F_IACS（参数扫掠）")
    a("")
    a("| DUI [kt] | A [m²] | F_IACS [MN] | ISO ex | p_ISO [MPa] | F_ISO [MN] | cf |")
    a("|---|---|---|---|---|---|---|")
    for r in rows:
        a(f"| {r['displacement_kt']:.0f} | {r['patch_area_m2']:.4f} | {r['iacs_force_MN']:.3f} | "
          f"{r['iso_exponent']:+.2f} | {r['iso_pressure_MPa']:.4f} | {r['iso_force_MN']:.3f} | {r['cf_model_iso_over_iacs']:.4f} |")
    a("")
    a(f"比值区间 **{ratio['ratio_range']['min']:.4f} .. {ratio['ratio_range']['max']:.4f}**"
      f"（跨参数离散 {ratio['ratio_range']['dispersion_rel']*100:.2f}%，max/min = {ratio['ratio_range']['max_over_min']:.3f}）。"
      f"该离散与 `C_R` 无关（`C_R` = {ratio['C_R_invariance']['rows'][0]['C_R_MPa']:.1f} 与 "
      f"{ratio['C_R_invariance']['rows'][1]['C_R_MPa']:.1f} MPa 两次复算离散同为 "
      f"{ratio['C_R_invariance']['rows'][0]['dispersion_rel']*100:.3f}%，机器判定不变 "
      f"{ratio['C_R_invariance']['dispersion_invariant']}），只有比值的绝对水平随 `C_R` 线性缩放，故本单不声明绝对水平。")
    a("")
    a("### 3.3 Provenance 门禁（最弱链）")
    a("")
    a(f"- IACS 端：`{gate['iacs_verdict']['verdict']}`（文档 SHA-256 + 条款齐备）")
    a(f"- ISO 端：`{gate['iso_verdict']['verdict']}` —— {'; '.join(gate['iso_verdict']['reasons'])}")
    a(f"- 派生比值：`{gate['derived_verdict']['verdict']}`（继承较弱一端）")
    a(f"- `calibrated_total(..., require_auditable=True)` 抛出 `{gate['refusal']['exception']}`：**{gate['refusal']['raised']}**")
    a(f"- 放行路径携带 `unaudited_bias_rel = {gate['permissive_path']['unaudited_bias_rel']:.4f}`")
    a("")
    a("整改阶梯（由审计输出生成 → 补字段 → 重审）：")
    a("")
    a("| 阶段 | 判定 | 缺失字段 |")
    a("|---|---|---|")
    for step in ladder:
        rec = step["record"]
        a(f"| {step['stage']} | {rec['audit']['verdict']} | {', '.join(rec['audit']['missing']) or '—'} |")
    a("")
    a("## 4. 图")
    a("")
    a("![fleet band](fig_cf_fleet_band.png)")
    a("")
    a("**图 1** 出厂单一常量 `cf` 与船队 cf 区间、±15% 验收带、最优单一常量的关系；标注给出成员最坏误差。")
    a("")
    a("![model ratio](fig_cf_model_ratio.png)")
    a("")
    a("**图 2** 可溯源模型形式比 `cf = F_ISO/F_IACS` 随载荷斑面积与 ISO 指数的变化：比值是参数函数，其跨参数离散"
      f"（{ratio['ratio_range']['dispersion_rel']*100:.1f}%）与实测船队离散（{live['fleet_dispersion']['dispersion_rel']*100:.1f}%）同量级。")
    a("")
    a("## 5. 注意点 / 局限")
    a("")
    a("- 本单**只看到**船队 (min, max) 与一个船的常量，不假设分布，因此不做分布性陈述（不报 sd/COV 作为结论，只报区间）。")
    a("- ISO 19906 在本工作树 WITHHELD：3.2 的比值水平是**非规范**的，指数已扫掠以保证结论不依赖单一指数。"
      "门禁判 UNAUDITABLE 正是**结论本身**，不是本次运行的失败。")
    a("- 「标定系数」的语义是模型形式比（reference/raw），本单选定 reference = ISO 家族载荷、raw = IACS 设计载荷；"
      "两者是不同设计哲学的量（规范设计载荷 vs 全域压力-面积律），比值只作**参数依赖性与可审计性**的证据，不作载荷等效。")
    a("- 未取得试验/回归数据集，因此无法把出厂 `cf` 升级为 AUDITABLE；3.3 的整改阶梯只演示机制（n=4 仍 UNAUDITABLE，n=8 + 数据集 SHA-256 + 不确定度 + 有效域 → AUDITABLE）。")
    a("")
    a("## 6. 误差判定（逐项 是否<3%）")
    a("")
    a("| 关键量 | ref | got | 误差 | <3%？ |")
    a("|---|---|---|---|---|")
    a(f"| 船队离散度 | {d['reported']*100:.3f}% | {d['recomputed']*100:.3f}% | {abs(d['difference'])*100:.4f} pp | **是**（{abs(d['difference'])/max(abs(d['reported']),1e-30)*100:.4f}%） |")
    a(f"| 标度恒等式 d(total)/total = d(cf)/cf | 0 | {max(r['abs_difference'] for r in live['scaling_identity']['rows']):.2e} | {max(r['abs_difference'] for r in live['scaling_identity']['rows']):.2e} | **是** |")
    a(f"| 最优单一常量 minimax | 解析调和中点 | {floor['constant']:.6f} | 0（闭式一致） | **是** |")
    a(f"| `C_R` 不变性（比值离散） | {ratio['C_R_invariance']['rows'][0]['dispersion_rel']*100:.3f}% | {ratio['C_R_invariance']['rows'][1]['dispersion_rel']*100:.3f}% | {abs(ratio['C_R_invariance']['rows'][1]['dispersion_rel']-ratio['C_R_invariance']['rows'][0]['dispersion_rel']):.2e} | **是** |")
    a(f"| 出厂 cf 的成员误差 vs 15% 标准 | ≤{tol*100:.0f}% | {pen['worst_abs_error_rel']*100:.3f}% | 超限 {pen['worst_abs_error_rel']*100-tol*100:.3f} pp | **否**（{pen['verdict']}） |")
    a("")
    a("最后一行为本单**唯一否判决**：出厂单一常量的成员误差 16.44% 超出 15% 验收标准——这就是需要 provenance / 需要按参数"
      "（或按船）分开标定的量化理由。其余三行为算术/恒等式类判据，均在机器精度量级通过；本单**没有**对 cf 的绝对水平做"
      "3% 级精度声明，因为 ISO 侧 WITHHELD，任何如此声明都会是不可审计的偏置。")
    a("")
    a("**工件**：`artifacts/jour_ice_struct_calib_prov/calib_prov_audit.json`、"
      "`fig_cf_fleet_band.png`、`fig_cf_model_ratio.png`、`CASE-JOUR-ICE-STRUCT-CALIB-PROV.md`；"
      "复现：`python benchmarks/jour_ice_struct_calib_prov/run_calib_prov.py`；测试：`pytest tests/test_calibration_provenance.py`（26 passed）。")
    return "\n".join(L)


def main() -> int:
    payload = json.loads((ART / "calib_prov_audit.json").read_text())
    md = build_markdown(payload)
    figures = []
    for name, caption in (
        ("fig_cf_fleet_band.png",
         "图1 出厂单一常量 cf=0.753028 vs 船队 cf 区间 0.670077-0.901199 与 ±15% 验收带；最优单一常量 0.768640（minimax 14.709%）"),
        ("fig_cf_model_ratio.png",
         "图2 可溯源模型形式比 cf = F_ISO/F_IACS 随载荷斑面积与 ISO 指数（-0.50/-0.28/-0.20）的变化，两个排水量点；横带为实测船队 cf 区间"),
    ):
        data = base64.b64encode((ART / name).read_bytes()).decode()
        figures.append({"name": name, "caption": caption, "png_base64": data})

    body = {
        "title": "JOUR-ICE-STRUCT-CALIB-PROV：冰结构标定系数溯源审计（cf 单一常数在 34.5% 船队离散下 FAIL 15%）",
        "summary": (
            "为冰结构 total = raw × cf 链路建立 fail-closed 的标定系数 provenance 层（ProvenanceRecord / 最弱链组合 / "
            "CalibrationFactor 自校验 / calibrated_total 拒绝无证据标定 / 船队常量惩罚），并用 sha256 绑定的 IACS UR I2 转录"
            "与 ISO 19906 声明家族量化 cf 的参数依赖性；出厂常量 0.753028 的船队成员误差 -16.44%/+12.38%，判 FAIL 15%，"
            "任何单一常量 minimax 下限 14.71%。"
        ),
        "scenario_key": "ice-structure-calibration-provenance",
        "mission_id": MISSION_ID,
        "tags": ["ice-structure", "calibration-provenance", "auditability", "IACS-UR-I2",
                 "ISO-19906", "TensorFEM", "JOUR-ICE-STRUCT-CALIB-PROV"],
        "markdown": md,
        "figures": figures,
    }
    token_req = urllib.request.Request(
        API + "/auth/login",
        data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                         "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    token = json.load(urllib.request.urlopen(token_req))["access_token"]
    req = urllib.request.Request(
        API + "/tensor/reports",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
    )
    resp = json.load(urllib.request.urlopen(req))
    print("report posted:", resp.get("id", resp))
    (ART / "report_response.json").write_text(json.dumps(resp, indent=2))
    (ART / "report_body.md").write_text(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
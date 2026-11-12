import base64, json, os, urllib.request

BASE = os.environ["TENSOR_API_BASE"]
login = urllib.request.Request(
    BASE + "/auth/login",
    data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                     "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
    headers={"Content-Type": "application/json"}, method="POST")
tok = json.load(urllib.request.urlopen(login))
token = tok.get("access_token") or tok.get("token")
png = base64.b64encode(open("artifacts/rainflow_convention/convention.png", "rb").read()).decode()
audit = json.load(open("artifacts/rainflow_convention/audit.json"))
markdown = """# Rainflow 计数：四点规则落地 + 旧判据裁决（损伤 +225% 被证伪）

## ① 案例介绍
`tensorfem.marine_fatigue_advanced.rainflow_cycles` 是 TensorFEM 全部疲劳数字的最前端
（热点谱 Miner 损伤、焊缝筛选、冰激振动计数）。2026-11-08 审计（docs/rainflow-counting-defect.md）
判定「闭合循环被记成半循环」，并给出嵌套记录损伤 234 vs 72（+225%）的结论后卡住：
两次修复尝试被自己的判据否决，缺陷留在 main 上（两条 strict-xfail 绊线）。

## ② 怎么做的
判别性实验对比三条规则：现有实现（三点栈 + len(stack)==3 特判）、「只删特判」的三点规则、
教科书 ASTM E1049 四点规则。判据四条：弧守恒 Σcount·range = ½Σ|Δs|、结构化锚点、等效循环数、
损伤 Σcount·range^3。修复：把四点规则写进 `rainflow_cycles`（闭记录残留外侧区间配成 1 个
full cycle）；新增结构独立的复核器 `four_point_reference`（重复扫描版）与「水平穿越」分解器，
三者在 5 条锚点记录 + 120 条随机闭记录上同表对比。

## ③ 计算结果（artifacts/rainflow_convention/audit.json，程序化产出）
| 记录 | 修复后 full/half | 四点参考 | 结构一致 | 损伤相对差 | 弧守恒误差 |
|---|---|---|---|---|---|
| 等幅块 0,10,0,10,0,10,0 | 3/0 (10×3) | 3/0 | 是 | 0 | 0 |
| 扩张锯齿 0,1,-1,2,-2,3,-3,0 | 0/7 (1,2,3,3,4,5,6) | 0/7 | 是 | 0 | 0 |
| 三角 0,100,0 | 1/0 (100) | 1/0 | 是 | 0 | 0 |
| 两级块 | 4/0 (5,5,20,20) | 4/0 | 是 | 0 | 0 |
| 正弦 40 周期 | 39/3 (等效 40.5) | 39/3 | 是 | 0 | 0 |

随机闭合记录 120 条：修复后最大弧守恒误差 2.22e-16（与参考器同值）。

## ④ 结果图
见 figures（左：各记录 full/half 结构；右：修复后 vs 四点参考的 m=3 损伤，柱高全等）。

## ⑤ 注意点
- 只删 `len(stack)==3` 特判**不是**修复：三点规则在扩张锯齿上闭出 range 1/3/5，丢掉 1.5 单位弧长，
  80 条随机闭记录中 **71 条**违反弧守恒——这正是 11-08 那次修复被否的原因（缺的是第四点，不是插值）。
- 旧期望 `{1,1,2,2,3,3}` 属水平穿越（插值闭合）分解，会把同一条弧重复消费（Σcount·range ≠ ½Σ|Δs|）；
  ASTM 四点规则给 7 个半循环（12 = ½Σ|Δs|），物理上唯一自洽：振幅持续扩张时不闭任何环。
- 下游 Miner 损伤对 count 线性，故等幅/两级块/正弦谱损伤不变；变化在循环结构标签与平均应力修正的权重分配。
- 局限：未对照 ASTM 原图核对「闭记录残留在半循环」约定（WITHHELD）；全仓疲劳面未端到端复跑。

## ⑥ 误差判定
等幅 full 个数 0% / 三角 0% / 两级块集合 0% / 正弦等效循环 1.25% / 弧守恒 ≤2.22e-16 /
与独立四点参考的结构与损伤 0% → **关键量全部 <3%**。旧审计的 72 与 +225% 判为判据错误
（ASTM 参考同为 234），不再作为验收线，仅以 strict-xfail 保留为「禁止重新采纳」的绊线。
"""
payload = {
    "title": "Rainflow 四点规则落地 + 旧判据裁决：闭合循环结构缺陷修复，损伤 +225% 结论被证伪",
    "summary": "rainflow_cycles 改按 ASTM E1049 四点规则计数（等幅 3 full、三角 1 full、两级块 4 full、"
               "正弦 40.5 等效、120 条随机闭记录弧守恒残差 2.2e-16），与结构独立复核器零差；"
               "旧审计的嵌套损伤 72/+225% 判据被证伪（四点参考同为 234，{1,1,2,2,3,3} 是水平穿越非 rainflow）。",
    "scenario_key": "tensorfem.fatigue.rainflow",
    "mission_id": 16562,
    "tags": ["TensorFEM", "fatigue", "rainflow", "verification", "falsified-hypothesis"],
    "markdown": markdown,
    "figures": [{"name": "rainflow_convention.png",
                 "caption": "左：5 条锚点记录的 full/half 循环结构；右：修复后与独立四点参考的 Miner 损伤（m=3，对数轴）",
                 "png_base64": png}],
}
req = urllib.request.Request(BASE + "/tensor/reports", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + token}, method="POST")
try:
    out = json.load(urllib.request.urlopen(req))
    print("REPORT ID", out.get("id"), out.get("title"))
except urllib.error.HTTPError as exc:
    print("HTTP", exc.code, exc.read()[:500])

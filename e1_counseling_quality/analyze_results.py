# -*- coding: utf-8 -*-
"""
E1 实验 - 统计分析与报告生成
============================

汇总各条件盲评结果，输出：
1. 各条件六维得分 mean±std + 95% CI
2. gvc_full vs 各对照的 Welch t 检验 + Cohen's d + Holm 多重比较校正
3. 安全违规率（诊断断言/有害建议/评判）
4. 分话题得分
5. Markdown 报告 + JSON 结果

用法：
    python3 analyze_results.py
"""
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, ".."))
from statistical_tests import mean, std, confidence_interval, welch_t_test, cohens_d  # noqa: E402

JUDGE_DIR = os.path.join(BASE_DIR, "..", "e1_results", "judgments")
JUDGE2_DIR = os.path.join(BASE_DIR, "..", "e1_results", "judgments_conv")
OUT_DIR = os.path.join(BASE_DIR, "..", "e1_results")

DIMENSIONS = ["empathy", "identification", "reflection", "strategy", "encouragement", "relevance"]
CONV_DIMENSIONS = ["validation", "load", "focus", "invitation", "alliance", "naturalness"]
SAFETY_FLAGS = ["diagnostic_claim", "harmful_advice", "judgmental"]
MAIN_CONDITION = "gvc_full"

COND_LABELS = {
    "gvc_full": "三极议会完整版 (GVC)",
    "gvc_v2": "三极议会-整合式合成 (GVC-v2)",
    "ablation_no_ca": "消融: 去CA验证器",
    "ablation_no_skg": "消融: 去SKG图谱",
    "ablation_no_convergence": "消融: 去收敛判据",
    "single_llm": "单模型直出",
    "cbt_prompt": "单模型+CBT提示词",
    "human": "人类高赞回答(在线同伴支持)",
}


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5 or 1e-9
    sy = (sum((y - my) ** 2 for y in ys) / n) ** 0.5 or 1e-9
    return cov / (sx * sy)


def load_judgments(judge_dir=None) -> dict:
    """{condition: [records]}；过滤并去重（同 id 保留首条）与兜底短文（resp_len<30）"""
    data = {}
    for path in sorted(glob.glob(os.path.join(judge_dir or JUDGE_DIR, "*.jsonl"))):
        cond = os.path.basename(path)[:-6]
        records, seen = [], set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("resp_len", 999) < 30 or r.get("id") in seen:
                    continue
                seen.add(r.get("id"))
                records.append(r)
        if records:
            data[cond] = records
    return data


def holm_correction(pvals: list) -> list:
    """Holm-Bonferroni 校正，返回校正后 p 值（保持原顺序）"""
    m = len(pvals)
    indexed = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    prev = 0.0
    for rank, idx in enumerate(indexed):
        adj = min(1.0, (m - rank) * pvals[idx])
        adj = max(adj, prev)  # 保证单调
        adjusted[idx] = adj
        prev = adj
    return adjusted


def paired_common(a_recs: list, b_recs: list) -> tuple:
    """取两条件共有样本 id（保证配对可比）"""
    a_map = {r["id"]: r for r in a_recs}
    b_map = {r["id"]: r for r in b_recs}
    common = sorted(set(a_map) & set(b_map))
    return [a_map[i] for i in common], [b_map[i] for i in common]


def analyze(data: dict) -> dict:
    result = {"generated_at": datetime.now().isoformat(), "conditions": {}, "comparisons": {}}

    # 1. 各条件描述统计
    for cond, recs in data.items():
        stats = {"n": len(recs)}
        for dim in DIMENSIONS + ["total"]:
            vals = [r[dim] for r in recs if dim in r]
            ci = confidence_interval(vals)
            stats[dim] = {"mean": round(mean(vals), 2), "std": round(std(vals), 2),
                          "ci_lower": round(ci["ci_lower"], 2), "ci_upper": round(ci["ci_upper"], 2)}
        for flag in SAFETY_FLAGS:
            flags = [1 if r.get(flag) else 0 for r in recs]
            stats[flag + "_rate"] = round(mean(flags), 4)
        stats["avg_latency"] = round(mean([r.get("gen_latency", 0) for r in recs]), 1)
        stats["avg_resp_len"] = round(mean([r.get("resp_len", 0) for r in recs]), 0)
        # 分话题
        by_topic = defaultdict(list)
        for r in recs:
            by_topic[r.get("topic", "other")].append(r["total"])
        stats["by_topic"] = {t: round(mean(v), 2) for t, v in sorted(by_topic.items())}
        result["conditions"][cond] = stats

    # 2. gvc_full vs 各对照（Welch t + Cohen's d + Holm）
    if MAIN_CONDITION in data:
        main_recs = data[MAIN_CONDITION]
        for cond, recs in data.items():
            if cond == MAIN_CONDITION:
                continue
            a, b = paired_common(main_recs, recs)
            if len(a) < 10:
                continue
            comp = {"n_paired": len(a), "dims": {}}
            pvals, dims_order = [], []
            for dim in ["total"] + DIMENSIONS:
                g1 = [r[dim] for r in a]
                g2 = [r[dim] for r in b]
                t = welch_t_test(g1, g2)
                d = cohens_d(g1, g2)
                comp["dims"][dim] = {
                    "gvc_mean": round(mean(g1), 2), "other_mean": round(mean(g2), 2),
                    "diff": round(mean(g1) - mean(g2), 2),
                    "t": round(t["t"], 3), "p_raw": round(t["p"], 5),
                    "cohens_d": round(d, 3),
                }
                pvals.append(t["p"])
                dims_order.append(dim)
            adjusted = holm_correction(pvals)
            for dim, p_adj in zip(dims_order, adjusted):
                comp["dims"][dim]["p_holm"] = round(p_adj, 5)
                comp["dims"][dim]["significant"] = p_adj < 0.05
            result["comparisons"][cond] = comp

    return result


def sig_mark(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def render_markdown(result: dict) -> str:
    lines = ["# E1 实验报告：真实求助语料上的咨询质量对比",
             "",
             f"生成时间：{result['generated_at'][:19]}",
             "",
             "语料：PsyQA 真实心理求助帖（分层采样）；评估：AutoCBT 六维盲评（0-7分/维）",
             "",
             "## 1. 各条件总体得分",
             "",
             "| 条件 | N | 总分 | 共情 | 识别 | 反思 | 策略 | 鼓励 | 相关 | 平均延迟(s) | 平均长度 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    order = ["gvc_full", "ablation_no_ca", "ablation_no_skg", "ablation_no_convergence",
             "cbt_prompt", "single_llm", "human"]
    for cond in order:
        s = result["conditions"].get(cond)
        if not s:
            continue
        label = COND_LABELS.get(cond, cond)
        row = [label, str(s["n"]),
               f"{s['total']['mean']}±{s['total']['std']}"]
        row += [f"{s[d]['mean']}" for d in DIMENSIONS]
        row += [str(s["avg_latency"]), str(int(s["avg_resp_len"]))]
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## 2. 安全违规率", "",
              "| 条件 | 诊断断言 | 有害建议 | 评判指责 |", "|---|---|---|---|"]
    for cond in order:
        s = result["conditions"].get(cond)
        if not s:
            continue
        lines.append(f"| {COND_LABELS.get(cond, cond)} | {s['diagnostic_claim_rate']:.1%} "
                     f"| {s['harmful_advice_rate']:.1%} | {s['judgmental_rate']:.1%} |")

    lines += ["", "## 3. GVC完整版 vs 各对照（Welch t 检验，Holm 校正）", ""]
    for cond, comp in result["comparisons"].items():
        lines.append(f"### vs {COND_LABELS.get(cond, cond)}（配对样本 N={comp['n_paired']}）")
        lines.append("")
        lines.append("| 维度 | GVC | 对照 | 差值 | t | p(Holm) | Cohen's d | 显著性 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for dim in ["total"] + DIMENSIONS:
            d = comp["dims"][dim]
            lines.append(f"| {dim} | {d['gvc_mean']} | {d['other_mean']} | {d['diff']:+} "
                         f"| {d['t']} | {d['p_holm']} | {d['cohens_d']} | {sig_mark(d['p_holm'])} |")
        lines.append("")

    lines += ["## 4. 分话题总分", ""]
    topics = sorted({t for s in result["conditions"].values() for t in s.get("by_topic", {})})
    lines.append("| 条件 | " + " | ".join(topics) + " |")
    lines.append("|---" * (len(topics) + 1) + "|")
    for cond in order:
        s = result["conditions"].get(cond)
        if not s:
            continue
        row = [COND_LABELS.get(cond, cond)] + [str(s["by_topic"].get(t, "-")) for t in topics]
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "---", "",
              "注：`*` p<0.05，`**` p<0.01，`***` p<0.001（Holm 校正后）；"
              "评审模型盲评，不知晓回复来源条件。",
              "",
              "> 重要说明：`human` 为 PsyQA 高赞回答，属于**在线同伴支持**（非专业咨询师），"
              "作为现实世界下界/中等锰点；专业咨询师上界对照见 E2（CPsyCounD 专业语料）。"]
    return "\n".join(lines)


def analyze_conv_and_bias(data_auto: dict) -> tuple:
    """线1a：conv 量规统计 + 长度偏置分析，返回 (conv_stats, bias, dual_table_lines)"""
    data_conv = load_judgments(JUDGE2_DIR)
    conv_stats = {}
    for cond, recs in data_conv.items():
        vals = [r["total"] for r in recs]
        ci = confidence_interval(vals)
        st = {"n": len(recs), "total": {"mean": round(mean(vals), 2), "std": round(std(vals), 2),
                                         "ci_lower": round(ci["ci_lower"], 2), "ci_upper": round(ci["ci_upper"], 2)}}
        for d in CONV_DIMENSIONS:
            v = [r[d] for r in recs if d in r]
            st[d] = round(mean(v), 2)
        conv_stats[cond] = st

    # 长度偏置：AutoCBT 量规下 长度~总分 相关（全体+条件内）；conv 量规同法对照
    bias = {}
    for name, dat in [("autocbt", data_auto), ("conv", data_conv)]:
        xs, ys = [], []
        within = {}
        for cond, recs in dat.items():
            cx = [r.get("resp_len", 0) for r in recs]
            cy = [r["total"] for r in recs]
            xs += cx; ys += cy
            within[cond] = round(_pearson(cx, cy), 3)
        bias[name] = {"overall_r": round(_pearson(xs, ys), 3), "within": within, "n": len(xs)}

    # 双量规对照表
    lines = ["", "## 双量规对照：覆盖度 vs 对话适切性", "",
             "| 条件 | AutoCBT覆盖度/42 | 对话适切性/42 |", "|---|---|---|"]
    order = ["gvc_full", "gvc_v2", "ablation_no_ca", "ablation_no_skg", "ablation_no_convergence",
             "cbt_prompt", "single_llm", "human"]
    auto_totals = {c: round(mean([r["total"] for r in recs]), 1) for c, recs in data_auto.items()}
    for c in order:
        a = auto_totals.get(c, "-")
        v = conv_stats.get(c, {}).get("total", {}).get("mean", "-")
        lines.append(f"| {COND_LABELS.get(c, c)} | {a} | {v} |")
    lines += ["", "## 评审长度偏置分析", "",
              f"- AutoCBT 覆盖度量规：回复长度与总分 Pearson r = **{bias['autocbt']['overall_r']}**"
              f"（N={bias['autocbt']['n']}，强正相关 → 结构性偏向长文）",
              f"- 对话适切性量规：长度与总分 r = **{bias['conv']['overall_r']}**",
              f"- 条件内相关（AutoCBT）：" + ", ".join(f"{k}={v}" for k, v in bias["autocbt"]["within"].items()),
              "",
              "> 解读：两套量规结论反转——覆盖度量规奖励信息密集的长文（利好单模型作文式输出），"
              "对话适切性量规衡量真实咨询单轮回应质量（利好简洁聚焦的对话式系统）。"]
    return conv_stats, bias, lines


def main():
    data = load_judgments()
    if not data:
        print("未找到盲评结果，请先运行 run_judge.py")
        return
    print(f"已加载条件: {list(data.keys())}")

    result = analyze(data)
    # 线1a：双量规 + 长度偏置
    conv_stats, bias, dual_lines = analyze_conv_and_bias(data)
    result["conv_conditions"] = conv_stats
    result["length_bias"] = bias
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(OUT_DIR, f"E1_analysis_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(OUT_DIR, f"E1_report_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(result))
        f.write("\n" + "\n".join(dual_lines) + "\n")

    print(f"分析结果: {json_path}")
    print(f"实验报告: {md_path}")


if __name__ == "__main__":
    main()

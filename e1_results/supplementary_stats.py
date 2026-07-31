#!/usr/bin/env python3
"""E1 补充统计（审稿修订）：配对 t 检验 + 95%CI + 双量规全比较 + 条件内长度相关。
纯标准库实现，读取 judgments/ 与 judgments_conv/，输出 supplementary_stats.md。"""
import json
import math
import os

BASE = os.path.dirname(os.path.abspath(__file__))
CONDS = ["gvc_full", "gvc_v2", "ablation_no_ca", "ablation_no_skg",
         "ablation_no_convergence", "single_llm", "cbt_prompt", "human"]
CN = {"gvc_full": "GVC", "gvc_v2": "GVC-v2", "ablation_no_ca": "no-CA",
      "ablation_no_skg": "no-SKG", "ablation_no_convergence": "no-conv",
      "single_llm": "single", "cbt_prompt": "single+CBT", "human": "human peer"}


def load(rubric_dir, cond):
    recs = {}
    p = os.path.join(BASE, rubric_dir, cond + ".jsonl")
    if not os.path.exists(p):
        return recs
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("resp_len", 0) >= 30:  # 与主分析一致的兜底过滤
            recs[r["id"]] = r
    return recs


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def t_cdf(t, df):
    """双侧 p 值用：t 分布 CDF（数值积分足够精度）。"""
    def integrand(x):
        return (1 + x * x / df) ** (-(df + 1) / 2)
    # 简单 Simpson 积分 [0, |t|]
    n, a, b = 2000, 0.0, abs(t)
    if b == 0:
        return 0.5
    h = (b - a) / n
    s = integrand(a) + integrand(b)
    for i in range(1, n):
        s += integrand(a + i * h) * (4 if i % 2 else 2)
    integral = s * h / 3
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    return 0.5 + c * integral


def paired_t(a, b):
    """配对 t：返回 t, df, p(双侧), dz, 差值均值, 差值95%CI"""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    md, sdd = mean(d), sd(d)
    se = sdd / math.sqrt(n)
    t = md / se if se > 0 else 0.0
    df = n - 1
    p = 2 * (1 - t_cdf(t, df))
    dz = md / sdd if sdd > 0 else 0.0
    tcrit = 1.96 + 2.0 / df  # 近似 t 临界值（n>=10 时误差<2%）
    ci = (md - tcrit * se, md + tcrit * se)
    return t, df, min(max(p, 0.0), 1.0), dz, md, ci


def ci95(xs):
    m, s = mean(xs), sd(xs)
    se = s / math.sqrt(len(xs))
    tcrit = 1.96 + 2.0 / (len(xs) - 1)
    return m, m - tcrit * se, m + tcrit * se


def pearson(xs, ys):
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else 0.0


def holm(pairs):
    """pairs: list of (name, p) → dict name->p_holm"""
    m = len(pairs)
    s = sorted(pairs, key=lambda x: x[1])
    out, prev = {}, 0.0
    for i, (name, p) in enumerate(s):
        adj = min(1.0, max(prev, (m - i) * p))
        out[name] = adj
        prev = adj
    return out


def analyze(rubric_dir, label, lines):
    data = {c: load(rubric_dir, c) for c in CONDS}
    lines.append(f"\n## {label}\n")
    lines.append("### 各条件总分均值与 95%CI\n")
    lines.append("| 条件 | N | 均值 | 95%CI |")
    lines.append("|---|---|---|---|")
    for c in CONDS:
        if not data[c]:
            continue
        tot = [r["total"] for r in data[c].values()]
        m, lo, hi = ci95(tot)
        lines.append(f"| {CN[c]} | {len(tot)} | {m:.2f} | [{lo:.2f}, {hi:.2f}] |")

    lines.append("\n### GVC vs 各对照（配对 t 检验，按 id 配对；Holm 校正族=7 对比较）\n")
    lines.append("| 对照 | n配对 | Δ(GVC−对照) | 95%CI | t | df | p(Holm) | d_z |")
    lines.append("|---|---|---|---|---|---|---|---|")
    gvc = data["gvc_full"]
    raw = []
    rows = {}
    for c in CONDS:
        if c == "gvc_full" or not data[c]:
            continue
        ids = sorted(set(gvc) & set(data[c]))
        if len(ids) < 5:
            continue
        a = [gvc[i]["total"] for i in ids]
        b = [data[c][i]["total"] for i in ids]
        t, df, p, dz, md, ci = paired_t(a, b)
        raw.append((c, p))
        rows[c] = (len(ids), md, ci, t, df, dz)
    ph = holm(raw)
    for c in CONDS:
        if c not in rows:
            continue
        n, md, ci, t, df, dz = rows[c]
        p = ph[c]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        lines.append(f"| {CN[c]} | {n} | {md:+.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] | "
                     f"{t:.2f} | {df} | {p:.4g} {sig} | {dz:+.2f} |")

    lines.append("\n### 条件内 长度–总分 Pearson 相关\n")
    lines.append("| 条件 | r(条件内) |")
    lines.append("|---|---|")
    for c in CONDS:
        if not data[c]:
            continue
        xs = [r["resp_len"] for r in data[c].values()]
        ys = [r["total"] for r in data[c].values()]
        if len(xs) >= 10 and sd(xs) > 0:
            lines.append(f"| {CN[c]} | {pearson(xs, ys):.3f} |")
    # pooled
    xs, ys = [], []
    for c in CONDS:
        for r in data[c].values():
            xs.append(r["resp_len"]); ys.append(r["total"])
    lines.append(f"| **pooled(跨条件, 含混淆)** | {pearson(xs, ys):.3f} |")


def main():
    lines = ["# E1 补充统计（修订版）",
             "",
             "响应审稿意见：配对 t 检验（同一 50 条求助帖，按 id 配对）替代独立样本检验；",
             "补 95%CI；补对话适切性量规全套显著性；补条件内长度相关（区分 pooled 混淆）。",
             "过滤规则与主分析一致（resp_len>=30 兜底过滤）。"]
    analyze("judgments", "AutoCBT 覆盖度量规", lines)
    analyze("judgments_conv", "对话适切性量规", lines)
    out = os.path.join(BASE, "supplementary_stats.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("written:", out)
    print("\n".join(lines[:120]))


if __name__ == "__main__":
    main()

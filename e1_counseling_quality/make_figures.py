# -*- coding: utf-8 -*-
"""
E1 出图脚本（纯 Python 生成 SVG，零第三方依赖）
==============================================

读取最新 E1_analysis_*.json，生成可直接放入论文的矢量图（SVG，可用浏览器/工具转 PDF/PNG）：
- E1_fig_total.svg   : 各条件咨询质量总分（横向柱状 + 95% CI 误差线）
- E1_fig_dims.svg    : gvc_full vs 关键对照 的六维得分对比（分组柱状）
- E1_fig_safety.svg  : 各条件安全违规率

设计取向：为什么用 SVG 而非 matplotlib —— 本机/容器均无 matplotlib 且沙盒无法安装；
SVG 为矢量、无依赖、可复现，项目 docs/diagrams 亦采用 SVG。

用法：
    python make_figures.py            # 用最新 E1_analysis_*.json
    python make_figures.py <path.json>
"""
import glob
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "..", "e1_results")

DIMS = ["empathy", "identification", "reflection", "strategy", "encouragement", "relevance"]
DIM_CN = {"empathy": "共情", "identification": "识别", "reflection": "反思",
          "strategy": "策略", "encouragement": "鼓励", "relevance": "相关"}
COND_CN = {"gvc_full": "三极议会", "gvc_v2": "议会-整合式", "ablation_no_ca": "去CA", "ablation_no_skg": "去SKG",
           "ablation_no_convergence": "去收敛", "single_llm": "单模型", "cbt_prompt": "CBT提示词",
           "human": "在线同伴"}
ORDER = ["gvc_full", "gvc_v2", "ablation_no_ca", "ablation_no_skg", "ablation_no_convergence",
         "cbt_prompt", "single_llm", "human"]
PALETTE = {"gvc_full": "#2563eb", "gvc_v2": "#1d4ed8", "ablation_no_ca": "#60a5fa", "ablation_no_skg": "#93c5fd",
           "ablation_no_convergence": "#bfdbfe", "cbt_prompt": "#f59e0b",
           "single_llm": "#ef4444", "human": "#10b981"}


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _svg_header(w, h, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="PingFang SC,Microsoft YaHei,sans-serif">',
            f'<rect width="{w}" height="{h}" fill="white"/>',
            f'<text x="{w/2}" y="26" font-size="16" font-weight="bold" text-anchor="middle">{_esc(title)}</text>']


def load_latest(path=None):
    if path is None:
        cands = sorted(glob.glob(os.path.join(RESULT_DIR, "E1_analysis_*.json")))
        if not cands:
            raise SystemExit("未找到 E1_analysis_*.json，请先运行 analyze_results.py")
        path = cands[-1]
    with open(path, encoding="utf-8") as f:
        return json.load(f), path


def fig_total(result):
    """各条件总分横向柱 + 95% CI"""
    conds = [c for c in ORDER if c in result["conditions"]]
    W, H = 720, 60 + 46 * len(conds) + 40
    left, right = 130, W - 40
    maxv = 42.0
    def x(v): return left + (right - left) * v / maxv
    s = _svg_header(W, H, "E1 各条件咨询质量总分（AutoCBT 六维合计, 0–42, 95% CI）")
    # 轴刻度
    for tick in range(0, 43, 7):
        gx = x(tick)
        s.append(f'<line x1="{gx:.1f}" y1="46" x2="{gx:.1f}" y2="{H-40}" stroke="#eee"/>')
        s.append(f'<text x="{gx:.1f}" y="{H-24}" font-size="10" fill="#888" text-anchor="middle">{tick}</text>')
    y = 60
    for c in conds:
        st = result["conditions"][c]
        t = st["total"]
        bx, bw = x(0), x(t["mean"]) - x(0)
        s.append(f'<text x="{left-8}" y="{y+18}" font-size="12" text-anchor="end">{_esc(COND_CN.get(c,c))}</text>')
        s.append(f'<rect x="{bx:.1f}" y="{y}" width="{max(0,bw):.1f}" height="26" fill="{PALETTE.get(c,"#888")}" rx="3"/>')
        # CI 误差线
        lo, hi = x(t.get("ci_lower", t["mean"])), x(t.get("ci_upper", t["mean"]))
        cy = y + 13
        s.append(f'<line x1="{lo:.1f}" y1="{cy}" x2="{hi:.1f}" y2="{cy}" stroke="#333" stroke-width="1.5"/>')
        s.append(f'<line x1="{lo:.1f}" y1="{cy-4}" x2="{lo:.1f}" y2="{cy+4}" stroke="#333"/>')
        s.append(f'<line x1="{hi:.1f}" y1="{cy-4}" x2="{hi:.1f}" y2="{cy+4}" stroke="#333"/>')
        s.append(f'<text x="{x(t["mean"])+6:.1f}" y="{y+18}" font-size="11" fill="#111">{t["mean"]}</text>')
        y += 46
    s.append("</svg>")
    return "\n".join(s)


def fig_dims(result):
    """六维分组柱状：选 gvc_full/cbt_prompt/single_llm/human"""
    sel = [c for c in ["gvc_full", "cbt_prompt", "single_llm", "human"] if c in result["conditions"]]
    W, H = 760, 340
    left, bottom, top = 50, H - 70, 50
    plotw = W - left - 20
    group_w = plotw / len(DIMS)
    bar_w = group_w / (len(sel) + 1)
    def y(v): return bottom - (bottom - top) * v / 7.0
    s = _svg_header(W, H, "E1 六维咨询质量对比（0–7）")
    for tick in range(0, 8):
        gy = y(tick)
        s.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{W-20}" y2="{gy:.1f}" stroke="#eee"/>')
        s.append(f'<text x="{left-6}" y="{gy+3:.1f}" font-size="10" fill="#888" text-anchor="end">{tick}</text>')
    for di, dim in enumerate(DIMS):
        gx0 = left + di * group_w
        s.append(f'<text x="{gx0+group_w/2:.1f}" y="{bottom+16}" font-size="11" text-anchor="middle">{_esc(DIM_CN[dim])}</text>')
        for si, c in enumerate(sel):
            v = result["conditions"][c].get(dim, {}).get("mean", 0)
            bx = gx0 + (si + 0.5) * bar_w
            by = y(v)
            s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w*0.9:.1f}" height="{bottom-by:.1f}" fill="{PALETTE.get(c,"#888")}"/>')
    # 图例
    lx = left
    for c in sel:
        s.append(f'<rect x="{lx}" y="{H-30}" width="12" height="12" fill="{PALETTE.get(c,"#888")}"/>')
        s.append(f'<text x="{lx+16}" y="{H-20}" font-size="11">{_esc(COND_CN.get(c,c))}</text>')
        lx += 110
    s.append("</svg>")
    return "\n".join(s)


def fig_safety(result):
    """安全违规率（诊断断言/有害建议/评判）"""
    conds = [c for c in ORDER if c in result["conditions"]]
    flags = [("diagnostic_claim_rate", "诊断断言"), ("harmful_advice_rate", "有害建议"), ("judgmental_rate", "评判指责")]
    W, H = 720, 60 + 40 * len(conds) + 40
    left, right = 130, W - 60
    maxv = max(0.05, max(result["conditions"][c].get(f, 0) for c in conds for f, _ in flags))
    def x(v): return left + (right - left) * v / maxv
    colors = ["#ef4444", "#b91c1c", "#f59e0b"]
    s = _svg_header(W, H, "E1 各条件安全违规率（越低越好）")
    y = 56
    for c in conds:
        st = result["conditions"][c]
        s.append(f'<text x="{left-8}" y="{y+16}" font-size="12" text-anchor="end">{_esc(COND_CN.get(c,c))}</text>')
        yy = y
        for (f, _), col in zip(flags, colors):
            v = st.get(f, 0)
            s.append(f'<rect x="{x(0):.1f}" y="{yy}" width="{max(0,x(v)-x(0)):.1f}" height="9" fill="{col}"/>')
            s.append(f'<text x="{x(v)+4:.1f}" y="{yy+8}" font-size="8" fill="#333">{v:.0%}</text>')
            yy += 11
        y += 40
    lx = left
    for (f, name), col in zip(flags, colors):
        s.append(f'<rect x="{lx}" y="{H-28}" width="10" height="10" fill="{col}"/>')
        s.append(f'<text x="{lx+14}" y="{H-19}" font-size="10">{_esc(name)}</text>')
        lx += 140
    s.append("</svg>")
    return "\n".join(s)


def fig_dual(result):
    """双量规对照（覆盖度 vs 对话适切性）分组横向柱——论文主图，体现结论反转"""
    conds = [c for c in ORDER if c in result.get("conditions", {})]
    conv = result.get("conv_conditions", {})
    W, H = 760, 70 + 52 * len(conds) + 30
    left, right = 150, W - 50
    maxv = 42.0
    def x(v): return left + (right - left) * v / maxv
    s = _svg_header(W, H, "E1 双量规对照：覆盖度 vs 对话适切性（0–42）")
    for tick in range(0, 43, 7):
        gx = x(tick)
        s.append(f'<line x1="{gx:.1f}" y1="46" x2="{gx:.1f}" y2="{H-30}" stroke="#eee"/>')
        s.append(f'<text x="{gx:.1f}" y="{H-14}" font-size="10" fill="#888" text-anchor="middle">{tick}</text>')
    y = 60
    for c in conds:
        cov = result["conditions"][c].get("total", {}).get("mean", 0)
        apt = conv.get(c, {}).get("total", {}).get("mean", 0)
        s.append(f'<text x="{left-8}" y="{y+16}" font-size="11" text-anchor="end">{_esc(COND_CN.get(c,c))}</text>')
        # 覆盖度（灰色）
        s.append(f'<rect x="{x(0):.1f}" y="{y}" width="{max(0,x(cov)-x(0)):.1f}" height="16" fill="#94a3b8"/>')
        s.append(f'<text x="{x(cov)+4:.1f}" y="{y+13}" font-size="10" fill="#475569">{cov}</text>')
        # 对话适切性（蓝，突出）
        s.append(f'<rect x="{x(0):.1f}" y="{y+18}" width="{max(0,x(apt)-x(0)):.1f}" height="16" fill="{PALETTE.get(c,"#2563eb")}"/>')
        s.append(f'<text x="{x(apt)+4:.1f}" y="{y+31}" font-size="10" fill="#1e3a8a">{apt}</text>')
        y += 52
    # 图例
    s.append(f'<rect x="{left}" y="{H-26}" width="12" height="12" fill="#94a3b8"/><text x="{left+16}" y="{H-16}" font-size="10">AutoCBT覆盖度</text>')
    s.append(f'<rect x="{left+140}" y="{H-26}" width="12" height="12" fill="#2563eb"/><text x="{left+156}" y="{H-16}" font-size="10">对话适切性</text>')
    s.append("</svg>")
    return "\n".join(s)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    result, used = load_latest(path)
    print("读取:", used)
    outdir = os.path.join(RESULT_DIR, "figures")
    os.makedirs(outdir, exist_ok=True)
    figs = {"E1_fig_total.svg": fig_total, "E1_fig_dims.svg": fig_dims,
            "E1_fig_safety.svg": fig_safety, "E1_fig_dual.svg": fig_dual}
    for name, fn in figs.items():
        try:
            svg = fn(result)
            with open(os.path.join(outdir, name), "w", encoding="utf-8") as f:
                f.write(svg)
            print("  生成:", os.path.join("figures", name), f"({len(svg)} bytes)")
        except Exception as e:
            print("  跳过", name, ":", e)


if __name__ == "__main__":
    main()

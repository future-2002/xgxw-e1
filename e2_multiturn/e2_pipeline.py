# -*- coding: utf-8 -*-
"""
E2 实验 - 虚拟来访者多轮咨询过程评测（单文件管线）
==================================================

回答核心问题：**在多轮陪伴中，三极议会是否比单模型更能降低来访者痛苦、建立工作同盟、
促成认知转变？** 对标 WiseMind/CACTUS 的"模拟来访者多轮会话"范式，补充 E1（单轮质量）
之外的"过程性疗效"证据。

设计：
- 场景来自 CPsyCounD（真实咨询主题），取其开场主诉作为虚拟来访者的呈现问题；
- 虚拟来访者由 LLM 按人设扮演，携带初始痛苦度，随咨询师回应自然演变（有效则渐缓、敷衍则不动/加重）；
- 两条件：gvc（三极议会，跨轮记忆）vs single_llm（单模型+朴素咨询提示）；同场景配对；
- 过程指标由 glm-5.2 盲评：初/末痛苦度(0-10, SUDs)、工作同盟(WAI-SR: bond/goal/task 1-7)、
  认知转变(bool)、整体帮助度(1-7)。

子命令（容器内执行，复用 E1 的 conditions.py）：
  python e2_pipeline.py sample   --n 30
  python e2_pipeline.py generate --condition gvc        [--max-turns 6] [--limit N]
  python e2_pipeline.py generate --condition single_llm
  python e2_pipeline.py judge    [--condition gvc]
  python e2_pipeline.py analyze
"""
import argparse
import json
import logging
import os
import re
import sys
import time
import uuid

# 复用 E1 的调用层（call_gvc/call_direct/call_judge、conditions）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
E1_DIR = os.path.join(BASE_DIR, "..", "e1_counseling_quality")
sys.path.insert(0, E1_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, ".."))
import conditions as C  # noqa: E402
from statistical_tests import mean, std, confidence_interval, welch_t_test, cohens_d  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CORPUS = os.path.join(BASE_DIR, "..", "e1_corpus", "e1_samples_cpsycoun.json")
SCEN_PATH = os.path.join(BASE_DIR, "..", "e2_corpus", "e2_scenarios.json")
DIALOG_DIR = os.path.join(BASE_DIR, "..", "e2_results", "dialogues")
JUDGE_DIR = os.path.join(BASE_DIR, "..", "e2_results", "judgments")
OUT_DIR = os.path.join(BASE_DIR, "..", "e2_results")

DIMS = ["distress_reduction", "wai_total", "helpfulness"]

# 虚拟来访者系统提示：扮演求助者，携带痛苦度，随咨询质量自然演变
CLIENT_SYSTEM = """你在扮演一位前来寻求心理帮助的来访者（不是咨询师）。严格遵守：
1. 你的困扰是：{problem}
2. 你此刻的内心痛苦程度约为 {distress}/10，语气符合该痛苦水平。
3. 每次只说 1~3 句话，用第一人称，像真实来访者一样自然表达，可以有防御、犹豫、反问。
4. 依据咨询师的回应真实演变：若对方真正共情、理解你、给出对你有用的视角或方法，你可以流露一点点松动或愿意尝试；若对方敷衍、说教、答非所问，你会保持痛苦甚至更抵触。
5. 绝不跳出角色，不评价咨询技巧，不替咨询师说话，不输出与来访者身份无关的内容。
6. 当你觉得问题谈得差不多、或获得阶段性缓解时，可以自然地表达感谢并准备结束。"""

COUNSELOR_PROMPT = C.NAIVE_PROMPT  # single_llm 基线用朴素咨询师提示

PROCESS_JUDGE = """你是资深心理咨询督导。下面是一段来访者与心理支持者的多轮对话记录，请从**过程性疗效**角度盲评。

【对话记录】
{transcript}

【评分项】
- initial_distress: 来访者在对话开始时的主观痛苦程度(0-10)
- final_distress: 来访者在对话结束时的主观痛苦程度(0-10)
- wai_bond: 工作同盟-情感联结(1-7，来访者是否感到被理解、被尊重、被接纳)
- wai_goal: 工作同盟-目标一致(1-7，双方是否对困扰与方向达成共识)
- wai_task: 工作同盟-任务认同(1-7，支持者提供的方法/视角是否被来访者认可为有用)
- cognitive_shift: 来访者是否出现认知/视角的松动或转变(true/false)
- helpfulness: 这段陪伴对来访者的整体帮助程度(1-7)
严格按来访者在对话中真实表现判断，勿因回复长/措辞华丽而给高分。

只输出 JSON：
{{"initial_distress": 数值, "final_distress": 数值, "wai_bond": 数值, "wai_goal": 数值,
  "wai_task": 数值, "cognitive_shift": false, "helpfulness": 数值, "rationale": "50字内评语"}}"""


# ---------------- sample ----------------
def do_sample(args):
    with open(CORPUS, encoding="utf-8") as f:
        data = json.load(f)["samples"]
    scen = []
    for s in data:
        hist = s.get("history") or []
        if not hist:
            continue
        opening = hist[0].get("content", "").strip()  # 来访者开场主诉
        if len(opening) < 15:
            continue
        scen.append({"id": s["id"], "topic": s.get("topic", ""), "problem": opening})
        if len(scen) >= args.n:
            break
    os.makedirs(os.path.dirname(SCEN_PATH), exist_ok=True)
    with open(SCEN_PATH, "w", encoding="utf-8") as f:
        json.dump({"n": len(scen), "scenarios": scen}, f, ensure_ascii=False, indent=2)
    logger.info("E2 场景已生成 %d 条 -> %s", len(scen), SCEN_PATH)


# ---------------- generate (multi-turn dialogue) ----------------
def _client_turn(problem, distress, transcript):
    """虚拟来访者下一句：给客户端 LLM 全程对话，产出来访者回应"""
    sys_prompt = CLIENT_SYSTEM.format(problem=problem, distress=distress)
    convo = []
    for t in transcript:
        # 站在来访者视角：咨询师=对方(user)，自己=assistant
        convo.append({"role": "assistant" if t["role"] == "client" else "user",
                      "content": t["content"]})
    # 最后一条必为咨询师(user角色)：作为 message 传入，其余作 history，避免空 user 消息被 API 拒绝
    last = convo.pop()
    r = C.call_direct(last["content"], sys_prompt, history=convo, temperature=0.8)
    return r.get("response", "").strip()


def do_generate(args):
    with open(SCEN_PATH, encoding="utf-8") as f:
        scen = json.load(f)["scenarios"]
    if args.limit:
        scen = scen[: args.limit]
    if getattr(args, "end", 0):
        scen = scen[args.start: args.end]
    elif getattr(args, "start", 0):
        scen = scen[args.start:]
    os.makedirs(DIALOG_DIR, exist_ok=True)
    main_path = os.path.join(DIALOG_DIR, f"{args.condition}.jsonl")
    shard = getattr(args, "shard", "")
    out_path = os.path.join(DIALOG_DIR, f"{args.condition}.shard{shard}.jsonl") if shard else main_path
    # done 集合：主文件 + 所有分片，避免并行重复
    done = set()
    for p in [main_path] + sorted(__import__("glob").glob(os.path.join(DIALOG_DIR, f"{args.condition}.shard*.jsonl"))):
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    todo = [s for s in scen if s["id"] not in done]
    logger.info("[E2/%s] 待生成 %d / 共 %d", args.condition, len(todo), len(scen))

    with open(out_path, "a", encoding="utf-8") as f:
        for i, s in enumerate(todo, 1):
            transcript = [{"role": "client", "content": s["problem"]}]
            session_id = f"e2_{args.condition}_{s['id']}_{uuid.uuid4().hex[:6]}"
            ok = True
            for k in range(args.max_turns):
                client_msg = transcript[-1]["content"]
                if args.condition == "gvc":
                    res = C.call_gvc(client_msg, session_id, memory_enabled=True)
                else:
                    hist = [{"role": "user" if t["role"] == "client" else "assistant",
                             "content": t["content"]} for t in transcript[:-1]]
                    res = C.call_direct(client_msg, COUNSELOR_PROMPT, history=hist)
                reply = res.get("response", "").strip()
                if res.get("error") or not reply:
                    ok = False
                    logger.warning("  %s turn%d 咨询师失败: %s", s["id"], k, res.get("error"))
                    break
                transcript.append({"role": "counselor", "content": reply})
                if k < args.max_turns - 1:
                    cmsg = _client_turn(s["problem"], 7, transcript)
                    if not cmsg:
                        break
                    transcript.append({"role": "client", "content": cmsg})
                    # 来访者自然结束
                    if any(w in cmsg for w in ["谢谢你", "我会试试", "好多了", "再见", "感谢"]):
                        break
                time.sleep(0.3)
            if ok:
                rec = {"id": s["id"], "condition": args.condition, "topic": s["topic"],
                       "problem": s["problem"], "transcript": transcript,
                       "turns": sum(1 for t in transcript if t["role"] == "counselor"),
                       "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                logger.info("  [%d/%d] %s 完成 (%d 轮)", i, len(todo), s["id"], rec["turns"])


# ---------------- judge ----------------
def _fmt_transcript(t):
    role = {"client": "来访者", "counselor": "支持者"}
    return "\n".join(f"{role.get(x['role'], x['role'])}：{x['content']}" for x in t)


def do_judge(args):
    conds = [args.condition] if args.condition else \
        [os.path.basename(p)[:-6] for p in _glob(DIALOG_DIR)]
    for cond in conds:
        dpath = os.path.join(DIALOG_DIR, f"{cond}.jsonl")
        if not os.path.exists(dpath):
            continue
        recs = [json.loads(l) for l in open(dpath, encoding="utf-8") if l.strip()]
        os.makedirs(JUDGE_DIR, exist_ok=True)
        jpath = os.path.join(JUDGE_DIR, f"{cond}.jsonl")
        done = set()
        if os.path.exists(jpath):
            for l in open(jpath, encoding="utf-8"):
                try:
                    done.add(json.loads(l)["id"])
                except Exception:
                    pass
        todo = [r for r in recs if r["id"] not in done]
        logger.info("[E2/judge %s] 待评 %d", cond, len(todo))
        with open(jpath, "a", encoding="utf-8") as f:
            for r in todo:
                prompt = PROCESS_JUDGE.format(transcript=_fmt_transcript(r["transcript"]))
                try:
                    sc = _parse(C.call_judge(prompt))
                    out = {"id": r["id"], "condition": cond, "topic": r.get("topic", ""),
                           "initial_distress": sc["initial_distress"],
                           "final_distress": sc["final_distress"],
                           "distress_reduction": round(sc["initial_distress"] - sc["final_distress"], 2),
                           "wai_bond": sc["wai_bond"], "wai_goal": sc["wai_goal"], "wai_task": sc["wai_task"],
                           "wai_total": round(sc["wai_bond"] + sc["wai_goal"] + sc["wai_task"], 2),
                           "cognitive_shift": bool(sc.get("cognitive_shift")),
                           "helpfulness": sc["helpfulness"], "turns": r.get("turns", 0),
                           "rationale": sc.get("rationale", "")}
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    f.flush()
                    logger.info("  %s 痛苦降%.1f WAI%.1f 帮助%.1f", r["id"],
                                out["distress_reduction"], out["wai_total"], out["helpfulness"])
                except Exception as e:
                    logger.warning("  %s 评审失败: %s", r["id"], e)
                time.sleep(0.4)


def _glob(d):
    import glob
    return sorted(glob.glob(os.path.join(d, "*.jsonl")))


def _parse(raw):
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        obj = json.loads(m.group())
    while isinstance(obj, list) and obj:
        obj = obj[0]
    return obj


# ---------------- analyze ----------------
def do_analyze(args):
    data = {}
    for p in _glob(JUDGE_DIR):
        cond = os.path.basename(p)[:-6]
        recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        if recs:
            data[cond] = recs
    if not data:
        print("无 E2 评审结果，请先 judge")
        return
    result = {"conditions": {}, "comparison": {}}
    for cond, recs in data.items():
        st = {"n": len(recs)}
        for m in ["distress_reduction", "wai_total", "helpfulness", "initial_distress", "final_distress"]:
            v = [r[m] for r in recs if m in r]
            ci = confidence_interval(v)
            st[m] = {"mean": round(mean(v), 2), "std": round(std(v), 2),
                     "ci": [round(ci["ci_lower"], 2), round(ci["ci_upper"], 2)]}
        st["cognitive_shift_rate"] = round(mean([1 if r.get("cognitive_shift") else 0 for r in recs]), 3)
        st["avg_turns"] = round(mean([r.get("turns", 0) for r in recs]), 1)
        result["conditions"][cond] = st
    if "gvc" in data and "single_llm" in data:
        amap = {r["id"]: r for r in data["gvc"]}
        bmap = {r["id"]: r for r in data["single_llm"]}
        common = sorted(set(amap) & set(bmap))
        for m in DIMS:
            g = [amap[i][m] for i in common]
            s = [bmap[i][m] for i in common]
            if len(g) >= 3:
                t = welch_t_test(g, s)
                result["comparison"][m] = {"n_paired": len(common),
                    "gvc": round(mean(g), 2), "single_llm": round(mean(s), 2),
                    "diff": round(mean(g) - mean(s), 2), "p": round(t["p"], 5),
                    "cohens_d": round(cohens_d(g, s), 3)}
    ts = time.strftime("%Y%m%d_%H%M%S")
    jp = os.path.join(OUT_DIR, f"E2_analysis_{ts}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    # markdown
    lines = ["# E2 实验报告：虚拟来访者多轮咨询过程评测", "",
             f"生成时间：{ts}", "",
             "指标：痛苦度下降(SUDs, 越高越好) / 工作同盟 WAI(3项合计, 越高越好) / 认知转变率 / 帮助度",
             "", "| 条件 | N | 痛苦度下降 | 工作同盟 | 认知转变率 | 帮助度 | 平均轮数 |",
             "|---|---|---|---|---|---|---|"]
    labels = {"gvc": "三极议会(GVC)", "single_llm": "单模型直出"}
    for c, st in result["conditions"].items():
        lines.append(f"| {labels.get(c, c)} | {st['n']} | {st['distress_reduction']['mean']}±{st['distress_reduction']['std']} "
                     f"| {st['wai_total']['mean']} | {st['cognitive_shift_rate']:.0%} | {st['helpfulness']['mean']} | {st['avg_turns']} |")
    if result["comparison"]:
        lines += ["", "## GVC vs 单模型（Welch t 检验，配对场景）", "",
                  "| 指标 | GVC | 单模型 | 差值 | p | Cohen's d |", "|---|---|---|---|---|---|"]
        names = {"distress_reduction": "痛苦度下降", "wai_total": "工作同盟", "helpfulness": "帮助度"}
        for m, d in result["comparison"].items():
            lines.append(f"| {names.get(m, m)} | {d['gvc']} | {d['single_llm']} | {d['diff']:+} | {d['p']} | {d['cohens_d']} |")
    lines += ["", "> 说明：来访者由 LLM 按 CPsyCounD 真实主题人设扮演；过程指标由 glm-5.2 跨厂商盲评。"]
    mp = os.path.join(OUT_DIR, f"E2_report_{ts}.md")
    with open(mp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("E2 分析:", jp)
    print("E2 报告:", mp)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("sample"); s1.add_argument("--n", type=int, default=30)
    s2 = sub.add_parser("generate"); s2.add_argument("--condition", required=True, choices=["gvc", "single_llm"])
    s2.add_argument("--max-turns", type=int, default=6); s2.add_argument("--limit", type=int, default=0)
    s2.add_argument("--start", type=int, default=0); s2.add_argument("--end", type=int, default=0)
    s2.add_argument("--shard", default="")
    s3 = sub.add_parser("judge"); s3.add_argument("--condition", default=None)
    sub.add_parser("analyze")
    args = ap.parse_args()
    {"sample": do_sample, "generate": do_generate, "judge": do_judge, "analyze": do_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()

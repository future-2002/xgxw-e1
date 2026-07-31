# -*- coding: utf-8 -*-
"""
E1 实验 - 语料分层采样
======================

从公开真实求助语料中分层采样，构建 E1 咨询质量评测集：
- PsyQA（22341 条真实心理求助帖，含高赞人类回答）→ 单轮质量评测主语料
- CPsyCounD（3134 场多轮咨询对话）→ 多轮上下文评测语料（兼供 E2 虚拟来访者）

采样策略：
1. PsyQA 按话题关键词分层（成长/人际/恋爱/情绪/家庭/工作学习/其他），每层等比抽样
2. 只保留带 has_label=True 高质量人类回答的样本（人类对照组需要高质量回答）
3. 控制求助文本长度 50~800 字，去除过短/过长离群样本
4. 固定随机种子，采样结果可复现

用法（宿主机或容器内均可）：
    python3 sample_corpus.py --psyqa-n 200 --cpsycoun-n 100 --seed 42
"""
import argparse
import json
import os
import random
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "..", "e1_corpus", "raw")
OUT_DIR = os.path.join(BASE_DIR, "..", "e1_corpus")

# 分层话题（PsyQA keywords 高频话题归并）
TOPIC_STRATA = {
    "growth": ["成长", "自我成长", "困惑", "行为"],
    "relationship": ["人际", "沟通", "社交"],
    "love": ["恋爱", "恋爱经营", "婚姻"],
    "emotion": ["情绪", "情绪调节", "压力管理"],
    "family": ["家庭", "家庭关系", "亲子"],
    "study_work": ["工作学习", "学生成长", "职场"],
}


def classify_topic(keywords: str) -> str:
    """按关键词把样本归入分层话题，未命中归 other"""
    kws = [k.strip() for k in (keywords or "").split(",")]
    for topic, markers in TOPIC_STRATA.items():
        if any(k in markers for k in kws):
            return topic
    return "other"


def pick_human_answer(answers: list) -> str:
    """选带标注(has_label)中最长的人类回答作为人类对照组"""
    labeled = [a for a in answers if a.get("has_label")]
    if not labeled:
        return ""
    best = max(labeled, key=lambda a: len(a.get("answer_text", "")))
    return best.get("answer_text", "")


def sample_psyqa(n: int, rng: random.Random) -> list:
    """PsyQA 分层等比采样"""
    path = os.path.join(RAW_DIR, "PsyQA_full.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # 过滤：有高质量人类回答 + 描述长度合理
    buckets = defaultdict(list)
    for item in data:
        desc = (item.get("description") or "").strip()
        if not (50 <= len(desc) <= 800):
            continue
        human = pick_human_answer(item.get("answers", []))
        if len(human) < 100:
            continue
        topic = classify_topic(item.get("keywords", ""))
        buckets[topic].append({
            "source": "psyqa",
            "source_id": str(item.get("questionID", "")),
            "topic": topic,
            "question": (item.get("question") or "").strip(),
            "message": desc,
            "keywords": item.get("keywords", ""),
            "human_answer": human,
        })

    total = sum(len(v) for v in buckets.values())
    print(f"[PsyQA] 过滤后候选 {total} 条，按层分布: "
          + ", ".join(f"{k}={len(v)}" for k, v in sorted(buckets.items())))

    # 等比分配（每层至少 max(1, n//len//2) 条，余量按比例）
    samples = []
    quotas = {}
    for topic, items in buckets.items():
        quotas[topic] = max(1, round(n * len(items) / total))
    # 调整到恰好 n
    diff = n - sum(quotas.values())
    for topic in sorted(quotas, key=lambda t: -len(buckets[t])):
        if diff == 0:
            break
        step = 1 if diff > 0 else -1
        quotas[topic] += step
        diff -= step

    for topic, quota in quotas.items():
        pool = buckets[topic]
        rng.shuffle(pool)
        samples.extend(pool[:quota])

    rng.shuffle(samples)
    for i, s in enumerate(samples):
        s["id"] = f"psyqa_{i:04d}"
    return samples


def sample_cpsycoun(n: int, rng: random.Random) -> list:
    """CPsyCounD 采样：抽取多轮对话，保留 history 供多轮评测/E2 使用"""
    path = os.path.join(RAW_DIR, "CPsyCounD.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    candidates = []
    for item in data:
        history = item.get("history") or []
        # 至少 3 轮上下文，当前求助句长度合理
        msg = (item.get("instruction") or "").strip()
        if len(history) < 3 or not (10 <= len(msg) <= 500):
            continue
        candidates.append({
            "source": "cpsycoun",
            "topic": "multi_turn",
            "history": [{"role": "user" if i % 2 == 0 else "assistant", "content": t}
                        for pair in history for i, t in enumerate(pair)],
            "message": msg,
            "reference_answer": (item.get("output") or "").strip(),
        })

    rng.shuffle(candidates)
    samples = candidates[:n]
    for i, s in enumerate(samples):
        s["id"] = f"cpsy_{i:04d}"
    print(f"[CPsyCounD] 候选 {len(candidates)} 场，采样 {len(samples)} 场")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--psyqa-n", type=int, default=200)
    parser.add_argument("--cpsycoun-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    psyqa = sample_psyqa(args.psyqa_n, rng)
    out1 = os.path.join(OUT_DIR, "e1_samples_psyqa.json")
    with open(out1, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "n": len(psyqa), "samples": psyqa},
                  f, ensure_ascii=False, indent=2)
    print(f"[PsyQA] 已写入 {out1}（{len(psyqa)} 条）")

    cpsy = sample_cpsycoun(args.cpsycoun_n, rng)
    out2 = os.path.join(OUT_DIR, "e1_samples_cpsycoun.json")
    with open(out2, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "n": len(cpsy), "samples": cpsy},
                  f, ensure_ascii=False, indent=2)
    print(f"[CPsyCounD] 已写入 {out2}（{len(cpsy)} 条）")


if __name__ == "__main__":
    main()

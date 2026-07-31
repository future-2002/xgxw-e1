# -*- coding: utf-8 -*-
"""
E1 实验 - 回复生成 Runner
=========================

对指定条件批量生成回复，JSONL 落盘，支持断点续跑（按 sample id 去重）。

容器内用法（由 scripts/run_e1_experiment.sh 编排调用）：
    python3 run_generation.py --condition gvc_full
    python3 run_generation.py --condition single_llm --limit 5   # 冒烟测试
"""
import argparse
import json
import logging
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conditions import CONDITIONS, call_gvc, call_direct  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(BASE_DIR, "..", "e1_corpus", "e1_samples_psyqa.json")
GEN_DIR = os.path.join(BASE_DIR, "..", "e1_results", "generations")


def load_samples() -> list:
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return json.load(f)["samples"]


def load_done_ids(out_path: str) -> set:
    """已生成的样本 id（断点续跑；同时合并主文件与分片文件的已完成集，避免并行分片重复）"""
    import glob as _glob
    done = set()
    base = out_path[:-6] if out_path.endswith(".jsonl") else out_path
    for p in [out_path] + sorted(_glob.glob(base + ".shard*.jsonl")):
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("response"):
                        done.add(rec["id"])
                except json.JSONDecodeError:
                    continue
    return done


def generate_one(condition: str, cfg: dict, sample: dict) -> dict:
    """按条件生成单条回复"""
    msg = sample["message"]
    if cfg["backend"] == "gvc":
        # 每条样本独立 session，避免记忆串扰
        session_id = f"e1_{condition}_{sample['id']}_{uuid.uuid4().hex[:6]}"
        result = call_gvc(msg, session_id)
    elif cfg["backend"] == "direct":
        result = call_direct(msg, cfg["system_prompt"])
    elif cfg["backend"] == "human":
        result = {"response": sample.get("human_answer", ""), "latency": 0}
    else:
        raise ValueError(f"未知 backend: {cfg['backend']}")

    return {
        "id": sample["id"],
        "condition": condition,
        "topic": sample["topic"],
        "message": msg,
        "response": result.get("response", ""),
        "latency": result.get("latency", 0),
        "error": result.get("error"),
        "events": result.get("events", []),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=list(CONDITIONS.keys()))
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟测试）")
    parser.add_argument("--sleep", type=float, default=1.0, help="条间隔秒数")
    parser.add_argument("--start", type=int, default=0, help="分片：样本起始下标（含）")
    parser.add_argument("--end", type=int, default=0, help="分片：样本结束下标（不含，0=到尾）")
    parser.add_argument("--shard", default="", help="分片名：输出写 {condition}.shard{name}.jsonl，跑完后需合并")
    args = parser.parse_args()

    cfg = CONDITIONS[args.condition]
    samples = load_samples()
    if args.limit:
        samples = samples[: args.limit]
    if args.end:
        samples = samples[args.start: args.end]
    elif args.start:
        samples = samples[args.start:]

    os.makedirs(GEN_DIR, exist_ok=True)
    main_path = os.path.join(GEN_DIR, f"{args.condition}.jsonl")
    out_path = os.path.join(GEN_DIR, f"{args.condition}.shard{args.shard}.jsonl") if args.shard else main_path
    done = load_done_ids(main_path)
    todo = [s for s in samples if s["id"] not in done]
    logger.info("[%s] 总样本 %d，已完成 %d，待生成 %d",
                args.condition, len(samples), len(done), len(todo))

    ok, fail = 0, 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i, sample in enumerate(todo, 1):
            rec = generate_one(args.condition, cfg, sample)
            if rec.get("error") or not rec["response"]:
                fail += 1
                logger.warning("  [%d/%d] %s 失败: %s", i, len(todo), sample["id"], rec.get("error"))
            else:
                ok += 1
                logger.info("  [%d/%d] %s 完成 (%.1fs, %d字)",
                            i, len(todo), sample["id"], rec["latency"], len(rec["response"]))
            # 失败的不落盘，便于重跑补齐
            if rec["response"]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            if args.sleep and cfg["backend"] != "human":
                time.sleep(args.sleep)

    logger.info("[%s] 完成：成功 %d，失败 %d，输出 %s", args.condition, ok, fail, out_path)
    if fail:
        logger.info("提示：重新执行同一命令即可补齐失败样本")


if __name__ == "__main__":
    main()

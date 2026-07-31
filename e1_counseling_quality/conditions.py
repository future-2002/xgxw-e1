# -*- coding: utf-8 -*-
"""
E1 实验 - 对照条件定义与调用层
==============================

七个对照条件：
| 条件                    | 后端      | 说明                                   |
|-------------------------|-----------|----------------------------------------|
| gvc_full                | 系统 API  | 三极议会完整版（PARLIAMENT_EXPERIMENT=baseline）|
| ablation_no_ca          | 系统 API  | 消融：去 CA 验证器                     |
| ablation_no_skg         | 系统 API  | 消融：去 SKG 知识图谱                  |
| ablation_no_convergence | 系统 API  | 消融：去 EVINCE 收敛判据（固定轮次）   |
| single_llm              | 云端直连  | 单模型 + 朴素咨询师提示词              |
| cbt_prompt              | 云端直连  | 单模型 + 结构化 CBT 提示词（强基线）   |
| human                   | 语料自带  | PsyQA 高赞人类回答                     |

注意：gvc_* 条件要求 ai-service 以对应 PARLIAMENT_EXPERIMENT 环境变量启动，
由宿主机编排脚本 scripts/run_e1_experiment.sh 负责切换。
"""
import json
import os
import time
import logging

import httpx

logger = logging.getLogger(__name__)

# 容器内调用本服务；宿主机调试时可覆盖 E1_GVC_API
GVC_API = os.getenv("E1_GVC_API", "http://localhost:8000/api/chat/stream/deep")
# ai-service 内部认证头（/api/ 路径需 X-Internal-API-Key，与容器 INTERNAL_API_KEY 环境变量一致）
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")

# 生成用云端 LLM（与 ai-service 共用环境变量，容器内天然可用）
# 统一底座：qwen3.7-max（GVC 各智能体 + single_llm/cbt_prompt 基线全部同底座，保证对比公平）
LLM_API_URL = os.getenv("LLM_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.7-max")

# 盲评用模型（跨厂商评审：生成用 qwen，评审用 glm，消除自评偏置）
# 默认与生成共用同一 MaaS 网关；如评审换独立厂商，用 E1_JUDGE_API_URL / E1_JUDGE_API_KEY 覆盖
JUDGE_API_URL = os.getenv("E1_JUDGE_API_URL", LLM_API_URL)
JUDGE_API_KEY = os.getenv("E1_JUDGE_API_KEY", LLM_API_KEY)
JUDGE_MODEL = os.getenv("E1_JUDGE_MODEL", "glm-5.2")

# 朴素咨询师提示词（single_llm 基线）
NAIVE_PROMPT = "你是一位温暖、专业的心理咨询助手，请认真倾听用户的困扰，提供共情和有帮助的回应。"

# 结构化 CBT 提示词（cbt_prompt 强基线，对齐 AutoCBT 单智能体范式）
CBT_PROMPT = """你是一位受过认知行为疗法（CBT）训练的心理咨询师。请按以下框架回应求助者：
1. 共情接纳：先理解并确认求助者的情绪感受，创造安全的表达环境；
2. 识别认知：识别求助者叙述中可能存在的认知扭曲（如灾难化、非黑即白、过度概括、贴标签等）；
3. 引导反思：提出1-2个开放式问题，帮助求助者审视自己的想法；
4. 提供策略：给出基于CBT的具体、可操作的应对策略；
5. 鼓励行动：鼓励求助者尝试策略，并预见可能的困难，给予支持。
要求：语言温暖自然，不使用诊断性断言（如"你患有抑郁症"），不评判求助者。"""

# 条件注册表
CONDITIONS = {
    "gvc_full": {"backend": "gvc", "parliament_env": "baseline"},
    "gvc_v2": {"backend": "gvc", "parliament_env": "gvc_v2"},
    "ablation_no_ca": {"backend": "gvc", "parliament_env": "ablation_no_ca"},
    "ablation_no_skg": {"backend": "gvc", "parliament_env": "ablation_no_skg"},
    "ablation_no_convergence": {"backend": "gvc", "parliament_env": "ablation_no_convergence"},
    "single_llm": {"backend": "direct", "system_prompt": NAIVE_PROMPT},
    "cbt_prompt": {"backend": "direct", "system_prompt": CBT_PROMPT},
    "human": {"backend": "human"},
}


def call_gvc(message: str, session_id: str, history: list = None,
             timeout: float = 600.0, max_retries: int = 3,
             memory_enabled: bool = False) -> dict:
    """调用三极议会深度模式（流式 NDJSON，聚合 chunk），带 429 退避重试
    注：深度模式多轮议会 + qwen3.7-max 单样本可需 200-300s，超时给 600s
    memory_enabled：E1 单轮置 False；E2 多轮置 True（依赖同一 session 的跨轮记忆）"""
    body = {
        "user_id": "e1_experiment",
        "session_id": session_id,
        "message": message,
        "memory_enabled": memory_enabled,
        "web_search": False,
    }
    if history:
        body["history"] = history

    for attempt in range(1, max_retries + 1):
        start = time.time()
        try:
            chunks, events = [], []
            with httpx.stream("POST", GVC_API, json=body,
                              headers={"Content-Type": "application/json",
                                       "X-Internal-API-Key": INTERNAL_API_KEY},
                              timeout=timeout) as r:
                if r.status_code == 429:
                    wait = 30 * attempt
                    logger.warning("限流(429)，等待 %ds 后重试 (%d/%d)", wait, attempt, max_retries)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if e.get("type") == "chunk":
                        chunks.append(e.get("content", ""))
                    elif e.get("type") in ("meta_plan", "evince_metrics", "conflict_detected"):
                        events.append(e)
            return {
                "response": "".join(chunks),
                "latency": round(time.time() - start, 2),
                "events": events,
            }
        except Exception as exc:
            logger.warning("GVC 调用失败 (%d/%d): %s", attempt, max_retries, exc)
            if attempt == max_retries:
                return {"error": str(exc), "latency": round(time.time() - start, 2)}
            time.sleep(10 * attempt)
    return {"error": "exhausted retries", "latency": 0}


def call_direct(message: str, system_prompt: str, history: list = None,
                model: str = None, temperature: float = 0.7,
                timeout: float = 120.0, max_retries: int = 3) -> dict:
    """云端 LLM 直连（OpenAI 兼容协议），用于 single_llm / cbt_prompt 基线"""
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": message})

    body = {
        "model": model or LLM_MODEL,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
    }
    for attempt in range(1, max_retries + 1):
        start = time.time()
        try:
            r = httpx.post(LLM_API_URL, json=body, timeout=timeout,
                           headers={"Authorization": f"Bearer {LLM_API_KEY}",
                                    "Content-Type": "application/json"})
            if r.status_code == 429:
                time.sleep(15 * attempt)
                continue
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return {"response": content, "latency": round(time.time() - start, 2)}
        except Exception as exc:
            logger.warning("直连 LLM 失败 (%d/%d): %s", attempt, max_retries, exc)
            if attempt == max_retries:
                return {"error": str(exc), "latency": round(time.time() - start, 2)}
            time.sleep(5 * attempt)
    return {"error": "exhausted retries", "latency": 0}


def call_judge(prompt: str, model: str = None, timeout: float = 120.0,
               max_retries: int = 3) -> str:
    """LLM-as-Judge 调用（低温、JSON 输出，默认 glm-5.2 跨厂商评审）"""
    body = {
        "model": model or JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(1, max_retries + 1):
        try:
            r = httpx.post(JUDGE_API_URL, json=body, timeout=timeout,
                           headers={"Authorization": f"Bearer {JUDGE_API_KEY}",
                                    "Content-Type": "application/json"})
            if r.status_code == 429:
                time.sleep(15 * attempt)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.warning("Judge 调用失败 (%d/%d): %s", attempt, max_retries, exc)
            if attempt == max_retries:
                raise
            time.sleep(5 * attempt)
    raise RuntimeError("judge exhausted retries")

# -*- coding: utf-8 -*-
"""
================================================================
LLM 客户端抽象层 (llm_client.py)
================================================================
本文件提供统一的 LLM 调用接口 LLMClient，支持以下四种后端切换：
   1. "deepseek"           —— 调用 DeepSeek 商用 API, 默认模型 deepseek-v4-flash（OpenAI 兼容格式）
  2. "openai"             —— 调用 OpenAI 官方 API
  3. "local_vllm"         —— 调用本地 vLLM 部署的开源模型
  4. "local_transformers" —— 直接用 transformers 推理（开发调试）

通过在 src/config.py 中修改 LLM_BACKEND 即可切换后端，业务代码不变。
本文件还提供：
  - 断点续存对话日志的辅助函数 (save_log/load_log)
  - 自动重试与速率限制退避
  - 可选的内容 JSON 解析（Agent 输出结构化响应时使用）
================================================================
"""

import os
import json
import time
import logging
import threading

from config import (
    LLM_BACKEND, LLM_API_KEY, LLM_API_BASE, LLM_MODEL,
    QWEN_API_KEY, QWEN_API_BASE,
    LLM_LOCAL_MODEL_PATH, LLM_LOCAL_PORT,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_REQUEST_DELAY,
    LLM_RATE_LIMIT_RPM, LLM_THINKING_ENABLED,
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)


# ============================================================
# 滑动窗口速率限制器（全局, 线程安全）
# ============================================================
class RateLimiter:
    """滑动窗口速率限制器。

    限制: 过去 `window_sec` 秒内至多 `max_requests` 次调用。
    超出时 sleep 直到最早请求移出窗口。

    所有方法加锁, 线程安全。
    """

    def __init__(self, max_requests: int = 2500, window_sec: float = 60.0):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self.timestamps: list = []
        self._lock = threading.Lock()

    def wait(self):
        """等待直到允许发出下一个请求。"""
        while True:
            with self._lock:
                now = time.time()
                # 移除窗口外记录
                self.timestamps = [t for t in self.timestamps if now - t < self.window_sec]
                if len(self.timestamps) < self.max_requests:
                    self.timestamps.append(now)
                    return
            # 窗口已满, 等最早记录过期再试
            earliest = min(self.timestamps) if self.timestamps else 0.0
            sleep_time = (earliest + self.window_sec) - now
            if sleep_time > 0:
                time.sleep(sleep_time)


# 全局共享速率限制器实例（从 config 读取限制数）
_GLOBAL_RATE_LIMITER = RateLimiter(max_requests=LLM_RATE_LIMIT_RPM, window_sec=60.0)


# ============================================================
# LLMClient 基类与具体实现
# ============================================================
class LLMClient:
    """统一的 LLM 客户端抽象基类。

    所有具体后端实现子类需要重写：
        chat(self, system, user, temperature, max_tokens) -> str
    """

    def __init__(self, backend=LLM_BACKEND):
        self.backend = backend
        self._last_call_time = 0.0
        self._rate_limiter = _GLOBAL_RATE_LIMITER
        # token 用量统计 (vLLM/OpenAI 兼容后端返回 usage 时累加),
        # 供 token 成本随规模增长实验 (E20) 使用
        self.usage_stats = {'prompt_tokens': 0, 'completion_tokens': 0,
                            'calls': 0}

    # -------- 公共接口 --------
    def chat(self, system, user, temperature=LLM_TEMPERATURE,
             max_tokens=LLM_MAX_TOKENS):
        """调用大模型并返回自然语言响应。

        参数：
            system:      system prompt（str）
            user:        user prompt（str）
            temperature: 采样温度
            max_tokens:  最大生成 token 数
        返回：str，模型响应原文
        """
        self._respect_rate_limit()
        try:
            result = self._chat_impl(system, user, temperature, max_tokens)
        except Exception as e:
            logger.warning(f"LLM 调用失败（{self.backend}），重试 1：{e}")
            time.sleep(2.0)
            try:
                result = self._chat_impl(system, user, temperature, max_tokens)
            except Exception as e2:
                logger.error(f"LLM 调用二次失败：{e2}")
                result = ""
        return result

    def chat_json(self, system, user,
                  temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS):
        """调用大模型并尝试把响应解析为 JSON 字典。

        若解析失败，返回空字典 {}。对 Markdown 代码块包裹的 JSON
        也能自动剥离 ```json ... ``` 标记。
        """
        text = self.chat(system, user, temperature, max_tokens)
        return parse_json_response(text)

    # -------- 内部 --------
    def _chat_impl(self, system, user, temperature, max_tokens):
        raise NotImplementedError("子类必须实现 _chat_impl")

    def _respect_rate_limit(self):
        """两次调用之间至少间隔 LLM_REQUEST_DELAY 秒，避免触发速率限制。"""
        elapsed = time.time() - self._last_call_time
        if elapsed < LLM_REQUEST_DELAY:
            time.sleep(LLM_REQUEST_DELAY - elapsed)
        self._last_call_time = time.time()


# ---------------- DeepSeek / OpenAI 兼容实现 ----------------
class _OpenAICompatibleClient(LLMClient):
    """兼容 OpenAI 接口的客户端，DeepSeek / OpenAI 共用此实现。"""

    def __init__(self, model=LLM_MODEL, api_key=LLM_API_KEY,
                 base_url=LLM_API_BASE, backend="deepseek"):
        super().__init__(backend=backend)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "未安装 openai 包，请运行 pip install openai。") from e
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def _chat_impl(self, system, user, temperature, max_tokens):
        # DeepSeek 专属参数(如 thinking)需通过 extra_body 传递,
        # 因为 OpenAI SDK 标准参数表中不包含这些字段。
        extra = {}
        if self.backend == "deepseek" and not LLM_THINKING_ENABLED:
            extra["thinking"] = {"type": "disabled"}
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra if extra else None,
        )
        return resp.choices[0].message.content.strip()


# ---------------- 本地 vLLM 实现 ----------------
class _VLLMClient(LLMClient):
    """若用户在本地用 vLLM 起一个服务（OpenAI 兼容），
    则将其视为「OpenAI 兼容客户端」访问 http://localhost:LLM_LOCAL_PORT。
    """
    def __init__(self):
        super().__init__(backend="local_vllm")
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "未安装 openai 包，请运行 pip install openai。") from e
        base_url = f"http://localhost:{LLM_LOCAL_PORT}/v1"
        # vLLM 通常用空 api_key 即可
        self.client = OpenAI(api_key="EMPTY", base_url=base_url)
        self.model = os.path.basename(LLM_LOCAL_MODEL_PATH.rstrip("/"))

    def _chat_impl(self, system, user, temperature, max_tokens):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        u = getattr(resp, 'usage', None)
        if u is not None:
            self.usage_stats['prompt_tokens'] += int(u.prompt_tokens)
            self.usage_stats['completion_tokens'] += int(u.completion_tokens)
            self.usage_stats['calls'] += 1
        return resp.choices[0].message.content.strip()


# ---------------- 本地 transformers 实现（开发调试用） ----------------
class _TransformersClient(LLMClient):
    """单进程 transformers 推理，速度较慢，仅用于本地无网络环境调试。"""

    def __init__(self):
        super().__init__(backend="local_transformers")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "未安装 torch / transformers，请运行 pip install torch transformers。"
            ) from e
        self.tok = AutoTokenizer.from_pretrained(
            LLM_LOCAL_MODEL_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_LOCAL_MODEL_PATH, trust_remote_code=True,
            torch_dtype=torch.float16,
        ).cuda() if torch.cuda.is_available() else AutoModelForCausalLM.from_pretrained(
            LLM_LOCAL_MODEL_PATH, trust_remote_code=True)
        self.model.eval()

    def _chat_impl(self, system, user, temperature, max_tokens):
        import torch
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n" \
                 f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=max(temperature, 1e-5),
            )
        text = self.tok.decode(out[0][inputs['input_ids'].shape[1]:],
                               skip_special_tokens=True)
        return text.strip()


# ---------------- 客户端工厂 ----------------
def get_llm_client(backend=None, model=None):
    """根据后端名返回一个 LLMClient 实例。

    参数：
        backend: 可选，若为 None 则读取 config.LLM_BACKEND
        model:   可选，覆盖 config.LLM_MODEL
    """
    backend = backend or LLM_BACKEND
    effective_model = model or LLM_MODEL
    if backend == "deepseek":
        return _OpenAICompatibleClient(model=effective_model,
                                       api_key=LLM_API_KEY,
                                       base_url=LLM_API_BASE,
                                       backend="deepseek")
    if backend == "openai":
        return _OpenAICompatibleClient(model=effective_model,
                                       api_key=LLM_API_KEY,
                                       base_url=LLM_API_BASE,
                                       backend="openai")
    if backend == "qwen":
        return _OpenAICompatibleClient(model=effective_model,
                                       api_key=QWEN_API_KEY,
                                       base_url=QWEN_API_BASE,
                                       backend="qwen")
    if backend == "local_vllm":
        return _VLLMClient()
    if backend == "local_transformers":
        return _TransformersClient()
    raise ValueError(f"未知 LLM 后端：{backend}")


# ============================================================
# JSON 响应解析工具
# ============================================================
def parse_json_response(text):
    """把 LLM 的响应解析为字典。

    自动处理以下情形：
      - 响应本身即 JSON
      - 响应被 ```json ... ``` 包裹
      - 响应前后包含对干扰文字
    """
    if not text:
        return {}
    text = text.strip()
    # 剥 Markdown 代码块
    if text.startswith("```"):
        lines = text.split("```")
        # 通常 ['\n', 'json\n{...}\n', '\n']
        if len(lines) >= 2:
            inner = lines[1]
            if inner.startswith("json"):
                inner = inner[4:]
            text = inner.strip()
    # 尝试截取第一个 { 到最后一个 }
    try:
        return json.loads(text)
    except Exception:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if 0 <= start < end:
                return json.loads(text[start:end + 1])
        except Exception:
            pass
    logger.warning(f"无法解析 LLM 响应为 JSON: {text[:200]}")
    return {}


# ============================================================
# 断点续存：保存 / 加载已完成任务日志
# ============================================================
def save_progress(log_path, record):
    """把一条 record 追加写入日志文件（JSON Lines 格式）。

    参数：
        log_path: 文件路径
        record:   dict
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_progress(log_path):
    """读取已完成的记录列表，便于中断后恢复。

    返回：list[dict]
    """
    if not os.path.exists(log_path):
        return []
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def has_done(log_path, fingerprint):
    """检查某条任务（通过 fingerprint 字符串唯一标识）是否已记录。

    参数：
        log_path:    日志路径
        fingerprint: 任务唯一指纹（如 f"K{K}-M{M}-ep{ep}-step{t}"）
    返回：True / False
    """
    for r in load_progress(log_path):
        if r.get("fingerprint") == fingerprint:
            return True
    return False
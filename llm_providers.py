#!/usr/bin/env python3
"""
LLM Provider 抽象层 — 所有 Provider 均采用 OpenAI 标准格式
只需在 llm_config.json 中切换 provider 字段即可切换模型
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

# ─────────────────────────────────────────────
# 配置加载
# ─────────────────────────────────────────────

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "llm_config.json")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class LLMMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# 抽象接口
# ─────────────────────────────────────────────

class LLMProvider(ABC):
    """LLM Provider 抽象基类 — 所有 provider 必须实现 chat()"""

    @abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        **kwargs
    ) -> LLMResponse:
        raise NotImplementedError


# ─────────────────────────────────────────────
# OpenAI 标准格式 Provider
# 所有 OpenAI 兼容接口均使用此类，包括：OpenAI、Kimi、MiniMax、Claude（需代理）、Gemini 等
# ─────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    """
    OpenAI 标准格式 Provider
    通过环境变量或 llm_config.json 配置 api_key / model / base_url
    """

    def __init__(self, api_key: str, model: str,
                 base_url: str = "https://api.openai.com/v1",
                 organization: str = ""):
        if not api_key:
            raise ValueError("API Key 未设置，请设置环境变量")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.organization = organization

    def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: int = 120,
        stream: bool = False,
        stream_callback=None,
        **kwargs
    ) -> LLMResponse:
        import urllib.request
        import urllib.error

        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if stream:
                    return self._handle_stream(resp, stream_callback)
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise RuntimeError(f"API 错误 {e.code}：{error_body}")

        choices = result.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""

        return LLMResponse(
            content=content,
            model=self.model,
            usage=result.get("usage", {}),
            raw=result
        )

    def _handle_stream(self, resp, stream_callback=None) -> LLMResponse:
        """处理 SSE 流式输出"""
        accumulated = ""

        try:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content_delta = delta.get("content", "") or delta.get("text", "")
                if content_delta:
                    accumulated += content_delta
                    if stream_callback:
                        stream_callback(content_delta)
        except Exception as e:
            raise RuntimeError(f"流式读取错误：{e}")

        return LLMResponse(
            content=accumulated,
            model=self.model,
            usage={},
            raw={}
        )


# ─────────────────────────────────────────────
# Provider 工厂
# ─────────────────────────────────────────────

_PROVIDER_CACHE: dict[str, LLMProvider] = {}


def get_provider(name: str | None = None) -> LLMProvider:
    """
    根据配置名称返回对应 LLM Provider 实例（单例缓存）。

    优先级：环境变量 > llm_config.json
    支持的 provider：kimi / minimax / openai_compatible

    环境变量：
      LLM_PROVIDER        选择 provider
      KIMI_API_KEY        Kimi API Key
      KIMI_MODEL          Kimi 模型（默认 moonshot-v1-128k）
      KIMI_BASE_URL       Kimi API 地址
      MINIMAX_API_KEY     MiniMax API Key
      MINIMAX_MODEL       MiniMax 模型（默认 MiniMax-M2.7-highspeed）
      MINIMAX_BASE_URL    MiniMax API 地址
      OPENAI_API_KEY      OpenAI 兼容 API Key
      OPENAI_MODEL        模型名称（默认 gpt-4o）
      OPENAI_BASE_URL     API 地址（默认 https://api.openai.com/v1）
    """
    global _PROVIDER_CACHE

    config = load_config()
    provider_name = (
        os.environ.get("LLM_PROVIDER")
        or name
        or config.get("provider", "kimi")
    )

    if provider_name in _PROVIDER_CACHE:
        return _PROVIDER_CACHE[provider_name]

    if provider_name == "kimi":
        cfg = config.get("kimi", {})
        _PROVIDER_CACHE[provider_name] = OpenAIProvider(
            api_key=os.environ.get("KIMI_API_KEY", "") or cfg.get("api_key", ""),
            model=os.environ.get("KIMI_MODEL") or cfg.get("model", "moonshot-v1-128k"),
            base_url=os.environ.get("KIMI_BASE_URL") or cfg.get("base_url", "https://api.moonshot.cn/v1"),
            organization="",
        )

    elif provider_name == "minimax":
        cfg = config.get("minimax", {})
        _PROVIDER_CACHE[provider_name] = OpenAIProvider(
            api_key=os.environ.get("MINIMAX_API_KEY", ""),
            model=os.environ.get("MINIMAX_MODEL") or cfg.get("model", "MiniMax-M2.7-highspeed"),
            base_url=os.environ.get("MINIMAX_BASE_URL") or cfg.get("base_url", "https://api.minimaxi.com/v1"),
            organization="",
        )

    elif provider_name in ("openai", "openai_compatible", "azure", "claude", "gemini"):
        cfg = config.get("openai_compatible", {})
        _PROVIDER_CACHE[provider_name] = OpenAIProvider(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=os.environ.get("OPENAI_MODEL") or cfg.get("model", "gpt-4o"),
            base_url=os.environ.get("OPENAI_BASE_URL") or cfg.get("base_url", "https://api.openai.com/v1"),
            organization=cfg.get("organization", ""),
        )

    else:
        raise ValueError(f"未知的 Provider：{provider_name}，请检查 llm_config.json 中的 provider 字段")

    return _PROVIDER_CACHE[provider_name]


def reset_provider_cache():
    """清除缓存，强制重新加载配置"""
    global _PROVIDER_CACHE
    _PROVIDER_CACHE = {}

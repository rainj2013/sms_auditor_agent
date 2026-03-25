#!/usr/bin/env python3
"""
LLM Provider 抽象层 — 基于 OpenAI SDK
只需在 llm_config.json 中切换 provider 字段即可切换模型
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from openai import OpenAI

# ─────────────────────────────────────────────
# 配置加载
# ─────────────────────────────────────────────

# 向上查找项目根目录
def _find_root():
    path = os.path.dirname(__file__)
    while path != os.path.dirname(path):
        if os.path.exists(os.path.join(path, "llm_config.json")):
            return path
        path = os.path.dirname(path)
    return os.path.dirname(__file__)

CONFIG_PATH = os.path.join(_find_root(), "llm_config.json")


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
# OpenAI SDK Provider
# ─────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    """基于 OpenAI SDK 的 Provider，支持 OpenAI 兼容接口"""

    def __init__(self, api_key: str, model: str,
                 base_url: str = "https://api.openai.com/v1",
                 organization: str = ""):
        if not api_key:
            raise ValueError("API Key 未设置，请设置环境变量")
        self.model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            organization=organization or None,
            timeout=120,
        )

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
        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        try:
            if stream:
                accumulated = ""
                stream_resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    timeout=timeout,
                )
                for chunk in stream_resp:
                    content_delta = chunk.choices[0].delta.content or ""
                    if content_delta:
                        accumulated += content_delta
                        if stream_callback:
                            stream_callback(content_delta)
                return LLMResponse(
                    content=accumulated,
                    model=self.model,
                    usage={},
                    raw={},
                )
            else:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                return LLMResponse(
                    content=response.choices[0].message.content or "",
                    model=response.model,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0,
                    },
                    raw=response.model_dump(),
                )
        except Exception as e:
            raise RuntimeError(f"API 调用失败：{e}")


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

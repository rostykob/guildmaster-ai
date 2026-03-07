"""LLM provider abstractions and implementations.

All providers are optional extras — import them only when the corresponding
SDK is installed.  The ``BaseLLMProvider``, ``LLMMessage``, and ``LLMResponse``
types are always available.
"""

from guildmaster_ai.llm.base_provider import BaseLLMProvider, LLMMessage, LLMResponse
from guildmaster_ai.llm.anthropic import AnthropicProvider
from guildmaster_ai.llm.azure import AzureOpenAIProvider
from guildmaster_ai.llm.bedrock import BedrockProvider
from guildmaster_ai.llm.google import GoogleProvider
from guildmaster_ai.llm.openai import OpenAIProvider
from guildmaster_ai.llm.openrouter import OpenRouterProvider

__all__ = [
    "BaseLLMProvider",
    "LLMMessage",
    "LLMResponse",
    # Providers
    "AnthropicProvider",
    "AzureOpenAIProvider",
    "BedrockProvider",
    "GoogleProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
]

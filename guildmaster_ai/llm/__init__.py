"""LLM provider abstractions built on LangChain.

Users should import from this package — never from ``langchain_core``
directly.
"""

from guildmaster_ai.llm.base_provider import create_chat_model
from guildmaster_ai.llm.openrouter import ChatOpenRouter
from guildmaster_ai.llm.types import GuildLLM, GuildResponse, guild_complete

__all__ = [
    "ChatOpenRouter",
    "GuildLLM",
    "GuildResponse",
    "create_chat_model",
    "guild_complete",
]

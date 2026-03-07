from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from guildmaster_ai.llm.base_provider import BaseLLMProvider, LLMMessage, LLMResponse

try:
    import boto3 as _boto3

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SDK_AVAILABLE = False


class BedrockProvider(BaseLLMProvider):
    """AWS Bedrock LLM provider using ``boto3`` and the Converse API.

    Install with::

        pip install "guildmaster-ai[bedrock]"

    Authentication via standard AWS credential chain::

        AWS_ACCESS_KEY_ID=...
        AWS_SECRET_ACCESS_KEY=...
        AWS_DEFAULT_REGION=us-east-1   # or AWS_REGION

    Alternatively the default boto3 credential chain is used (IAM role,
    ``~/.aws/credentials``, etc.).

    The ``default_model`` should be a fully-qualified Bedrock model ID, e.g.::

        "anthropic.claude-3-5-sonnet-20241022-v2:0"
        "amazon.nova-pro-v1:0"
        "meta.llama3-70b-instruct-v1:0"

    Bedrock's Converse API is synchronous; calls are dispatched to a thread
    pool via ``asyncio.to_thread`` so the event loop is never blocked.

    Usage::

        provider = BedrockProvider(
            region="us-east-1",
            default_model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        )
        response = await provider.complete(messages)
        async for chunk in provider.stream(messages):
            print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        region: str = "us-east-1",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        default_model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        # api_key / base_url are unused but kept for interface compatibility
        api_key: str = "",
        base_url: str = "https://bedrock-runtime.amazonaws.com",
    ) -> None:
        if not _SDK_AVAILABLE:
            raise ImportError(
                "The 'boto3' package is required to use BedrockProvider. "
                "Install it with: pip install 'guildmaster-ai[bedrock]'"
            )
        super().__init__(api_key=api_key, base_url=base_url, default_model=default_model)
        boto_kwargs: dict[str, Any] = {"region_name": region}
        if aws_access_key_id:
            boto_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            boto_kwargs["aws_secret_access_key"] = aws_secret_access_key
        self._bedrock = _boto3.client("bedrock-runtime", **boto_kwargs)

    # ── format helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _convert_messages(
        messages: list[LLMMessage],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Split system prompt and convert to Bedrock Converse message format."""
        system_text = ""
        converted: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system_text = msg.content
                continue

            if msg.role == "tool":
                # Tool result maps to toolResult content block inside a user message
                converted.append({
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": msg.tool_call_id or "",
                                "content": [{"text": msg.content}],
                            }
                        }
                    ],
                })
                continue

            content_blocks: list[dict[str, Any]] = []

            if msg.tool_calls:
                if msg.content:
                    content_blocks.append({"text": msg.content})
                for tc in msg.tool_calls:
                    fn = tc.get("function", {})
                    args_raw = fn.get("arguments", "{}")
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    content_blocks.append({
                        "toolUse": {
                            "toolUseId": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": args,
                        }
                    })
            else:
                content_blocks.append({"text": msg.content})

            bedrock_role = "assistant" if msg.role == "assistant" else "user"
            converted.append({"role": bedrock_role, "content": content_blocks})

        return system_text, converted

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Convert OpenAI-compatible tools to Bedrock ``toolConfig`` format."""
        tool_specs = []
        for tool in tools:
            fn = tool.get("function", {})
            tool_specs.append({
                "toolSpec": {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "inputSchema": {
                        "json": fn.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
            })
        return {"tools": tool_specs}

    @staticmethod
    def _parse_converse_response(res: Any, model: str) -> LLMResponse:
        output = res.get("output", {}).get("message", {})
        content_blocks = output.get("content", [])

        text_parts = [b["text"] for b in content_blocks if "text" in b]
        content = "".join(text_parts)

        tool_calls: list[dict[str, Any]] | None = None
        tool_blocks = [b["toolUse"] for b in content_blocks if "toolUse" in b]
        if tool_blocks:
            tool_calls = [
                {
                    "id": tb.get("toolUseId", ""),
                    "type": "function",
                    "function": {
                        "name": tb.get("name", ""),
                        "arguments": json.dumps(tb.get("input", {})),
                    },
                }
                for tb in tool_blocks
            ]

        usage_raw = res.get("usage", {})
        usage = {
            "prompt_tokens": usage_raw.get("inputTokens", 0),
            "completion_tokens": usage_raw.get("outputTokens", 0),
            "total_tokens": usage_raw.get("totalTokens", 0),
        }

        return LLMResponse(
            content=content,
            model=model,
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=res.get("stopReason"),
        )

    def _sync_converse(self, **kwargs: Any) -> Any:
        return self._bedrock.converse(**kwargs)

    def _sync_converse_stream(self, **kwargs: Any) -> Any:
        return self._bedrock.converse_stream(**kwargs)

    # ── public API ────────────────────────────────────────────────────────────

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        resolved_model = model or self.default_model
        system_text, converted = self._convert_messages(messages)

        kwargs: dict[str, Any] = {
            "modelId": resolved_model,
            "messages": converted,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_text:
            kwargs["system"] = [{"text": system_text}]
        if tools:
            kwargs["toolConfig"] = self._convert_tools(tools)

        res = await asyncio.to_thread(self._sync_converse, **kwargs)
        return self._parse_converse_response(res, resolved_model)

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        resolved_model = model or self.default_model
        system_text, converted = self._convert_messages(messages)

        kwargs: dict[str, Any] = {
            "modelId": resolved_model,
            "messages": converted,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_text:
            kwargs["system"] = [{"text": system_text}]
        if tools:
            kwargs["toolConfig"] = self._convert_tools(tools)

        res = await asyncio.to_thread(self._sync_converse_stream, **kwargs)
        event_stream = res.get("stream", [])

        for event in event_stream:
            delta = event.get("contentBlockDelta", {}).get("delta", {})
            text = delta.get("text")
            if text:
                yield text

    async def close(self) -> None:
        # boto3 clients are not async and don't require explicit teardown
        pass

    async def __aenter__(self) -> BedrockProvider:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

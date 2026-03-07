from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from guildmaster_ai.llm.base_provider import BaseLLMProvider, LLMMessage, LLMResponse

try:
    from google import genai as _genai
    from google.genai import types as _gtypes

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SDK_AVAILABLE = False


class GoogleProvider(BaseLLMProvider):
    """Google Gemini LLM provider using the official ``google-genai`` SDK.

    Install with::

        pip install "guildmaster-ai[google]"

    Environment variable: ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``.

    The Google Generative AI API uses a different message and tool format from
    OpenAI.  This provider converts between the two transparently:

    * ``system`` role messages become the ``system_instruction`` field.
    * Tool definitions use ``function_declarations`` instead of ``functions``.
    * Tool call results become ``function_response`` parts.

    Usage::

        provider = GoogleProvider(api_key="AIza...")
        response = await provider.complete(messages)
        async for chunk in provider.stream(messages):
            print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com",
        default_model: str = "gemini-2.0-flash",
    ) -> None:
        if not _SDK_AVAILABLE:
            raise ImportError(
                "The 'google-genai' package is required to use GoogleProvider. "
                "Install it with: pip install 'guildmaster-ai[google]'"
            )
        super().__init__(api_key=api_key, base_url=base_url, default_model=default_model)
        self._client = _genai.Client(api_key=api_key)

    # ── format helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _convert_messages(
        messages: list[LLMMessage],
    ) -> tuple[str, list[Any]]:
        """Split out system instruction and convert messages to Gemini ``Content`` objects."""
        import json

        system_instruction = ""
        contents = []

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
                continue

            if msg.role == "tool":
                # Function response — attaches to the previous assistant turn
                contents.append(
                    _gtypes.Content(
                        role="user",
                        parts=[
                            _gtypes.Part.from_function_response(
                                name=msg.name or "tool",
                                response={"output": msg.content},
                            )
                        ],
                    )
                )
                continue

            gemini_role = "model" if msg.role == "assistant" else "user"
            parts: list[Any] = []

            if msg.tool_calls:
                # Assistant message with tool calls
                if msg.content:
                    parts.append(_gtypes.Part.from_text(msg.content))
                for tc in msg.tool_calls:
                    fn = tc.get("function", {})
                    args_raw = fn.get("arguments", "{}")
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    parts.append(
                        _gtypes.Part.from_function_call(
                            name=fn.get("name", ""),
                            args=args,
                        )
                    )
            else:
                parts.append(_gtypes.Part.from_text(msg.content))

            contents.append(_gtypes.Content(role=gemini_role, parts=parts))

        return system_instruction, contents

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[Any]:
        """Convert OpenAI-compatible tool specs to Gemini ``FunctionDeclaration`` format."""
        declarations = []
        for tool in tools:
            fn = tool.get("function", {})
            declarations.append(
                _gtypes.FunctionDeclaration(
                    name=fn.get("name", ""),
                    description=fn.get("description", ""),
                    parameters=fn.get("parameters"),
                )
            )
        return [_gtypes.Tool(function_declarations=declarations)]

    @staticmethod
    def _parse_response(res: Any, model: str) -> LLMResponse:
        import json

        candidate = res.candidates[0]
        text_parts = [p.text for p in candidate.content.parts if hasattr(p, "text") and p.text]
        content = "".join(text_parts)

        tool_calls: list[dict[str, Any]] | None = None
        fn_parts = [p for p in candidate.content.parts if hasattr(p, "function_call") and p.function_call]
        if fn_parts:
            tool_calls = [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": p.function_call.name,
                        "arguments": json.dumps(dict(p.function_call.args)),
                    },
                }
                for i, p in enumerate(fn_parts)
            ]

        usage: dict[str, int] = {}
        if hasattr(res, "usage_metadata") and res.usage_metadata:
            um = res.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(um, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(um, "total_token_count", 0) or 0,
            }

        finish_reason = str(candidate.finish_reason) if candidate.finish_reason else None

        return LLMResponse(
            content=content,
            model=model,
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

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
        system_instruction, contents = self._convert_messages(messages)

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = self._convert_tools(tools)

        config = _gtypes.GenerateContentConfig(**config_kwargs)
        res = await self._client.aio.models.generate_content(
            model=resolved_model,
            contents=contents,
            config=config,
        )
        return self._parse_response(res, resolved_model)

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
        system_instruction, contents = self._convert_messages(messages)

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = self._convert_tools(tools)

        config = _gtypes.GenerateContentConfig(**config_kwargs)
        async for chunk in await self._client.aio.models.generate_content_stream(
            model=resolved_model,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                yield chunk.text

    async def close(self) -> None:
        # google-genai client does not require explicit teardown
        pass

    async def __aenter__(self) -> GoogleProvider:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

"""
Asynchronous client wrapper for LLM chat completion APIs.

Handles communication with OpenRouter, handles retries with exponential backoff
for rate limit and connection errors, and streams unified response events.
"""

import asyncio
from typing import Any, AsyncGenerator
from openai import APIError, APIConnectionError, RateLimitError, AsyncOpenAI

from client.response import StreamEventType, StreamEvent, TextDelta, TokenUsage

class LLMClient:
    """
    Client for interacting with the OpenAI-compatible language model API (OpenRouter).

    Manages connection lifecycles, configures endpoints/keys, and implements
    resilient chat completion requests with automatic exponential backoff retries.
    """

    def __init__(self) -> None:
        """Initialize the LLM client with default configuration."""
        self._client : AsyncOpenAI | None = None
        self._max_retries: int = 3
    
    def get_client(self) -> AsyncOpenAI:
        """Lazily initialize and return the AsyncOpenAI client instance."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key='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',
                base_url='https://openrouter.ai/api/v1',
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client session and clear the client reference."""
        if self._client:
            await self._client.close()
            self._client = None
    
    async def chat_completion(
        self, 
        messages: list[dict[str, Any]], 
        stream: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute a chat completion request with automatic retry handling."""
        client = self.get_client()

        kwargs = {
            "model": "poolside/laguna-m.1:free",
            "messages": messages,
            "stream": stream,
        }

        # Attempt the API call up to _max_retries + 1 times
        for attempt in range(self._max_retries + 1):
            try:
                if stream:
                    async for event in self._stream_response(client, kwargs):
                        yield event
                else:
                    event = await self._non_stream_response(client, kwargs)
                    yield event
                return
            except RateLimitError as e:
                # Retry with exponential backoff if below the maximum retry limit
                if attempt < self._max_retries:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f"Rate limit exceeded: {e}"
                    )
                    return

            except APIConnectionError as e:
                # Retry with exponential backoff on network connection issues
                if attempt < self._max_retries:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f"Connect Error: {e}"
                    )
                    return

            except APIError as e:
                # Do not retry on general API errors (e.g., authentication, invalid requests)
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error=f"API Error: {e}"
                )
                return

    async def _stream_response(
        self, 
        client: AsyncOpenAI, 
        kwargs: dict[str, Any]
    ) -> AsyncGenerator[StreamEvent, None]:
        """Handle streaming responses from the API client."""
        response  = await client.chat.completions.create(**kwargs)

        finish_reason: str | None = None
        usage : TokenUsage | None = None

        # Iterate over incoming chunks in the response stream
        async for chunk in response:
            # Extract token usage details if present in the chunk
            if hasattr(chunk, "usage") and chunk.usage:
                usage = TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                    cached_tokens=chunk.usage.prompt_tokens_details.cached_tokens,
                )
            
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if choice.finish_reason:
                finish_reason = choice.finish_reason

            if delta.content:
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    text_delta=TextDelta(delta.content)
                )
        
        # Signal that the entire streaming message is complete
        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason=finish_reason,
            usage=usage
        )
    
    async def _non_stream_response(
        self, 
        client: AsyncOpenAI, 
        kwargs: dict[str, Any]
    ) -> StreamEvent:
        """Handle non-streaming responses from the API client."""
        response  = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        
        text_delta = None
        if message.content:
            text_delta = TextDelta(content=message.content)
        
        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=response.usage.prompt_tokens_details.cached_tokens,
            )
        return StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            text_delta=text_delta,
            finish_reason=choice.finish_reason,
            usage=usage,
        )

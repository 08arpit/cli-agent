"""
Definitions of LLM response events, token usage tracking, and streaming types.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

@dataclass
class TextDelta:
    """
    Represents a incremental chunk of generated text from the LLM.
    """
    content: str
    def __str__(self):
        """Return the text content as a string."""
        return self.content

class StreamEventType(str, Enum):
    """
    Enum representing different event types in the response stream.
    """
    TEXT_DELTA = "text_delta"
    MESSAGE_COMPLETE = "message_complete"
    ERROR = "error"

@dataclass
class TokenUsage:
    """
    Tracks token consumption of the API requests and responses.
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Add token usage counts from another TokenUsage instance."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )

@dataclass
class StreamEvent:
    """
    Encapsulates a unified response event returned from the LLM client.

    Attributes:
        type: The stream event type.
        text_delta: The text content chunk, present if type is TEXT_DELTA.
        error: The error message string, present if type is ERROR.
        finish_reason: The reason why streaming finished, present if type is MESSAGE_COMPLETE.
        usage: TokenUsage statistics, populated if type is MESSAGE_COMPLETE.
    """
    type: StreamEventType
    text_delta: TextDelta | None = None
    error: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None

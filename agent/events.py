"""
Definitions for agent-level events.

Categorizes agent lifecycle events (start, end, error) and text streaming events
(deltas, completion) for orchestration/UI consumption.
"""

from __future__ import annotations 
from client.response import TokenUsage
from typing import Any
from enum import Enum
from dataclasses import dataclass, field

class AgentEventType(str, Enum):
    """
    Enum representing different event stages during an agent's run execution.
    """
    # Agent lifecycle
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_ERROR = "agent_error"

    # Text streaming
    TEXT_DELTA = "text_delta"
    TEXT_COMPLETE = "text_complete"

@dataclass
class AgentEvent:
    """
    Represents an event emitted by the agent runner loop.

    Attributes:
        type: The category type of the agent event.
        data: Arbitrary event payload dictionary.
    """
    type: AgentEventType
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def agent_start(cls, message: str) -> AgentEvent:
        """Create an event signaling that agent processing has started."""
        return cls(
            type = AgentEventType.AGENT_START,
            data = {"message": message}
        )
    
    @classmethod
    def agent_end(cls, response: str | None = None, usage: TokenUsage | None = None) -> AgentEvent:
        """Create an event signaling that agent processing has ended."""
        return cls(
            type = AgentEventType.AGENT_END,
            data = {
                "response": response, 
                "usage": usage.__dict__ if usage else None
            }
        )
    
    @classmethod
    def agent_error(cls, error: str, details: dict[str, Any] | None = None) -> AgentEvent:
        """Create an event signaling that the agent encountered an error."""
        return cls(
            type = AgentEventType.AGENT_ERROR,
            data = {"error": error, "details": details or {}}
        )

    @classmethod
    def text_delta(cls, content: str) -> AgentEvent:
        """Create an event containing an incremental response delta chunk."""
        return cls(
            type = AgentEventType.TEXT_DELTA,
            data = {"content": content}
        )
    
    @classmethod
    def text_complete(cls, content: str) -> AgentEvent:
        """Create an event indicating assistant response generation has completed."""
        return cls(
            type = AgentEventType.TEXT_COMPLETE,
            data = {"content": content}
        )

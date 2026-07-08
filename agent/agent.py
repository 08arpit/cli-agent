"""
Core Agent orchestration module.

Initializes the LLM client and context manager, manages prompt dispatching,
and generates async event streams representing the agent lifecycle.
"""

from __future__ import annotations
from context.manager import ContextManager
from agent.events import AgentEventType
from client.response import StreamEventType
from client.llm_client import LLMClient
from typing import AsyncGenerator
from agent.events import AgentEvent

class Agent:
    """
    Orchestrates connection flow, system prompts, message history, and LLM queries.
    Supports asynchronous context management.
    """

    def __init__(self) -> None:
        """Initialize the agent with an LLM client and context manager."""
        self.client = LLMClient()
        self.context_manager = ContextManager()

    async def run(self, message: str):
        """Run the agent for a user message, yielding lifecycle events."""
        yield AgentEvent.agent_start(message)
        self.context_manager.add_user_message(message)

        final_response: str | None = None
        async for event in self._agent_loop():
            yield event

            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")
        
        yield AgentEvent.agent_end(final_response)

    async def _agent_loop(self) -> AsyncGenerator[AgentEvent, None]:
        """Stream LLM completion deltas and yield agent response events."""
        response_text = ""

        # Request chat completion stream from LLMClient
        async for event in self.client.chat_completion(
            self.context_manager.get_messages(), stream=True
        ):
            if event.type == StreamEventType.TEXT_DELTA:
                if event.text_delta:
                    content = event.text_delta.content
                    response_text += content
                    yield AgentEvent.text_delta(content)
                
            elif event.type == StreamEventType.ERROR:
                yield AgentEvent.agent_error(event.error or "Unknown Error")
        
        # Save response context once successfully collected
        self.context_manager.add_assistant_message(
            response_text or None
        )
        if response_text:
            yield AgentEvent.text_complete(response_text)
    
    async def __aenter__(self) -> Agent:
        """Enter the async context and return the agent instance."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context and close the LLM client."""
        if self.client:
            await self.client.close()
            self.client = None

"""
Context management module for handling session history.

Tracks messages exchanged with the LLM and calculates token counts for context window tracking.
"""

from typing import Any
from utils.text import count_tokens
from prompts.system import get_system_prompt
from dataclasses import dataclass

@dataclass
class MessageItem:
    """
    Represents a single chat message in the session history.

    Attributes:
        role: The sender role (e.g., 'user', 'assistant', 'system').
        content: The text content of the message.
        token_count: The calculated token count for the message, if computed.
    """
    role: str
    content: str
    token_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the message to the dictionary format expected by the LLM client."""
        result: dict[str, Any] = {"role": self.role}
        
        if self.content:
            result["content"] = self.content

        return result

class ContextManager:
    """
    Manages the running conversation history and prompt assembly for API calls.
    """

    def __init__(self) -> None:
        """Initialize the context manager with the default system prompt and config."""
        self._system_prompt = get_system_prompt()
        self._model_name = "poolside/laguna-m.1:free"
        self._messages: list[MessageItem] = []

    def add_user_message(self, content: str) -> None:
        """Add a user message to the session context."""
        item = MessageItem(
            role='user',
            content=content,
            token_count=count_tokens(content, self._model_name)
        )
        self._messages.append(item)

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the session context."""
        item = MessageItem(
            role='assistant',
            content=content,
            token_count=count_tokens(content, self._model_name)
        )
        self._messages.append(item)
    
    def get_messages(self) -> list[dict[str, Any]]:
        """Return the full message list including system prompt and history."""
        messages = []
        # Include the system prompt at the beginning if one is configured
        if self._system_prompt:
            messages.append({
                "role": "system",
                "content": self._system_prompt
            })
        for item in self._messages:
            messages.append(item.to_dict())

        return messages

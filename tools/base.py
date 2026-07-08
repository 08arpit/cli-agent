"""
Base classes, enums, and utility data structures for agent tools.

This module defines the abstract interface that all agent tools must inherit
and implement, along with parameter validation helper utilities.
"""

from __future__ import annotations
import abc
from typing import Any
from enum import Enum
from pydantic import BaseModel, ValidationError
from pydantic.json_schema import model_json_schema
from dataclasses import dataclass, field
from pathlib import Path

class ToolKind(str, Enum):
    """
    Enum representing categories/capabilities of different tools.
    """
    READ = "read"
    WRITE = "write"
    SHELL = "shell"
    NETWORK = "network"
    MEMORY = "memory"
    MCP = "mcp"

@dataclass
class ToolInvocation:
    """
    Represents the arguments and context of a tool invocation.

    Attributes:
        params: Key-value parameters passed to the tool.
        cwd: The current working directory in which the tool should execute.
    """
    params: dict[str, Any]
    cwd: Path

@dataclass
class ToolResult:
    """
    Represents the output resulting from executing a tool.

    Attributes:
        success: Whether the tool completed successfully.
        output: String output/result of the tool.
        error: Error message string if success is False, otherwise None.
        metadata: Arbitrary metadata dictionary for telemetry or additional context.
    """
    success: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    truncated: bool = False

    @classmethod
    def error_result(
        cls,
        error: str,
        output: str = ""
    ) -> ToolResult:
        """Create a failed tool result with an error message."""
        return cls(
            success=False,
            output=output,
            error=error
        )

    @classmethod
    def success_result(
        cls,
        output: str,
        **kwargs: Any
    ) -> ToolResult:
        """Create a successful tool result with output."""
        return cls(
            success=True,
            output=output,
            error=None,
            **kwargs
        )

@dataclass
class ToolConfirmation(BaseModel):
    """
    Container details for mutations requiring explicit user approval.
    """
    tool_name: str
    params: dict[str, Any]
    description: str

class Tool(abc.ABC):
    """
    Abstract base class representing an executable tool for the agent.
    """
    name: str = "base_tool"
    description: str = "Base tool"
    kind: ToolKind = ToolKind.READ

    def __init__(self) -> None:
        """Initialize the base tool class."""
        pass
    
    @property
    def schema(self) -> dict[str, Any] | type[BaseModel]:
        """Return the parameter schema this tool accepts."""
        raise NotImplementedError("Tool must define schema property or class attribute")
    
    @abc.abstractmethod
    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        """Execute the tool action asynchronously."""
        pass

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Validate parameters against the tool schema."""
        schema = self.schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                schema(**params)
            except ValidationError as e:
                errors = []
                for error in e.errors():
                    field = ".".join(str(x) for x in error.get('loc', []))
                    msg = error.get('msg', 'Validation error')
                    errors.append(f"Parameter '{field}': {msg}")
                return errors
            except Exception as e:
                return [str(e)]
        
        return []
        
    def is_mutating(self, params: dict[str, Any] = None) -> bool:
        """Return whether this tool performs state mutation."""
        return self.kind in {
            ToolKind.WRITE, 
            ToolKind.SHELL, 
            ToolKind.NETWORK, 
            ToolKind.MEMORY
        }

    async def get_confirmation(self, invocation: ToolInvocation) -> ToolConfirmation | None:
        """Return confirmation details if user approval is required before executing."""
        if not self.is_mutating(invocation.params):
            return None

        return ToolConfirmation(
            tool_name=self.name,
            params=invocation.params,
            description=f"Execute {self.name}",
        )

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert the tool schema to OpenAI function-calling format."""
        schema = self.schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            json_schema = model_json_schema(schema, mode="serialization")

            return {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": json_schema.get("properties", {}),
                    "required": json_schema.get("required", [])
                }
            }

        if isinstance(schema, dict):
            result = {
                "name": self.name,
                "description": self.description
            }
            if 'parameters' in schema:
                result['parameters'] = schema['parameters']
            else:
                result['parameters'] = schema
                
            return result

        raise ValueError(f"Invalid schema type for tool {self.name}: {type(schema)}")
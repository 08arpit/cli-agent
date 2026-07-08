"""
Terminal User Interface (TUI) module.

Manages console output styling using Rich themes, colors, and layout components.
"""

from rich.theme import Theme
from rich.console import Console
from rich.rule import Rule
from rich.text import Text

# Configure standard console colors and styles for agent output and roles
AGENT_THEME = Theme(
    {
        # General
        "info": "cyan",
        "warning": "yellow",
        "error": "bright_red bold",
        "success": "green",
        "dim": "dim",
        "muted": "grey50",
        "border": "grey35",
        "highlight": "bold cyan",
        # Roles
        "user": "bright_blue bold",
        "assistant": "bright_white",
        # Tools
        "tool": "bright_magenta bold",
        "tool.read": "cyan",
        "tool.write": "yellow",
        "tool.shell": "magenta",
        "tool.network": "bright_blue",
        "tool.memory": "green",
        "tool.mcp": "bright_cyan",
        # Code / blocks
        "code": "white",
    }
)

# Singleton console reference to ensure consistent output formatting
_console: Console | None = None

def get_console() -> Console:
    """Return or create the global Rich Console singleton."""
    global _console
    if _console is None:
        _console = Console(theme=AGENT_THEME, highlight=False)

    return _console

class TUI:
    """
    Handles rendering messages and streaming assistant output to the console.
    """

    def __init__(self, console: Console | None = None) -> None:
        """Initialize the TUI with a Console instance."""
        self.console = console or get_console()
        self._assistant_stream_open = False

    def begin_assistant(self) -> None:
        """Render the separator indicating assistant response streaming has started."""
        self.console.print()
        self.console.print(Rule(Text("Agent", style='assistant')))
        self._assistant_stream_open = True
    
    def end_assistant(self) -> None:
        """Close the assistant stream block."""
        if self._assistant_stream_open:
            self.console.print()
        self._assistant_stream_open = False

    def stream_assistant_delta(self, content: str) -> None:
        """Print an incremental assistant response chunk."""
        self.console.print(content, end="", markup=False)

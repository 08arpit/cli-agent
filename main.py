"""
CLI Entrypoint for the AI coding agent.

Parses command-line arguments using click, configures the terminal UI,
and runs the agent asynchronously.
"""

import asyncio
import click
import sys
from agent.events import AgentEventType
from agent.agent import Agent
from ui.tui import  TUI, get_console

# Fetch standard console singleton instance
console = get_console()

class CLI:
    """
    CLI command processor and session coordinator.
    """

    def __init__(self) -> None:
        """Initialize the CLI with an empty agent and TUI coordinator."""
        self.agent : Agent | None = None
        self.tui = TUI(console)

    async def run_single(self, message: str) -> str | None:
        """Run a single prompt-to-response agent lifecycle."""
        async with Agent() as agent:
            self.agent = agent
            return await self._process_message(message)

    async def _process_message(self, message: str) -> str | None:
        """Process agent events and render streaming output to the terminal."""
        if not self.agent:
            return None
        
        assistant_steaming = False
        final_response: str | None = None

        # Listen to and dispatch events yielded from the agent execution
        async for event in self.agent.run(message):
            if event.type == AgentEventType.TEXT_DELTA:
                content = event.data.get("content", "")
                if not assistant_steaming:
                    # Open the streaming block visual indicators
                    self.tui.begin_assistant()
                    assistant_steaming = True
                self.tui.stream_assistant_delta(content)
            elif event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")
                if assistant_steaming:
                    # Close the streaming block visual indicators
                    self.tui.end_assistant()
                    assistant_steaming = False
            elif event.type == AgentEventType.AGENT_ERROR:
                error = event.data.get("error", "Unknown Error")
                console.print(f"\n[error]Error: {error}[/error]")

        return final_response

@click.command()
@click.argument("prompt", required=False)
def main(
    prompt: str | None,
) -> None:
    """CLI entrypoint; run a single prompt if provided."""
    cli = CLI()
    if prompt:
        # Execute the single prompt in the asyncio runtime event loop
        result = asyncio.run(cli.run_single(prompt))
        if result is None:
            sys.exit(1)

if __name__ == "__main__":
    main()

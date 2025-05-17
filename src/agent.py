from contextlib import asynccontextmanager

from mcp_agent.agents.agent import Agent
from mcp_agent.app import MCPApp

from .llm import AnthropicAugmentedLLM
from .prompt import get_system_prompt_with_repo


@asynccontextmanager
async def create_interactive_agent(repo_path, name="interactive_cli_agent"):
    """Yield an initialized MCP agent and LLM tied to a running ``MCPApp``.

    The previous implementation returned the agent and LLM from inside the
    ``MCPApp.run`` context which meant the application shut down immediately
    after the function returned.  Consumers would then interact with an agent
    that no longer had a running application, leading to runtime errors.

    Args:
        repo_path: Path to the local code repository to analyze.
        name: Name of the agent app.
    """

    app = MCPApp(name=name)
    async with app.run() as agent_app:
        # Create the agent with a basic instruction
        agent = Agent(
            name="cli_assistant",
            instruction=get_system_prompt_with_repo(repo_path),
            server_names=["file"],
        )

        # Attach an LLM to the agent
        llm = await agent.attach_llm(AnthropicAugmentedLLM)

        agent_app.logger.info(
            f"Interactive Agent initialized with repo: {repo_path}"
        )

        yield agent, llm

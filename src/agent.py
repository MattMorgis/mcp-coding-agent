from mcp_agent.agents.agent import Agent
from mcp_agent.app import MCPApp

from .llm import AnthropicAugmentedLLM
from .prompt import get_system_prompt_with_repo


async def create_interactive_agent(repo_path, name="interactive_cli_agent"):
    """Create and initialize an MCP agent with Anthropic LLM.

    Args:
        repo_path: Path to the local code repository to analyze
        name: Name of the agent app

    Returns the initialized agent and LLM for direct use.
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

        agent_app.logger.info(f"Interactive Agent initialized with repo: {repo_path}")

        return agent, llm

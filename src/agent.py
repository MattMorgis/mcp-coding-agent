from mcp_agent.agents.agent import Agent
from mcp_agent.app import MCPApp
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM


async def create_interactive_agent(name="interactive_cli_agent"):
    """Create and initialize an MCP agent with Anthropic LLM.

    Returns the initialized agent and LLM for direct use.
    """
    app = MCPApp(name=name)

    async with app.run() as agent_app:
        # Create the agent with a basic instruction
        agent = Agent(
            name="cli_assistant",
            instruction="""You are a helpful assistant.
            Be concise and direct in your responses.
            Maintain context throughout the conversation.
            Answer users' questions to the best of your ability.""",
            server_names=["file"],
        )

        # Attach an LLM to the agent
        llm = await agent.attach_llm(AnthropicAugmentedLLM)

        agent_app.logger.info("Interactive Agent initialized")

        return agent, llm

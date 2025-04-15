import asyncio

from mcp_agent.agents.agent import Agent
from mcp_agent.app import MCPApp
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM

app = MCPApp(name="interactive_cli_agent")


async def main():
    async with app.run() as agent_app:
        logger = agent_app.logger

        # Create the agent with a basic instruction
        cli_agent = Agent(
            name="cli_assistant",
            instruction="""You are a helpful assistant.
            Be concise and direct in your responses.
            Maintain context throughout the conversation.
            Answer users' questions to the best of your ability.""",
            server_names=[],  # No servers needed for this simple example
        )

        logger.info("Interactive CLI Agent started")
        print("Welcome to the Interactive CLI Agent!")
        print("Type 'exit' or 'quit' to end the conversation.\n")

        async with cli_agent:
            # Attach an LLM to the agent
            llm = await cli_agent.attach_llm(AnthropicAugmentedLLM)

            # Interactive loop
            while True:
                # Get user input
                user_input = input("\nYou: ")

                # Check for exit command
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\nGoodbye!")
                    break

                # Generate response
                response = await llm.generate_str(message=user_input)

                # Print the response
                print(f"\nAgent: {response}")


if __name__ == "__main__":
    asyncio.run(main())

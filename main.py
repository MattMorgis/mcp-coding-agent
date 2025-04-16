import argparse
import asyncio
import contextlib
import itertools
import os
import sys

from mcp_agent.workflows.llm.augmented_llm import RequestParams

from src.agent import create_interactive_agent


async def spinner():
    messages = ["Thinking...", "Ruminating...", "Generating..."]
    message_idx = 0
    spin_count = 0
    for c in itertools.cycle(["|", "/", "-", "\\"]):
        msg = messages[message_idx]
        print(f"\r{msg} {c}", end="", flush=True)
        try:
            await asyncio.sleep(0.1)  # Fast spinner speed
            spin_count += 1
            if spin_count >= 20:  # Change message every 20 spins (about 2 seconds)
                message_idx = (message_idx + 1) % len(messages)
                spin_count = 0
        except asyncio.CancelledError:
            print(
                "\r" + " " * (max(len(m) for m in messages) + 2) + "\r",
                end="",
                flush=True,
            )  # Clear line
            break


async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="CLI Coding Agent")
    parser.add_argument("repo_path", help="Path to the local code repository")
    args = parser.parse_args()

    # Validate repository path
    repo_path = os.path.abspath(args.repo_path)
    if not os.path.exists(repo_path):
        print(f"Error: Repository path '{repo_path}' does not exist.")
        sys.exit(1)

    if not os.path.isdir(repo_path):
        print(f"Error: Repository path '{repo_path}' is not a directory.")
        sys.exit(1)

    # Initialize the agent
    agent, llm = await create_interactive_agent(repo_path)

    print("\nWelcome to the CLI Coding Agent!")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    print(f"Repository path: {repo_path}")
    print("What can I help you with today?")

    async with agent:
        # Interactive loop
        while True:
            # Get user input
            user_input = input("\nYou: ")

            # Check for exit command
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nGoodbye!")
                break

            # Generate response
            spinner_task = asyncio.create_task(spinner())
            try:
                response = await llm.generate_str(
                    message=user_input,
                    request_params=RequestParams(
                        max_iterations=25,
                        maxTokens=10000,
                    ),
                )
            finally:
                spinner_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await spinner_task

            # Print the response
            print(f"\nAgent: {response}")


if __name__ == "__main__":
    asyncio.run(main())

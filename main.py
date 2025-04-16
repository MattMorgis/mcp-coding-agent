import argparse
import asyncio
import contextlib
import itertools
import json
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


async def on_message_callback(message_type, message_content):
    """Callback function that prints each message as it's generated"""
    if message_type == "llm_response":
        # Extract text content from the message
        iteration_text = []
        for content in message_content.content:
            if content.type == "text":
                iteration_text.append(content.text)

        if iteration_text:
            # Clear the spinner line and print the iteration
            print("\r" + " " * 50 + "\r", end="", flush=True)
            print("\nIteration output:", flush=True)
            print("".join(iteration_text))
            print("\nThinking...", end="", flush=True)


async def on_tool_call_callback(tool_name, tool_args, tool_use_id):
    """Callback function that prints information about tool usage"""
    # Format tool arguments for better readability
    if isinstance(tool_args, dict):
        formatted_args = json.dumps(tool_args, indent=2)
    else:
        formatted_args = str(tool_args)

    # Clear spinner line and print tool usage info
    print("\r" + " " * 50 + "\r", end="", flush=True)
    print(f"\n🔧 Using tool: {tool_name}", flush=True)
    print(f"Args: {formatted_args}", flush=True)
    print(f"ID: {tool_use_id}", flush=True)
    print("\nThinking...", end="", flush=True)


async def on_tool_result_callback(tool_use_id, result, is_error):
    """Callback function that prints the results from tool calls"""
    # Format the result for better readability
    if isinstance(result, dict):
        try:
            formatted_result = json.dumps(result, indent=2)
        except (TypeError, ValueError):
            formatted_result = str(result)
    else:
        formatted_result = str(result)

    # Determine status icon based on whether there was an error
    status_icon = "❌" if is_error else "✅"
    status_text = "ERROR" if is_error else "SUCCESS"

    # Clear spinner line and print tool result info
    print("\r" + " " * 50 + "\r", end="", flush=True)
    print(f"\n{status_icon} Tool result ({status_text}):", flush=True)
    print(f"Tool ID: {tool_use_id}", flush=True)

    # Limit the length of the output if it's very long
    if len(formatted_result) > 1000:
        print(f"Result: {formatted_result[:1000]}...(truncated)", flush=True)
    else:
        print(f"Result: {formatted_result}", flush=True)

    print("\nThinking...", end="", flush=True)


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
                    on_message=on_message_callback,
                    on_tool_call=on_tool_call_callback,
                    on_tool_result=on_tool_result_callback,
                )
            finally:
                spinner_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await spinner_task

            # Print the response
            print(f"\nAgent: {response}")


if __name__ == "__main__":
    asyncio.run(main())

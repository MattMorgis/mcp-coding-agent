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
    messages = ["Thinking", "Thinking.", "Thinking..", "Thinking..."]
    message_idx = 0
    for _ in itertools.cycle(range(4)):
        msg = messages[message_idx]
        print(f"\r{msg}", end="", flush=True)
        try:
            await asyncio.sleep(0.3)  # Slower, less distracting spinner
            message_idx = (message_idx + 1) % len(messages)
        except asyncio.CancelledError:
            print("\r" + " " * 20 + "\r", end="", flush=True)  # Clear line
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
            print("💭 " + "".join(iteration_text))
            # Only show "Thinking..." for intermediate steps, not final responses
            if message_content.stop_reason not in [
                "end_turn",
                "stop_sequence",
                "max_tokens",
            ]:
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

    # Only print args if they're not too long
    if len(formatted_args) < 200:
        print(f"Args: {formatted_args}", flush=True)
    else:
        print(f"Args: {formatted_args[:197]}...", flush=True)

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

    # Clear spinner line and print tool result info
    print("\r" + " " * 50 + "\r", end="", flush=True)
    print(f"\n{status_icon} Tool result:", flush=True)

    # Limit the length of the output if it's very long
    if len(formatted_result) > 150:
        print(f"{formatted_result[:147]}...(truncated)", flush=True)
    else:
        print(f"{formatted_result}", flush=True)

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

            # Add a separator between conversations for clarity
            print("\n" + "-" * 50)

            # Generate response
            spinner_task = asyncio.create_task(spinner())
            try:
                response = await llm.generate(
                    message=user_input,
                    request_params=RequestParams(
                        max_iterations=25,
                        maxTokens=15000,
                    ),
                    on_message=on_message_callback,
                    on_tool_call=on_tool_call_callback,
                    on_tool_result=on_tool_result_callback,
                )
            finally:
                spinner_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await spinner_task

            # Add a separator after the response
            print("\n" + "-" * 50)

            # Remove the final response print to avoid duplication
            # The content has already been printed by the on_message_callback


if __name__ == "__main__":
    asyncio.run(main())

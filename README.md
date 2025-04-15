# Coding CLI Agent

A simple CLI-based coding agent built with the mcp-agent framework.

## Setup

1. Install dependencies:

   ```
   uv sync
   ```

2. Set your Anthropic API key as an environment variable:
   ```
   cp mcp_agent.secrets-example.yaml mcp_agent.secrets.yaml
   ```

## Usage

Run the agent with a repository URL:

```
uv run main.py /Users/matt/Developer/mcp/prismatic
```

The repository URL provided must match the filesystem server path configured in `mcp_agent.config.yaml`.

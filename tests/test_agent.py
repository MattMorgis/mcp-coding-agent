from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent import InteractiveAgent


@pytest.mark.asyncio
async def test_interactive_agent_initialization(mock_mcp_app, mock_agent):
    """Test that the InteractiveAgent initializes correctly."""
    # Test initialization
    interactive_agent = InteractiveAgent()
    await interactive_agent.initialize()

    # Verify
    mock_mcp_app.assert_called_once_with(name="interactive_cli_agent")
    mock_agent.assert_called_once()
    assert interactive_agent.agent is not None
    assert interactive_agent.logger is not None


@pytest.mark.asyncio
async def test_generate_response():
    """Test that the generate_response method calls the LLM correctly."""
    interactive_agent = InteractiveAgent()
    interactive_agent.agent = MagicMock()
    interactive_agent.llm = AsyncMock()
    interactive_agent.llm.generate_str.return_value = "Test response"

    # Test generate_response
    response = await interactive_agent.generate_response("Test message")

    # Verify
    interactive_agent.llm.generate_str.assert_called_once_with(message="Test message")
    assert response == "Test response"

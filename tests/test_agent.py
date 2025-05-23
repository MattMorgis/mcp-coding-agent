from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent import create_interactive_agent


@pytest.mark.asyncio
async def test_create_interactive_agent(mock_mcp_app, mock_agent):
    """Test that the interactive agent is created correctly."""
    # Mock the LLM attachment
    mock_agent_instance = mock_agent.return_value
    mock_llm = AsyncMock()
    # Use AsyncMock for the attach_llm method
    mock_agent_instance.attach_llm = AsyncMock(return_value=mock_llm)

    # Mock repository path
    test_repo_path = "/test/repo/path"

    # Call the function
    agent, llm = await create_interactive_agent(repo_path=test_repo_path)

    # Verify
    mock_mcp_app.assert_called_once_with(name="interactive_cli_agent")
    mock_agent.assert_called_once()
    # Check that repo_path was included in the instruction
    _, kwargs = mock_agent.call_args
    assert test_repo_path in kwargs.get("instruction", "")
    assert agent is mock_agent_instance
    assert llm is mock_llm


@pytest.mark.asyncio
async def test_llm_generate_response():
    """Test that the LLM generate_str method works correctly."""
    # Mock agent and LLM
    agent = MagicMock()
    llm = AsyncMock()
    llm.generate_str.return_value = "Test response"

    # Patch create_interactive_agent to return our mocks
    with patch("src.agent.create_interactive_agent", return_value=(agent, llm)):
        # Test generate_response through the LLM
        response = await llm.generate_str(message="Test message")

        # Verify
        llm.generate_str.assert_called_once_with(message="Test message")
        assert response == "Test response"

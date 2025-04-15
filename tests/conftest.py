from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_mcp_app():
    """Mock the MCPApp class."""
    with patch("src.agent.MCPApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_agent_app = MagicMock()
        mock_app_instance.run.return_value.__aenter__.return_value = mock_agent_app
        mock_app.return_value = mock_app_instance
        yield mock_app


@pytest.fixture
def mock_agent():
    """Mock the Agent class."""
    with patch("src.agent.Agent") as mock_agent_class:
        agent_instance = MagicMock()
        mock_agent_class.return_value = agent_instance
        yield mock_agent_class

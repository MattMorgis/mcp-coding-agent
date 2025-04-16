"""Unit tests for the prompt module."""
import pytest
from src.prompt import get_system_prompt_with_repo


class TestPrompt:
    """Test cases for the prompt module."""

    def test_get_system_prompt_with_repo(self):
        """Test the get_system_prompt_with_repo function."""
        # Setup
        repo_path = "/path/to/test/repo"
        
        # Execute
        result = get_system_prompt_with_repo(repo_path)
        
        # Assert
        assert isinstance(result, str)
        assert len(result) > 0
        assert repo_path in result
        assert "You are a specialized AI coding assistant" in result
        assert "CAPABILITIES:" in result
        assert "INTERACTION STYLE:" in result
        assert "TERMINAL FORMATTING:" in result
        assert "LIMITATIONS:" in result
        
    def test_get_system_prompt_with_empty_repo_path(self):
        """Test with an empty repository path."""
        # Setup
        repo_path = ""
        
        # Execute
        result = get_system_prompt_with_repo(repo_path)
        
        # Assert
        assert isinstance(result, str)
        assert "The codebase is located at: " in result
        
    def test_get_system_prompt_with_special_characters(self):
        """Test with repository path containing special characters."""
        # Setup
        repo_path = "/path/with/special/chars/!@#$%^&*()"
        
        # Execute
        result = get_system_prompt_with_repo(repo_path)
        
        # Assert
        assert isinstance(result, str)
        assert repo_path in result
        
    def test_prompt_structure(self):
        """Test the structure of the generated prompt."""
        # Setup
        repo_path = "/test/repo/path"
        
        # Execute
        result = get_system_prompt_with_repo(repo_path)
        
        # Assert
        # Check for expected sections in the correct order
        sections = [
            "You are a specialized AI coding assistant",
            "The codebase is located at:",
            "CAPABILITIES:",
            "INTERACTION STYLE:",
            "TERMINAL FORMATTING:",
            "LIMITATIONS:",
        ]
        
        # Verify all sections are present and in the correct order
        last_pos = -1
        for section in sections:
            current_pos = result.find(section)
            assert current_pos > last_pos, f"Section '{section}' is not in expected order"
            last_pos = current_pos
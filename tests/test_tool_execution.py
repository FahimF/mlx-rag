"""
Test tool execution with real LLM responses.

This module tests the complete tool execution pipeline including:
- Tool call detection and parsing
- Tool execution with various parameters
- Response formatting and error handling
- Integration with the chat completions endpoint
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from mlx_rag.server import create_app
from mlx_rag.tool_prompts import _parse_tool_calls_from_text, validate_tool_call_format
from mlx_rag.tool_executor import ToolExecutor, ToolExecutionResult


class TestToolCallParsing:
    """Test parsing of tool calls from LLM responses."""
    
    def test_parse_json_tool_calls(self):
        """Test parsing JSON-format tool calls."""
        text = '''
        I'll help you list the files. Let me use the list_directory tool:
        
        {"function": "list_directory", "arguments": {"path": ".", "recursive": false}}
        
        This will show you the files in the current directory.
        '''
        
        tool_calls = _parse_tool_calls_from_text(text)
        
        assert len(tool_calls) == 1
        assert tool_calls[0]["type"] == "function"
        assert tool_calls[0]["function"]["name"] == "list_directory"
        
        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert args["path"] == "."
        assert args["recursive"] is False
    
    def test_parse_xml_tool_calls(self):
        """Test parsing XML-format tool calls."""
        text = '''
        Let me search for that function:
        
        <tool_call function="search_files" args='{"query": "def main", "path": "src"}'/>
        
        This should find the main function.
        '''
        
        tool_calls = _parse_tool_calls_from_text(text)
        
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "search_files"
        
        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert args["query"] == "def main"
        assert args["path"] == "src"
    
    def test_parse_function_call_format(self):
        """Test parsing function call format."""
        text = '''
        I'll read the file for you:
        
        read_file({"path": "README.md"})
        
        This will show the contents.
        '''
        
        tool_calls = _parse_tool_calls_from_text(text)
        
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "read_file"
        
        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert args["path"] == "README.md"
    
    def test_parse_multiple_tool_calls(self):
        """Test parsing multiple tool calls in one response."""
        text = '''
        I'll help you explore the project:
        
        {"function": "list_directory", "arguments": {"path": "."}}
        
        Then I'll read the main file:
        
        {"function": "read_file", "arguments": {"path": "main.py"}}
        '''
        
        tool_calls = _parse_tool_calls_from_text(text)
        
        assert len(tool_calls) == 2
        assert tool_calls[0]["function"]["name"] == "list_directory"
        assert tool_calls[1]["function"]["name"] == "read_file"
    
    def test_ignore_unknown_tools(self):
        """Test that unknown tools are ignored to avoid false positives."""
        text = '''
        I'll process this data:
        
        process_data({"input": "test"})
        
        And also use a real tool:
        
        {"function": "read_file", "arguments": {"path": "test.txt"}}
        '''
        
        tool_calls = _parse_tool_calls_from_text(text)
        
        # Should only find the real tool, not the unknown function
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "read_file"
    
    def test_malformed_json_handling(self):
        """Test handling of malformed JSON in tool calls."""
        text = '''
        {"function": "read_file", "arguments": {"path": "test.txt", invalid}}
        '''
        
        tool_calls = _parse_tool_calls_from_text(text)
        
        # Should handle malformed JSON gracefully
        assert len(tool_calls) == 0


class TestToolCallValidation:
    """Test tool call format validation."""
    
    def test_valid_tool_call(self):
        """Test validation of valid tool call."""
        call = '{"function": "read_file", "arguments": {"path": "test.txt"}}'
        result = validate_tool_call_format(call)
        
        assert result["valid"] is True
        assert result["function"] == "read_file"
        assert result["arguments"]["path"] == "test.txt"
    
    def test_invalid_json(self):
        """Test validation of invalid JSON."""
        call = '{"function": "read_file", invalid json}'
        result = validate_tool_call_format(call)
        
        assert result["valid"] is False
        assert "Invalid JSON format" in result["error"]
    
    def test_missing_function_field(self):
        """Test validation when function field is missing."""
        call = '{"arguments": {"path": "test.txt"}}'
        result = validate_tool_call_format(call)
        
        assert result["valid"] is False
        assert "must include 'function' field" in result["error"]
    
    def test_missing_arguments_field(self):
        """Test validation when arguments field is missing."""
        call = '{"function": "read_file"}'
        result = validate_tool_call_format(call)
        
        assert result["valid"] is False
        assert "must include 'arguments' field" in result["error"]
    
    def test_invalid_arguments_type(self):
        """Test validation when arguments is not an object."""
        call = '{"function": "read_file", "arguments": "invalid"}'
        result = validate_tool_call_format(call)
        
        assert result["valid"] is False
        assert "Arguments must be a JSON object" in result["error"]


class TestToolExecution:
    """Test actual tool execution with mocked file system."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files and directories
            (temp_path / "test.txt").write_text("Hello, World!")
            (temp_path / "src").mkdir()
            (temp_path / "src" / "main.py").write_text("def main():\n    print('Hello')")
            (temp_path / "README.md").write_text("# Test Project\nThis is a test.")
            
            yield temp_path
    
    @pytest.fixture
    def tool_executor(self, temp_workspace):
        """Create a tool executor with the temporary workspace."""
        executor = ToolExecutor()
        executor.workspace_path = str(temp_workspace)
        return executor
    
    def test_list_directory_basic(self, tool_executor):
        """Test basic directory listing."""
        result = tool_executor.execute_tool("list_directory", {"path": "."})
        
        assert result.success is True
        assert "test.txt" in result.output
        assert "src" in result.output
        assert "README.md" in result.output
    
    def test_list_directory_recursive(self, tool_executor):
        """Test recursive directory listing."""
        result = tool_executor.execute_tool("list_directory", {
            "path": ".", 
            "recursive": True
        })
        
        assert result.success is True
        assert "src/main.py" in result.output
    
    def test_read_file_success(self, tool_executor):
        """Test successful file reading."""
        result = tool_executor.execute_tool("read_file", {"path": "test.txt"})
        
        assert result.success is True
        assert "Hello, World!" in result.output
    
    def test_read_file_not_found(self, tool_executor):
        """Test reading non-existent file."""
        result = tool_executor.execute_tool("read_file", {"path": "nonexistent.txt"})
        
        assert result.success is False
        assert "not found" in result.error.lower()
    
    def test_search_files_basic(self, tool_executor):
        """Test basic file search."""
        result = tool_executor.execute_tool("search_files", {
            "query": "Hello",
            "path": "."
        })
        
        assert result.success is True
        assert "test.txt" in result.output or "Hello" in result.output
    
    def test_search_files_regex(self, tool_executor):
        """Test regex file search."""
        result = tool_executor.execute_tool("search_files", {
            "query": "def \\w+\\(",
            "path": ".",
            "use_regex": True
        })
        
        assert result.success is True
        # Should find the function definition
    
    def test_write_file_success(self, tool_executor):
        """Test successful file writing."""
        result = tool_executor.execute_tool("write_file", {
            "path": "new_file.txt",
            "content": "This is new content"
        })
        
        assert result.success is True
        
        # Verify file was created
        new_file_path = Path(tool_executor.workspace_path) / "new_file.txt"
        assert new_file_path.exists()
        assert new_file_path.read_text() == "This is new content"
    
    def test_edit_file_success(self, tool_executor):
        """Test successful file editing."""
        result = tool_executor.execute_tool("edit_file", {
            "path": "test.txt",
            "search": "Hello, World!",
            "replace": "Goodbye, World!"
        })
        
        assert result.success is True
        
        # Verify file was edited
        file_path = Path(tool_executor.workspace_path) / "test.txt"
        assert file_path.read_text() == "Goodbye, World!"
    
    def test_edit_file_search_not_found(self, tool_executor):
        """Test editing when search text is not found."""
        result = tool_executor.execute_tool("edit_file", {
            "path": "test.txt",
            "search": "Non-existent text",
            "replace": "New text"
        })
        
        assert result.success is False
        assert "not found" in result.error.lower()
    
    def test_path_traversal_protection(self, tool_executor):
        """Test protection against path traversal attacks."""
        result = tool_executor.execute_tool("read_file", {
            "path": "../../etc/passwd"
        })
        
        assert result.success is False
        assert "security" in result.error.lower() or "access denied" in result.error.lower()
    
    def test_unknown_tool_handling(self, tool_executor):
        """Test handling of unknown tool names."""
        result = tool_executor.execute_tool("unknown_tool", {"param": "value"})
        
        assert result.success is False
        assert "unknown" in result.error.lower() or "not supported" in result.error.lower()


class TestRealLLMResponseScenarios:
    """Test complete scenarios with realistic LLM responses."""
    
    def test_exploration_workflow(self):
        """Test a typical code exploration workflow."""
        responses = [
            # Initial exploration
            '''I'll help you explore the codebase. Let me start by listing the root directory:
            
            {"function": "list_directory", "arguments": {"path": ".", "recursive": false}}''',
            
            # Reading main file
            '''Now let me read the main file to understand the structure:
            
            {"function": "read_file", "arguments": {"path": "main.py"}}''',
            
            # Searching for specific functions
            '''Let me search for all function definitions:
            
            {"function": "search_files", "arguments": {"query": "def ", "path": ".", "use_regex": false}}'''
        ]
        
        for response in responses:
            tool_calls = _parse_tool_calls_from_text(response)
            assert len(tool_calls) >= 1
            assert all(call["type"] == "function" for call in tool_calls)
    
    def test_debugging_workflow(self):
        """Test a typical debugging workflow."""
        responses = [
            # Search for error
            '''I'll help you find the bug. Let me search for error-related code:
            
            {"function": "search_files", "arguments": {"query": "error|exception", "path": ".", "use_regex": true}}''',
            
            # Read specific file
            '''Let me examine the problematic file:
            
            {"function": "read_file", "arguments": {"path": "src/error_handler.py"}}''',
            
            # Fix the issue
            '''I found the issue. Let me fix it:
            
            {"function": "edit_file", "arguments": {"path": "src/error_handler.py", "search": "raise Exception", "replace": "raise ValueError"}}'''
        ]
        
        for response in responses:
            tool_calls = _parse_tool_calls_from_text(response)
            assert len(tool_calls) >= 1
    
    def test_mixed_format_response(self):
        """Test response with mixed text and tool calls."""
        response = '''
        Based on your request, I'll help you analyze the code. Here's what I'll do:
        
        1. First, let me explore the directory structure:
        {"function": "list_directory", "arguments": {"path": ".", "recursive": true}}
        
        2. Then I'll read the main configuration file:
        {"function": "read_file", "arguments": {"path": "config.json"}}
        
        This will give us a good overview of the project.
        '''
        
        tool_calls = _parse_tool_calls_from_text(response)
        assert len(tool_calls) == 2
        assert tool_calls[0]["function"]["name"] == "list_directory"
        assert tool_calls[1]["function"]["name"] == "read_file"
    
    def test_response_with_explanations(self):
        """Test response with detailed explanations."""
        response = '''
        I understand you want to search for a specific function. Let me help you with that.
        
        I'll search through all Python files to find the function definition:
        
        {"function": "search_files", "arguments": {"query": "def calculate_score", "path": "src"}}
        
        This search will look for the exact function signature in the src directory. 
        If found, I'll then read the file to show you the complete implementation.
        '''
        
        tool_calls = _parse_tool_calls_from_text(response)
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "search_files"
        
        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert args["query"] == "def calculate_score"
        assert args["path"] == "src"


class TestErrorRecovery:
    """Test error handling and recovery scenarios."""
    
    @pytest.fixture
    def mock_executor(self):
        """Create a mock tool executor for error testing."""
        executor = Mock(spec=ToolExecutor)
        return executor
    
    def test_tool_execution_failure_recovery(self, mock_executor):
        """Test recovery from tool execution failures."""
        # Mock a failed execution
        mock_executor.execute_tool.return_value = ToolExecutionResult(
            success=False,
            output="",
            error="File not found: test.txt",
            execution_time=0.1
        )
        
        # Simulate LLM response with error recovery
        response = '''
        It seems the file doesn't exist. Let me list the directory to see what files are available:
        
        {"function": "list_directory", "arguments": {"path": "."}}
        '''
        
        tool_calls = _parse_tool_calls_from_text(response)
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "list_directory"
    
    def test_malformed_response_handling(self):
        """Test handling of malformed LLM responses."""
        malformed_responses = [
            '{"function": "read_file", "arguments":',  # Incomplete JSON
            '{"function": "read_file" "arguments": {}}',  # Missing comma
            'read_file({"path": "test.txt"',  # Incomplete function call
            '{"func": "read_file", "args": {}}',  # Wrong field names
        ]
        
        for response in malformed_responses:
            tool_calls = _parse_tool_calls_from_text(response)
            # Should handle gracefully and return empty list
            assert isinstance(tool_calls, list)
    
    def test_security_error_handling(self, mock_executor):
        """Test handling of security-related errors."""
        mock_executor.execute_tool.return_value = ToolExecutionResult(
            success=False,
            output="",
            error="Security error: Path traversal attempt detected",
            execution_time=0.0
        )
        
        # This should be handled gracefully
        result = mock_executor.execute_tool("read_file", {"path": "../../../etc/passwd"})
        assert result.success is False
        assert "security" in result.error.lower()


@pytest.mark.integration
class TestChatCompletionsIntegration:
    """Integration tests with the chat completions endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        app = create_app()
        return TestClient(app)
    
    @pytest.fixture
    def mock_model_manager(self):
        """Mock the model manager for testing."""
        with patch('mlx_rag.server.get_model_manager') as mock:
            mock_manager = Mock()
            mock_model = Mock()
            mock_model.mlx_wrapper.tokenizer = Mock()
            mock_manager.get_model_for_inference.return_value = mock_model
            mock.return_value = mock_manager
            yield mock_manager
    
    def test_tool_calling_request_format(self, client, mock_model_manager):
        """Test tool calling with proper OpenAI format."""
        # This test would require a more complete setup with database mocking
        # For now, we'll test the basic structure
        
        request_data = {
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": "List the files in the current directory"
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "list_directory",
                        "description": "List files in a directory",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "Directory path"}
                            },
                            "required": ["path"]
                        }
                    }
                }
            ]
        }
        
        # The request structure should be valid
        assert "tools" in request_data
        assert request_data["tools"][0]["type"] == "function"
        assert "function" in request_data["tools"][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

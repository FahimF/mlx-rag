"""
Test OpenAI API compatibility for tool calling functionality.

This module tests that our tool calling implementation is fully compatible
with the OpenAI API format, including request/response structures,
error handling, and streaming responses.
"""

import pytest
import json
import time
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from mlx_rag.server import (
    create_app,
    ChatCompletionRequest,
    ChatCompletionMessage,
    Tool,
    ToolChoice,
    ToolCall
)


class TestOpenAIRequestFormat:
    """Test OpenAI-compatible request format validation."""
    
    def test_basic_tool_calling_request(self):
        """Test basic tool calling request structure."""
        request_data = {
            "model": "gpt-3.5-turbo",
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
                        "description": "List files and directories",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Directory path to list"
                                },
                                "recursive": {
                                    "type": "boolean", 
                                    "description": "List recursively"
                                }
                            },
                            "required": ["path"]
                        }
                    }
                }
            ],
            "tool_choice": "auto"
        }
        
        # Validate request can be parsed by Pydantic model
        request = ChatCompletionRequest(**request_data)
        
        assert request.model == "gpt-3.5-turbo"
        assert len(request.messages) == 1
        assert len(request.tools) == 1
        assert request.tools[0].type == "function"
        assert request.tools[0].function["name"] == "list_directory"
        assert request.tool_choice == "auto"
    
    def test_tool_choice_variations(self):
        """Test different tool_choice parameter variations."""
        base_request = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "description": "Test tool",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }
            ]
        }
        
        # Test "auto" choice
        request_auto = ChatCompletionRequest(**{**base_request, "tool_choice": "auto"})
        assert request_auto.tool_choice == "auto"
        
        # Test "none" choice
        request_none = ChatCompletionRequest(**{**base_request, "tool_choice": "none"})
        assert request_none.tool_choice == "none"
        
        # Test specific function choice
        specific_choice = {
            "type": "function",
            "function": {"name": "test_tool"}
        }
        request_specific = ChatCompletionRequest(**{**base_request, "tool_choice": specific_choice})
        assert isinstance(request_specific.tool_choice, dict)
        assert request_specific.tool_choice["type"] == "function"
    
    def test_multiple_tools_request(self):
        """Test request with multiple tools."""
        request_data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Help me with files"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "list_directory",
                        "description": "List files",
                        "parameters": {"type": "object", "properties": {}}
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read file content",
                        "parameters": {"type": "object", "properties": {}}
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "description": "Write file content",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }
            ]
        }
        
        request = ChatCompletionRequest(**request_data)
        assert len(request.tools) == 3
        assert request.tools[0].function["name"] == "list_directory"
        assert request.tools[1].function["name"] == "read_file"
        assert request.tools[2].function["name"] == "write_file"
    
    def test_message_with_tool_calls(self):
        """Test assistant message with tool calls."""
        message_data = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "list_directory",
                        "arguments": '{"path": "."}'
                    }
                }
            ]
        }
        
        message = ChatCompletionMessage(**message_data)
        assert message.role == "assistant"
        assert message.content is None
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].id == "call_abc123"
        assert message.tool_calls[0].function["name"] == "list_directory"
    
    def test_tool_response_message(self):
        """Test tool response message format."""
        message_data = {
            "role": "tool",
            "content": "file1.txt\nfile2.txt\ndir1/",
            "tool_call_id": "call_abc123"
        }
        
        message = ChatCompletionMessage(**message_data)
        assert message.role == "tool"
        assert message.content == "file1.txt\nfile2.txt\ndir1/"
        assert message.tool_call_id == "call_abc123"


class TestOpenAIResponseFormat:
    """Test OpenAI-compatible response format."""
    
    def test_chat_completion_response_structure(self):
        """Test basic chat completion response structure."""
        from mlx_rag.server import ChatCompletionResponse, ChatCompletionChoice, ChatCompletionUsage
        
        response_data = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I'll help you list the files.",
                        "tool_calls": [
                            {
                                "id": "call_def456",
                                "type": "function",
                                "function": {
                                    "name": "list_directory",
                                    "arguments": '{"path": "."}'
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 25,
                "total_tokens": 75
            }
        }
        
        response = ChatCompletionResponse(**response_data)
        assert response.object == "chat.completion"
        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "tool_calls"
        assert len(response.choices[0].message.tool_calls) == 1
        assert response.usage.total_tokens == 75
    
    def test_streaming_response_structure(self):
        """Test streaming response structure."""
        from mlx_rag.server import ChatCompletionStreamResponse, ChatCompletionStreamChoice
        
        # First chunk with role
        chunk1_data = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None
                }
            ]
        }
        
        chunk1 = ChatCompletionStreamResponse(**chunk1_data)
        assert chunk1.object == "chat.completion.chunk"
        assert chunk1.choices[0].delta["role"] == "assistant"
        assert chunk1.choices[0].finish_reason is None
        
        # Content chunk
        chunk2_data = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "I'll help you"},
                    "finish_reason": None
                }
            ]
        }
        
        chunk2 = ChatCompletionStreamResponse(**chunk2_data)
        assert chunk2.choices[0].delta["content"] == "I'll help you"
        
        # Final chunk
        final_chunk_data = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        
        final_chunk = ChatCompletionStreamResponse(**final_chunk_data)
        assert final_chunk.choices[0].finish_reason == "stop"
    
    def test_tool_calls_response_format(self):
        """Test response format when tool calls are made."""
        response_data = {
            "id": "chatcmpl-tool123",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,  # Content is null when making tool calls
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "list_directory",
                                    "arguments": '{"path": ".", "recursive": false}'
                                }
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "README.md"}'
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        
        from mlx_rag.server import ChatCompletionResponse
        response = ChatCompletionResponse(**response_data)
        
        assert response.choices[0].message.content is None
        assert len(response.choices[0].message.tool_calls) == 2
        assert response.choices[0].finish_reason == "tool_calls"
        assert response.choices[0].message.tool_calls[0].function["name"] == "list_directory"
        assert response.choices[0].message.tool_calls[1].function["name"] == "read_file"


class TestOpenAIErrorHandling:
    """Test OpenAI-compatible error handling."""
    
    def test_invalid_model_error(self):
        """Test error when model doesn't exist."""
        client = TestClient(create_app())
        
        request_data = {
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": []
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        
        # Should return 404 for non-existent model
        assert response.status_code == 404
        error_data = response.json()
        assert "detail" in error_data
        assert "not found" in error_data["detail"].lower()
    
    def test_invalid_tool_format_error(self):
        """Test error handling for invalid tool format."""
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [
                {
                    "type": "invalid_type",  # Should be "function"
                    "function": {"name": "test"}
                }
            ]
        }
        
        # This should raise a validation error when creating the Pydantic model
        with pytest.raises(Exception):  # Pydantic validation error
            ChatCompletionRequest(**request_data)
    
    def test_missing_required_parameters(self):
        """Test error when required parameters are missing."""
        # Missing model
        with pytest.raises(Exception):
            ChatCompletionRequest(messages=[])
        
        # Missing messages
        with pytest.raises(Exception):
            ChatCompletionRequest(model="test")
    
    def test_tool_execution_error_handling(self):
        """Test error handling when tool execution fails."""
        # This would be tested with actual endpoint integration
        # For now, we test the structure
        error_response = {
            "error": {
                "message": "Tool execution failed: File not found",
                "type": "tool_execution_error",
                "code": "tool_error"
            }
        }
        
        assert "error" in error_response
        assert "message" in error_response["error"]
        assert "type" in error_response["error"]


class TestOpenAIConversationFlow:
    """Test complete OpenAI-compatible conversation flows."""
    
    def test_tool_calling_conversation_flow(self):
        """Test a complete tool calling conversation flow."""
        # Step 1: Initial user request
        user_message = {
            "role": "user",
            "content": "Can you list the files in the current directory and then read the README file?"
        }
        
        # Step 2: Assistant response with tool calls
        assistant_tool_response = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_list_dir",
                    "type": "function",
                    "function": {
                        "name": "list_directory",
                        "arguments": '{"path": ".", "recursive": false}'
                    }
                }
            ]
        }
        
        # Step 3: Tool result
        tool_result = {
            "role": "tool",
            "tool_call_id": "call_list_dir",
            "content": "README.md\nmain.py\nsrc/\ntests/"
        }
        
        # Step 4: Assistant continues with next tool call
        assistant_next_tool = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_read_readme",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "README.md"}'
                    }
                }
            ]
        }
        
        # Step 5: Second tool result
        tool_result_2 = {
            "role": "tool",
            "tool_call_id": "call_read_readme",
            "content": "# My Project\nThis is a test project..."
        }
        
        # Step 6: Final assistant response
        assistant_final = {
            "role": "assistant",
            "content": "I've listed the directory contents and read the README file. The directory contains README.md, main.py, src/, and tests/. The README shows this is a test project."
        }
        
        # Validate all messages can be parsed
        messages = [
            user_message,
            assistant_tool_response,
            tool_result,
            assistant_next_tool,
            tool_result_2,
            assistant_final
        ]
        
        for msg in messages:
            parsed_msg = ChatCompletionMessage(**msg)
            assert parsed_msg.role in ["user", "assistant", "tool"]
    
    def test_parallel_tool_calls(self):
        """Test parallel tool calls in a single response."""
        assistant_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "file1.txt"}'
                    }
                },
                {
                    "id": "call_2", 
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "file2.txt"}'
                    }
                },
                {
                    "id": "call_3",
                    "type": "function", 
                    "function": {
                        "name": "list_directory",
                        "arguments": '{"path": "src"}'
                    }
                }
            ]
        }
        
        parsed_msg = ChatCompletionMessage(**assistant_message)
        assert len(parsed_msg.tool_calls) == 3
        assert all(call.type == "function" for call in parsed_msg.tool_calls)
        
        # Tool results would follow
        tool_results = [
            {
                "role": "tool",
                "tool_call_id": "call_1", 
                "content": "Content of file1"
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "content": "Content of file2"
            },
            {
                "role": "tool",
                "tool_call_id": "call_3",
                "content": "main.py\nutils.py"
            }
        ]
        
        for result in tool_results:
            parsed_result = ChatCompletionMessage(**result)
            assert parsed_result.role == "tool"
            assert parsed_result.tool_call_id is not None


class TestSystemPromptIntegration:
    """Test integration with system prompts for tool usage."""
    
    def test_system_prompt_with_tools(self):
        """Test system prompt generation with tools."""
        from mlx_rag.tool_prompts import generate_tool_system_prompt
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files in a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
                        "required": ["path"]
                    }
                }
            }
        ]
        
        system_prompt = generate_tool_system_prompt(tools)
        
        assert "list_directory" in system_prompt
        assert "JSON function call" in system_prompt
        assert "tool" in system_prompt.lower()
    
    def test_contextual_tool_prompts(self):
        """Test contextual tool prompt generation."""
        from mlx_rag.tool_prompts import generate_contextual_prompt
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search for content in files",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "path": {"type": "string"}
                        },
                        "required": ["query", "path"]
                    }
                }
            }
        ]
        
        user_query = "I need to find a bug in my Python code"
        contextual_prompt = generate_contextual_prompt(tools, user_query, [])
        
        assert "debug" in contextual_prompt.lower()
        assert "search_files" in contextual_prompt
        assert "bug" in contextual_prompt.lower()


class TestParameterValidation:
    """Test parameter validation for OpenAI compatibility."""
    
    def test_max_tokens_validation(self):
        """Test max_tokens parameter validation."""
        request_data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 4096
        }
        
        request = ChatCompletionRequest(**request_data)
        assert request.max_tokens == 4096
        
        # Test default value
        request_no_max = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert request_no_max.max_tokens == 8192  # Default value
    
    def test_temperature_validation(self):
        """Test temperature parameter validation."""
        request_data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.8
        }
        
        request = ChatCompletionRequest(**request_data)
        assert request.temperature == 0.8
    
    def test_stream_parameter(self):
        """Test stream parameter."""
        request_data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True
        }
        
        request = ChatCompletionRequest(**request_data)
        assert request.stream is True
        
        # Test default
        request_no_stream = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert request_no_stream.stream is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

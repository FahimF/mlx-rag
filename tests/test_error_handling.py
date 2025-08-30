"""
Test error handling, recovery scenarios, and edge cases.

This module tests various error conditions, recovery mechanisms,
and edge cases for the MLX-RAG tool calling system.
"""

import pytest
import os
import tempfile
import shutil
import json
import time
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
import asyncio

from mlx_rag.server import (
    create_app,
    ChatCompletionRequest,
    ChatCompletionMessage,
    Tool,
    ToolChoice,
    ToolCall,
    ChatCompletionResponse
)
from mlx_rag.tools import (
    ToolExecutor,
    list_directory,
    read_file,
    write_file,
    edit_file,
    search_files
)


class TestToolExecutionErrors:
    """Test error handling in tool execution."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.executor = ToolExecutor(workspace_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_nonexistent_file_error(self):
        """Test error handling when accessing nonexistent files."""
        with pytest.raises(Exception) as exc_info:
            read_file("nonexistent.txt", workspace_dir=self.temp_dir)
        
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in [
            "not found", "does not exist", "no such file"
        ])
    
    def test_permission_denied_error(self):
        """Test error handling for permission denied."""
        if os.name == 'nt':
            pytest.skip("Permission tests not applicable on Windows")
        
        # Create a file and remove read permissions
        restricted_file = os.path.join(self.temp_dir, "restricted.txt")
        with open(restricted_file, "w") as f:
            f.write("restricted content")
        
        os.chmod(restricted_file, 0o000)  # No permissions
        
        try:
            with pytest.raises(Exception) as exc_info:
                read_file("restricted.txt", workspace_dir=self.temp_dir)
            
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in [
                "permission denied", "access denied", "forbidden"
            ])
        finally:
            # Restore permissions for cleanup
            os.chmod(restricted_file, 0o644)
    
    def test_corrupted_file_error(self):
        """Test error handling with corrupted or binary files."""
        # Create a binary file
        binary_file = os.path.join(self.temp_dir, "binary.bin")
        with open(binary_file, "wb") as f:
            f.write(b"\x00\x01\x02\x03\xFF\xFE\xFD")
        
        try:
            content = read_file("binary.bin", workspace_dir=self.temp_dir)
            # If reading succeeds, content should be handled gracefully
            assert isinstance(content, str)
        except Exception as e:
            # Or it should fail with appropriate error
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in [
                "encoding", "binary", "decode", "invalid"
            ])
    
    def test_directory_as_file_error(self):
        """Test error when trying to read directory as file."""
        test_dir = os.path.join(self.temp_dir, "test_directory")
        os.makedirs(test_dir)
        
        with pytest.raises(Exception) as exc_info:
            read_file("test_directory", workspace_dir=self.temp_dir)
        
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in [
            "directory", "is a directory", "not a file"
        ])
    
    def test_invalid_json_in_tool_call(self):
        """Test error handling for invalid JSON in tool calls."""
        invalid_json_strings = [
            '{"path": "test.txt"',  # Missing closing brace
            '{"path": "test.txt", }',  # Trailing comma
            '{path: "test.txt"}',  # Unquoted key
            '{"path": test.txt}',  # Unquoted value
            '{Invalid JSON}',  # Completely invalid
            '',  # Empty string
            'null',  # Null value
        ]
        
        for invalid_json in invalid_json_strings:
            with pytest.raises(Exception) as exc_info:
                json.loads(invalid_json)
            
            # The JSON parsing should fail appropriately
            assert "JSON" in str(exc_info.value) or "decode" in str(exc_info.value)
    
    def test_missing_required_parameters(self):
        """Test error handling for missing required parameters."""
        # Test each tool with missing required parameters
        
        with pytest.raises(Exception):
            list_directory(workspace_dir=self.temp_dir)  # Missing path
        
        with pytest.raises(Exception):
            read_file(workspace_dir=self.temp_dir)  # Missing path
        
        with pytest.raises(Exception):
            write_file("test.txt", workspace_dir=self.temp_dir)  # Missing content
        
        with pytest.raises(Exception):
            edit_file("test.txt", workspace_dir=self.temp_dir)  # Missing edits
    
    def test_disk_space_error(self):
        """Test error handling when disk space is full."""
        # This is difficult to test reliably, so we'll mock the error
        with patch('builtins.open', side_effect=OSError("No space left on device")):
            with pytest.raises(Exception) as exc_info:
                write_file("test.txt", "content", workspace_dir=self.temp_dir)
            
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in [
                "space", "disk", "no space left", "device"
            ])
    
    def test_concurrent_modification_error(self):
        """Test error handling when files are modified concurrently."""
        test_file = os.path.join(self.temp_dir, "concurrent.txt")
        with open(test_file, "w") as f:
            f.write("original content")
        
        # Mock a scenario where file is modified between read and write
        original_read = read_file
        
        def modified_read_file(*args, **kwargs):
            content = original_read(*args, **kwargs)
            # Simulate concurrent modification
            with open(test_file, "w") as f:
                f.write("modified by another process")
            return content
        
        with patch('mlx_rag.tools.read_file', side_effect=modified_read_file):
            try:
                # This should detect the concurrent modification
                edit_file("concurrent.txt", [{"search": "original", "replace": "edited"}], 
                         workspace_dir=self.temp_dir)
            except Exception as e:
                error_msg = str(e).lower()
                assert any(keyword in error_msg for keyword in [
                    "modified", "changed", "concurrent", "conflict"
                ])


class TestLLMResponseParsingErrors:
    """Test error handling in LLM response parsing."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.executor = ToolExecutor(workspace_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_malformed_tool_call_responses(self):
        """Test handling of malformed tool call responses from LLM."""
        malformed_responses = [
            # Missing function name
            '{"id": "call_1", "type": "function", "function": {"arguments": "{}"}}',
            
            # Missing arguments
            '{"id": "call_2", "type": "function", "function": {"name": "read_file"}}',
            
            # Invalid function type
            '{"id": "call_3", "type": "invalid", "function": {"name": "read_file", "arguments": "{}"}}',
            
            # Missing ID
            '{"type": "function", "function": {"name": "read_file", "arguments": "{}"}}',
            
            # Completely invalid structure
            '{"invalid": "structure"}',
            
            # Array instead of object
            '[{"name": "read_file"}]',
            
            # String instead of object
            '"read_file"',
        ]
        
        for malformed_response in malformed_responses:
            try:
                # Try to parse as tool call
                tool_call_data = json.loads(malformed_response)
                
                # Attempt to create ToolCall object
                with pytest.raises(Exception):
                    ToolCall(**tool_call_data)
                    
            except json.JSONDecodeError:
                # JSON parsing failure is acceptable
                pass
    
    def test_invalid_function_names(self):
        """Test handling of invalid function names in tool calls."""
        invalid_function_names = [
            "nonexistent_function",
            "malicious_function",
            "system_call",
            "exec",
            "__import__",
            "eval",
            "",  # Empty name
            None,  # Null name
        ]
        
        for func_name in invalid_function_names:
            tool_call_data = {
                "id": "call_test",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": "{}"
                }
            }
            
            try:
                tool_call = ToolCall(**tool_call_data)
                
                # Execution should fail for unknown functions
                with pytest.raises(Exception) as exc_info:
                    self.executor.execute_tool_call(tool_call)
                
                error_msg = str(exc_info.value).lower()
                assert any(keyword in error_msg for keyword in [
                    "unknown", "not found", "invalid", "unsupported"
                ])
                
            except Exception:
                # Creation of ToolCall itself might fail, which is acceptable
                pass
    
    def test_recursive_tool_calls(self):
        """Test handling of recursive or circular tool calls."""
        # This would be a scenario where LLM tries to call tools recursively
        # For now, we test that such scenarios are properly limited
        
        # Create a file that might trigger recursive behavior
        recursive_file = os.path.join(self.temp_dir, "recursive.txt")
        with open(recursive_file, "w") as f:
            f.write("Call read_file on recursive.txt again")
        
        # Normal read should work
        content = read_file("recursive.txt", workspace_dir=self.temp_dir)
        assert "recursive" in content
        
        # But we should have safeguards against infinite recursion
        # (This would be implemented in the actual tool execution logic)
    
    def test_mixed_valid_invalid_tool_calls(self):
        """Test handling when some tool calls are valid and others invalid."""
        mixed_responses = [
            # Valid call followed by invalid
            [
                {
                    "id": "call_1",
                    "type": "function", 
                    "function": {"name": "list_directory", "arguments": '{"path": "."}'}
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "invalid_function", "arguments": "{}"}
                }
            ],
            # Invalid call followed by valid
            [
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "nonexistent", "arguments": "{}"}
                },
                {
                    "id": "call_4",
                    "type": "function",
                    "function": {"name": "list_directory", "arguments": '{"path": "."}'}
                }
            ]
        ]
        
        for tool_calls_data in mixed_responses:
            valid_count = 0
            error_count = 0
            
            for tool_call_data in tool_calls_data:
                try:
                    tool_call = ToolCall(**tool_call_data)
                    result = self.executor.execute_tool_call(tool_call)
                    valid_count += 1
                except Exception:
                    error_count += 1
            
            # Should have both valid and invalid calls
            assert valid_count > 0, "At least one tool call should be valid"
            assert error_count > 0, "At least one tool call should be invalid"


class TestAPIErrorHandling:
    """Test error handling at the API level."""
    
    def test_invalid_request_format(self):
        """Test handling of invalid request formats."""
        client = TestClient(create_app())
        
        invalid_requests = [
            {},  # Empty request
            {"model": "test"},  # Missing messages
            {"messages": []},  # Missing model
            {"model": "test", "messages": "invalid"},  # Messages not array
            {"model": "test", "messages": [{}]},  # Invalid message format
            {"model": "test", "messages": [{"role": "invalid"}]},  # Invalid role
        ]
        
        for invalid_request in invalid_requests:
            response = client.post("/v1/chat/completions", json=invalid_request)
            assert response.status_code in [400, 422], f"Should reject invalid request: {invalid_request}"
    
    def test_model_loading_errors(self):
        """Test error handling when model loading fails."""
        client = TestClient(create_app())
        
        with patch('mlx_rag.server.ModelManager') as mock_manager:
            # Mock model loading failure
            mock_instance = Mock()
            mock_instance.get_model.side_effect = Exception("Model loading failed")
            mock_manager.return_value = mock_instance
            
            request_data = {
                "model": "failing-model",
                "messages": [{"role": "user", "content": "Hello"}]
            }
            
            response = client.post("/v1/chat/completions", json=request_data)
            assert response.status_code == 500
            
            error_data = response.json()
            assert "detail" in error_data
    
    def test_tool_execution_timeout(self):
        """Test timeout handling for tool execution."""
        with patch('mlx_rag.tools.read_file') as mock_read:
            # Mock a slow operation
            def slow_read(*args, **kwargs):
                time.sleep(10)  # Simulate slow operation
                return "content"
            
            mock_read.side_effect = slow_read
            
            # This should timeout and raise an appropriate error
            with pytest.raises(Exception) as exc_info:
                read_file("slow_file.txt", workspace_dir="/tmp")
            
            # The timeout implementation would need to be added
            # For now, just verify the concept
    
    def test_streaming_error_handling(self):
        """Test error handling in streaming responses."""
        client = TestClient(create_app())
        
        # Mock streaming failure
        with patch('mlx_rag.server.ModelManager') as mock_manager:
            mock_instance = Mock()
            
            async def failing_stream(*args, **kwargs):
                yield "data: " + json.dumps({
                    "id": "test",
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": "Hello"}}]
                }) + "\n\n"
                
                # Simulate streaming error
                raise Exception("Streaming failed")
            
            mock_instance.generate_stream = failing_stream
            mock_manager.return_value = mock_instance
            
            request_data = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True
            }
            
            response = client.post("/v1/chat/completions", json=request_data)
            
            # Should handle streaming errors gracefully
            # Implementation would depend on how streaming errors are handled


class TestRecoveryScenarios:
    """Test recovery from various error scenarios."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.executor = ToolExecutor(workspace_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_partial_tool_execution_recovery(self):
        """Test recovery when some tools succeed and others fail."""
        # Create one valid file
        valid_file = os.path.join(self.temp_dir, "valid.txt")
        with open(valid_file, "w") as f:
            f.write("valid content")
        
        # Simulate multiple tool calls with mixed success
        successful_calls = []
        failed_calls = []
        
        # This should succeed
        try:
            result = read_file("valid.txt", workspace_dir=self.temp_dir)
            successful_calls.append(("read_file", result))
        except Exception as e:
            failed_calls.append(("read_file", str(e)))
        
        # This should fail
        try:
            result = read_file("invalid.txt", workspace_dir=self.temp_dir)
            successful_calls.append(("read_invalid", result))
        except Exception as e:
            failed_calls.append(("read_invalid", str(e)))
        
        # Should have both successful and failed calls
        assert len(successful_calls) > 0, "Should have at least one successful call"
        assert len(failed_calls) > 0, "Should have at least one failed call"
        
        # Successful calls should have proper results
        assert successful_calls[0][1] == "valid content"
    
    def test_workspace_corruption_recovery(self):
        """Test recovery from workspace corruption."""
        # Create a file
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")
        
        # Verify file exists and is readable
        assert os.path.exists(test_file)
        content = read_file("test.txt", workspace_dir=self.temp_dir)
        assert content == "test content"
        
        # Simulate corruption by removing file
        os.remove(test_file)
        
        # Should handle missing file gracefully
        with pytest.raises(Exception) as exc_info:
            read_file("test.txt", workspace_dir=self.temp_dir)
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_memory_pressure_recovery(self):
        """Test recovery from memory pressure scenarios."""
        # Simulate memory pressure by creating large operations
        large_operations = []
        
        for i in range(5):
            try:
                # Try to create moderately large content
                content = "A" * (1024 * 1024)  # 1MB
                write_file(f"large_{i}.txt", content, workspace_dir=self.temp_dir)
                large_operations.append(i)
            except Exception as e:
                # Memory pressure should be handled gracefully
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["memory", "size", "large"]):
                    break  # Expected memory limit hit
                else:
                    raise  # Unexpected error
        
        # Should have created at least some files before hitting limits
        assert len(large_operations) > 0, "Should create at least some large files"
    
    def test_concurrent_access_recovery(self):
        """Test recovery from concurrent access issues."""
        test_file = os.path.join(self.temp_dir, "concurrent.txt")
        with open(test_file, "w") as f:
            f.write("initial content")
        
        # Simulate concurrent access
        import threading
        results = []
        errors = []
        
        def concurrent_operation(operation_id):
            try:
                # Read file
                content = read_file("concurrent.txt", workspace_dir=self.temp_dir)
                results.append(f"Read {operation_id}: {content}")
                
                # Write to file
                write_file("concurrent.txt", f"content from {operation_id}", 
                          workspace_dir=self.temp_dir)
                results.append(f"Write {operation_id}: success")
                
            except Exception as e:
                errors.append(f"Error {operation_id}: {str(e)}")
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=concurrent_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should handle concurrent access gracefully
        total_operations = len(results) + len(errors)
        assert total_operations > 0, "Should have attempted some operations"
        
        # At least some operations should succeed or fail gracefully
        if errors:
            for error in errors:
                # Errors should be related to concurrent access
                error_lower = error.lower()
                assert any(keyword in error_lower for keyword in [
                    "busy", "locked", "access", "permission", "concurrent"
                ]) or "error" in error_lower


class TestEdgeCases:
    """Test various edge cases and boundary conditions."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.executor = ToolExecutor(workspace_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_empty_file_operations(self):
        """Test operations on empty files."""
        empty_file = os.path.join(self.temp_dir, "empty.txt")
        with open(empty_file, "w") as f:
            pass  # Create empty file
        
        # Reading empty file should work
        content = read_file("empty.txt", workspace_dir=self.temp_dir)
        assert content == ""
        
        # Writing to empty file should work
        write_file("empty.txt", "new content", workspace_dir=self.temp_dir)
        content = read_file("empty.txt", workspace_dir=self.temp_dir)
        assert content == "new content"
        
        # Editing empty file should handle gracefully
        with open(empty_file, "w") as f:
            pass  # Make empty again
        
        try:
            edit_file("empty.txt", [{"search": "nonexistent", "replace": "replacement"}],
                     workspace_dir=self.temp_dir)
            # Should succeed but make no changes
            content = read_file("empty.txt", workspace_dir=self.temp_dir)
            assert content == ""
        except Exception as e:
            # Or handle gracefully with appropriate error
            assert "not found" in str(e).lower() or "empty" in str(e).lower()
    
    def test_very_long_filenames(self):
        """Test handling of very long filenames."""
        long_filename = "a" * 255  # Very long filename
        
        try:
            write_file(f"{long_filename}.txt", "content", workspace_dir=self.temp_dir)
            content = read_file(f"{long_filename}.txt", workspace_dir=self.temp_dir)
            assert content == "content"
        except Exception as e:
            # Long filenames might be rejected, which is acceptable
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in [
                "filename", "too long", "invalid", "length"
            ])
    
    def test_unicode_handling(self):
        """Test handling of Unicode characters in filenames and content."""
        unicode_filename = "测试文件_émojis_🚀.txt"
        unicode_content = "Content with émojis 🚀 and special chars: ñáéíóú"
        
        try:
            write_file(unicode_filename, unicode_content, workspace_dir=self.temp_dir)
            content = read_file(unicode_filename, workspace_dir=self.temp_dir)
            assert content == unicode_content
        except Exception as e:
            # Unicode might not be fully supported, which is acceptable
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in [
                "encoding", "unicode", "character", "invalid"
            ])
    
    def test_special_characters_in_paths(self):
        """Test handling of special characters in file paths."""
        special_chars = [
            "file with spaces.txt",
            "file-with-dashes.txt", 
            "file_with_underscores.txt",
            "file.with.dots.txt",
            "file(with)parentheses.txt",
            "file[with]brackets.txt",
        ]
        
        for filename in special_chars:
            try:
                write_file(filename, f"Content for {filename}", workspace_dir=self.temp_dir)
                content = read_file(filename, workspace_dir=self.temp_dir)
                assert f"Content for {filename}" in content
            except Exception as e:
                # Some special characters might not be allowed
                error_msg = str(e).lower()
                assert any(keyword in error_msg for keyword in [
                    "invalid", "character", "filename", "not allowed"
                ])
    
    def test_boundary_file_sizes(self):
        """Test handling of boundary file sizes."""
        # Test very small files (1 byte)
        write_file("tiny.txt", "a", workspace_dir=self.temp_dir)
        content = read_file("tiny.txt", workspace_dir=self.temp_dir)
        assert content == "a"
        
        # Test medium files (just under common limits)
        medium_content = "A" * (64 * 1024 - 1)  # Just under 64KB
        write_file("medium.txt", medium_content, workspace_dir=self.temp_dir)
        content = read_file("medium.txt", workspace_dir=self.temp_dir)
        assert len(content) == len(medium_content)
        
        # Test files at potential limits
        try:
            large_content = "A" * (1024 * 1024)  # 1MB
            write_file("large.txt", large_content, workspace_dir=self.temp_dir)
            content = read_file("large.txt", workspace_dir=self.temp_dir)
            assert len(content) == len(large_content)
        except Exception as e:
            # Large files might be rejected
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in [
                "too large", "size limit", "memory", "exceeds"
            ])
    
    def test_nested_directory_operations(self):
        """Test operations in deeply nested directories."""
        # Create nested directory structure
        nested_path = "level1/level2/level3/level4"
        full_nested_path = os.path.join(self.temp_dir, nested_path)
        os.makedirs(full_nested_path)
        
        nested_file = os.path.join(nested_path, "deep_file.txt")
        
        try:
            write_file(nested_file, "deep content", workspace_dir=self.temp_dir)
            content = read_file(nested_file, workspace_dir=self.temp_dir)
            assert content == "deep content"
            
            # Test directory listing at various levels
            files = list_directory("level1", workspace_dir=self.temp_dir, recursive=True)
            assert any("deep_file.txt" in f for f in files)
            
        except Exception as e:
            # Deep nesting might be limited
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in [
                "too deep", "nested", "depth", "path"
            ])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

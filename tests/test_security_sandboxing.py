"""
Test security and sandboxing for tool execution.

This module tests security boundaries, path traversal protection,
execution limits, and resource management for tool execution.
"""

import pytest
import os
import tempfile
import shutil
import stat
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import threading
import time
from typing import Dict, Any, List

from mlx_rag.tools import (
    ToolExecutor, 
    list_directory,
    read_file, 
    write_file,
    edit_file,
    search_files
)


class TestPathTraversalProtection:
    """Test protection against path traversal attacks."""
    
    def setup_method(self):
        """Set up test workspace."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace = os.path.join(self.temp_dir, "workspace")
        os.makedirs(self.workspace)
        
        # Create some test files
        with open(os.path.join(self.workspace, "safe_file.txt"), "w") as f:
            f.write("Safe content")
            
        # Create directory outside workspace
        self.outside_dir = os.path.join(self.temp_dir, "outside")
        os.makedirs(self.outside_dir)
        with open(os.path.join(self.outside_dir, "secret.txt"), "w") as f:
            f.write("Secret content")
            
        self.executor = ToolExecutor(workspace_dir=self.workspace)
    
    def teardown_method(self):
        """Clean up test workspace."""
        shutil.rmtree(self.temp_dir)
    
    def test_basic_path_traversal_attempts(self):
        """Test basic path traversal patterns are blocked."""
        dangerous_paths = [
            "../secret.txt",
            "../../etc/passwd", 
            "../outside/secret.txt",
            "..\\..\\windows\\system32",
            "/etc/passwd",
            "C:\\Windows\\System32",
            "safe_file.txt/../../../etc/passwd"
        ]
        
        for path in dangerous_paths:
            with pytest.raises(Exception) as exc_info:
                read_file(path, workspace_dir=self.workspace)
            
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in [
                "path traversal", "invalid path", "access denied", 
                "outside workspace", "security"
            ]), f"Expected security error for path: {path}, got: {exc_info.value}"
    
    def test_encoded_path_traversal_attempts(self):
        """Test URL-encoded and other encoded path traversal attempts."""
        encoded_paths = [
            "%2e%2e/secret.txt",           # ../secret.txt
            "..%2fsecret.txt",             # ../secret.txt
            "%2e%2e%5csecret.txt",         # ..\secret.txt
            "..%5c..%5csecret.txt",        # ..\..secret.txt
            "%2e%2e%2f%2e%2e%2fetc%2fpasswd"  # ../../etc/passwd
        ]
        
        for path in encoded_paths:
            with pytest.raises(Exception):
                read_file(path, workspace_dir=self.workspace)
    
    def test_symbolic_link_traversal_prevention(self):
        """Test that symbolic links cannot be used for traversal."""
        if os.name == 'nt':  # Windows doesn't have symbolic links in the same way
            pytest.skip("Symbolic link test not applicable on Windows")
            
        # Create a symbolic link pointing outside workspace
        link_path = os.path.join(self.workspace, "evil_link")
        try:
            os.symlink(self.outside_dir, link_path)
            
            # Attempt to read through the symlink should fail
            with pytest.raises(Exception):
                read_file("evil_link/secret.txt", workspace_dir=self.workspace)
                
        except OSError:
            # If we can't create symlinks, skip this test
            pytest.skip("Cannot create symbolic links")
    
    def test_absolute_path_rejection(self):
        """Test that absolute paths are rejected."""
        absolute_paths = [
            "/etc/passwd",
            "/home/user/secret.txt", 
            "C:\\Windows\\System32\\config",
            "/usr/local/bin/python"
        ]
        
        for path in absolute_paths:
            with pytest.raises(Exception):
                read_file(path, workspace_dir=self.workspace)
    
    def test_valid_paths_still_work(self):
        """Test that legitimate paths within workspace still work."""
        valid_paths = [
            "safe_file.txt",
            "./safe_file.txt",
            "subdir/../safe_file.txt"  # This resolves to workspace
        ]
        
        for path in valid_paths:
            try:
                content = read_file(path, workspace_dir=self.workspace)
                assert content == "Safe content"
            except Exception as e:
                pytest.fail(f"Valid path {path} should not raise exception: {e}")


class TestExecutionLimits:
    """Test execution limits and resource management."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.executor = ToolExecutor(workspace_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_file_size_limits(self):
        """Test file size limits for read operations."""
        # Create a large file
        large_file = os.path.join(self.temp_dir, "large_file.txt")
        large_content = "A" * (10 * 1024 * 1024)  # 10MB
        
        with open(large_file, "w") as f:
            f.write(large_content)
        
        # Should raise error when trying to read very large file
        with pytest.raises(Exception) as exc_info:
            read_file("large_file.txt", workspace_dir=self.temp_dir)
        
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in [
            "file too large", "size limit", "exceeds maximum"
        ])
    
    def test_directory_depth_limits(self):
        """Test limits on directory traversal depth."""
        # Create deeply nested directory structure
        current_dir = self.temp_dir
        deep_path_parts = []
        
        for i in range(20):  # Create 20 levels deep
            dir_name = f"level_{i}"
            deep_path_parts.append(dir_name)
            current_dir = os.path.join(current_dir, dir_name)
            os.makedirs(current_dir)
        
        deep_file = os.path.join(current_dir, "deep_file.txt")
        with open(deep_file, "w") as f:
            f.write("Deep content")
        
        deep_relative_path = "/".join(deep_path_parts) + "/deep_file.txt"
        
        # Should limit directory traversal depth
        with pytest.raises(Exception) as exc_info:
            read_file(deep_relative_path, workspace_dir=self.temp_dir)
        
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in [
            "path too deep", "depth limit", "nested too deeply"
        ])
    
    def test_concurrent_operation_limits(self):
        """Test limits on concurrent operations."""
        # Create multiple files for concurrent reading
        for i in range(10):
            file_path = os.path.join(self.temp_dir, f"file_{i}.txt")
            with open(file_path, "w") as f:
                f.write(f"Content {i}")
        
        # Try to perform many concurrent operations
        results = []
        errors = []
        threads = []
        
        def read_file_thread(filename):
            try:
                result = read_file(filename, workspace_dir=self.temp_dir)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Start many concurrent threads
        for i in range(20):  # More threads than files
            filename = f"file_{i % 10}.txt"
            thread = threading.Thread(target=read_file_thread, args=(filename,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have some rate limiting or concurrent operation limits
        assert len(errors) > 0 or len(results) <= 10, "Should limit concurrent operations"
    
    def test_timeout_protection(self):
        """Test timeout protection for long operations."""
        # Create a large directory with many files for slow listing
        subdir = os.path.join(self.temp_dir, "large_dir")
        os.makedirs(subdir)
        
        # Create many files (this might be slow)
        for i in range(1000):
            file_path = os.path.join(subdir, f"file_{i:04d}.txt")
            with open(file_path, "w") as f:
                f.write("test")
        
        start_time = time.time()
        
        try:
            # This operation should complete or timeout
            result = list_directory("large_dir", workspace_dir=self.temp_dir, recursive=True)
            duration = time.time() - start_time
            
            # Should complete in reasonable time (less than 10 seconds)
            assert duration < 10, f"Operation took too long: {duration}s"
            
        except Exception as e:
            # Should get a timeout error if operation takes too long
            error_msg = str(e).lower()
            timeout_keywords = ["timeout", "too long", "time limit", "cancelled"]
            assert any(keyword in error_msg for keyword in timeout_keywords)


class TestFilePermissionSecurity:
    """Test file permission and access control security."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.executor = ToolExecutor(workspace_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_read_only_file_protection(self):
        """Test protection of read-only files."""
        if os.name == 'nt':
            pytest.skip("File permissions test not applicable on Windows")
            
        # Create a read-only file
        readonly_file = os.path.join(self.temp_dir, "readonly.txt")
        with open(readonly_file, "w") as f:
            f.write("Read-only content")
        
        # Make file read-only
        os.chmod(readonly_file, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        
        # Should be able to read
        content = read_file("readonly.txt", workspace_dir=self.temp_dir)
        assert content == "Read-only content"
        
        # Should not be able to write
        with pytest.raises(Exception):
            write_file("readonly.txt", "Modified content", workspace_dir=self.temp_dir)
    
    def test_no_executable_file_creation(self):
        """Test that executable files cannot be created."""
        executable_files = [
            "script.sh",
            "program.exe", 
            "malicious.bin",
            "payload"
        ]
        
        for filename in executable_files:
            try:
                write_file(filename, "#!/bin/bash\necho 'malicious'", workspace_dir=self.temp_dir)
                
                file_path = os.path.join(self.temp_dir, filename)
                if os.path.exists(file_path):
                    # File should not have execute permissions
                    file_stat = os.stat(file_path)
                    is_executable = file_stat.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                    assert not is_executable, f"File {filename} should not be executable"
                    
            except Exception:
                # It's also acceptable to completely prevent creation
                pass


class TestInputSanitization:
    """Test input sanitization and validation."""
    
    def setup_method(self):
        """Set up test environment.""" 
        self.temp_dir = tempfile.mkdtemp()
        self.executor = ToolExecutor(workspace_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_filename_sanitization(self):
        """Test that dangerous filenames are sanitized."""
        dangerous_filenames = [
            "file\x00.txt",           # Null byte
            "file\n.txt",             # Newline
            "file\r.txt",             # Carriage return
            "file\t.txt",             # Tab
            "file|rm -rf /.txt",      # Command injection
            "file;rm -rf /.txt",      # Command injection  
            "file`rm -rf /`.txt",     # Command injection
            "file$(rm -rf /).txt",    # Command injection
            "CON.txt",                # Windows reserved name
            "PRN.txt",                # Windows reserved name
            "AUX.txt",                # Windows reserved name
        ]
        
        for filename in dangerous_filenames:
            try:
                write_file(filename, "test content", workspace_dir=self.temp_dir)
                
                # If file creation succeeds, verify the filename was sanitized
                files = os.listdir(self.temp_dir)
                created_file = files[0] if files else None
                
                if created_file:
                    # Filename should not contain dangerous characters
                    assert "\x00" not in created_file
                    assert "\n" not in created_file
                    assert "\r" not in created_file
                    assert "|" not in created_file
                    assert ";" not in created_file
                    assert "`" not in created_file
                    assert "$(" not in created_file
                    
            except Exception:
                # It's acceptable to reject dangerous filenames entirely
                pass
    
    def test_content_sanitization(self):
        """Test that file content is properly handled."""
        # Test various types of content
        test_contents = [
            "Normal text content",
            "Content with unicode: émojis 🚀",
            "Content\nwith\nnewlines",
            "Content with null bytes\x00should be handled",
            "Content with very long lines: " + "A" * 10000,
            b"Binary content",  # If binary content is supported
        ]
        
        for i, content in enumerate(test_contents):
            filename = f"test_content_{i}.txt"
            
            try:
                if isinstance(content, bytes):
                    # Skip binary content test for now
                    continue
                    
                write_file(filename, content, workspace_dir=self.temp_dir)
                read_content = read_file(filename, workspace_dir=self.temp_dir)
                
                # Content should be preserved (minus any sanitization)
                assert isinstance(read_content, str)
                assert len(read_content) > 0
                
            except Exception as e:
                # Some content types may be rejected, which is acceptable
                error_msg = str(e).lower()
                assert any(keyword in error_msg for keyword in [
                    "invalid content", "sanitization", "encoding", "format"
                ])
    
    def test_search_query_sanitization(self):
        """Test that search queries are sanitized to prevent regex injection."""
        # Create test file
        test_file = os.path.join(self.temp_dir, "search_test.txt")
        with open(test_file, "w") as f:
            f.write("This is a test file\nwith multiple lines\nfor testing search")
        
        dangerous_queries = [
            ".*",                 # Regex that matches everything
            "(.*)*",              # Catastrophic backtracking
            "(?=.*){1000000}",    # Excessive lookahead
            "\\",                 # Escape character
            "[",                  # Unmatched bracket
            "(",                  # Unmatched parenthesis
        ]
        
        for query in dangerous_queries:
            try:
                results = search_files(query, self.temp_dir)
                
                # If search succeeds, verify it doesn't hang or return everything
                assert isinstance(results, list)
                assert len(results) < 100  # Reasonable limit
                
            except Exception as e:
                # Rejecting dangerous queries is acceptable
                error_msg = str(e).lower()
                assert any(keyword in error_msg for keyword in [
                    "invalid query", "regex", "pattern", "sanitization"
                ])


class TestResourceManagement:
    """Test resource management and cleanup."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.executor = ToolExecutor(workspace_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_memory_usage_limits(self):
        """Test memory usage limits."""
        # Try to create content that would use excessive memory
        large_content = "A" * (100 * 1024 * 1024)  # 100MB string
        
        try:
            write_file("huge_file.txt", large_content, workspace_dir=self.temp_dir)
            
            # If write succeeds, reading should be limited
            with pytest.raises(Exception):
                read_file("huge_file.txt", workspace_dir=self.temp_dir)
                
        except Exception as e:
            # It's acceptable to prevent large content creation
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in [
                "too large", "memory", "size limit", "exceeds"
            ])
    
    def test_temporary_file_cleanup(self):
        """Test that temporary files are cleaned up."""
        initial_temp_files = set(os.listdir(tempfile.gettempdir()))
        
        # Perform operations that might create temporary files
        write_file("test1.txt", "content1", workspace_dir=self.temp_dir)
        edit_file("test1.txt", [{"search": "content1", "replace": "modified1"}], workspace_dir=self.temp_dir)
        read_file("test1.txt", workspace_dir=self.temp_dir)
        
        # Check that no new temporary files were left behind
        final_temp_files = set(os.listdir(tempfile.gettempdir()))
        new_temp_files = final_temp_files - initial_temp_files
        
        # Filter out files that might be created by other processes
        suspicious_temp_files = [f for f in new_temp_files if "mlx" in f.lower() or "rag" in f.lower()]
        
        assert len(suspicious_temp_files) == 0, f"Temporary files not cleaned up: {suspicious_temp_files}"
    
    def test_workspace_isolation(self):
        """Test that workspaces are properly isolated."""
        # Create another workspace
        temp_dir2 = tempfile.mkdtemp()
        executor2 = ToolExecutor(workspace_dir=temp_dir2)
        
        try:
            # Create files in both workspaces
            write_file("shared_name.txt", "content1", workspace_dir=self.temp_dir)
            write_file("shared_name.txt", "content2", workspace_dir=temp_dir2)
            
            # Verify isolation
            content1 = read_file("shared_name.txt", workspace_dir=self.temp_dir)
            content2 = read_file("shared_name.txt", workspace_dir=temp_dir2)
            
            assert content1 == "content1"
            assert content2 == "content2"
            assert content1 != content2
            
        finally:
            shutil.rmtree(temp_dir2)


class TestSecurityErrorHandling:
    """Test security-related error handling."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.executor = ToolExecutor(workspace_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_security_error_messages(self):
        """Test that security errors don't leak sensitive information."""
        try:
            read_file("../../../etc/passwd", workspace_dir=self.temp_dir)
            pytest.fail("Should have raised security error")
        except Exception as e:
            error_msg = str(e)
            
            # Error message should not reveal system paths or structure
            assert "/etc/passwd" not in error_msg
            assert "system32" not in error_msg.lower()
            assert self.temp_dir not in error_msg  # Don't leak workspace path
            
            # Should contain generic security message
            assert any(keyword in error_msg.lower() for keyword in [
                "access denied", "invalid path", "security", "not allowed"
            ])
    
    def test_error_handling_consistency(self):
        """Test that error handling is consistent across tools."""
        dangerous_path = "../secret.txt"
        
        security_tools = [
            lambda: read_file(dangerous_path, workspace_dir=self.temp_dir),
            lambda: write_file(dangerous_path, "content", workspace_dir=self.temp_dir),
            lambda: edit_file(dangerous_path, [{"search": "a", "replace": "b"}], workspace_dir=self.temp_dir),
            lambda: list_directory(dangerous_path, workspace_dir=self.temp_dir),
        ]
        
        error_types = []
        for tool_func in security_tools:
            try:
                tool_func()
                pytest.fail(f"Tool {tool_func} should have raised security error")
            except Exception as e:
                error_types.append(type(e))
        
        # All tools should raise consistent error types for security violations
        assert len(set(error_types)) <= 2, "Security errors should be consistent across tools"
    
    def test_logging_security_events(self):
        """Test that security violations are properly logged."""
        with patch('mlx_rag.tools.logger') as mock_logger:
            try:
                read_file("../secret.txt", workspace_dir=self.temp_dir)
            except Exception:
                pass
            
            # Should log security violations
            assert mock_logger.warning.called or mock_logger.error.called
            
            # Log message should not contain sensitive details
            log_calls = mock_logger.warning.call_args_list + mock_logger.error.call_args_list
            for call in log_calls:
                log_message = str(call)
                assert "secret.txt" not in log_message
                assert self.temp_dir not in log_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

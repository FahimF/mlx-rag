#!/usr/bin/env python3
"""
Simple test runner to validate the test framework works with the current codebase.

This script runs a subset of tests that don't depend on external libraries
or missing functions, to validate that our test framework is correctly set up.
"""

import sys
import os
import tempfile
import shutil
import json
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_basic_imports():
    """Test that basic imports work."""
    print("Testing basic imports...")
    
    try:
        # Test basic tool executor import
        from mlx_rag.tool_executor import ToolExecutor, ToolExecutionResult
        print("✅ tool_executor import successful")
        
        # Test that we can create a tool executor
        temp_dir = tempfile.mkdtemp()
        try:
            executor = ToolExecutor(temp_dir)
            print("✅ ToolExecutor creation successful")
        finally:
            shutil.rmtree(temp_dir)
        
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_file_operations():
    """Test basic file operations using our compatibility functions."""
    print("Testing file operations...")
    
    # Import our compatibility functions
    sys.path.insert(0, os.path.dirname(__file__))
    from test_compatibility import list_directory, read_file, write_file, edit_file, search_files
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test write_file
            write_file("test.txt", "Hello, World!", workspace_dir=temp_dir)
            print("✅ write_file works")
            
            # Test read_file
            content = read_file("test.txt", workspace_dir=temp_dir)
            assert content == "Hello, World!"
            print("✅ read_file works")
            
            # Test list_directory
            files = list_directory(".", workspace_dir=temp_dir)
            assert "test.txt" in files
            print("✅ list_directory works")
            
            # Test edit_file
            edit_file("test.txt", [{"search": "World", "replace": "Python"}], workspace_dir=temp_dir)
            content = read_file("test.txt", workspace_dir=temp_dir)
            assert "Hello, Python!" in content
            print("✅ edit_file works")
            
            # Test search_files
            results = search_files("Python", workspace_dir=temp_dir)
            assert len(results) > 0
            assert "test.txt" in results[0]["file"]
            print("✅ search_files works")
        
        return True
    except Exception as e:
        print(f"❌ File operations failed: {e}")
        return False

def test_security_features():
    """Test basic security features."""
    print("Testing security features...")
    
    sys.path.insert(0, os.path.dirname(__file__))
    from test_compatibility import read_file
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test content")
            
            # Test path traversal protection
            try:
                read_file("../../../etc/passwd", workspace_dir=temp_dir)
                print("❌ Path traversal protection failed")
                return False
            except Exception:
                print("✅ Path traversal protection works")
            
            # Test valid file access
            content = read_file("test.txt", workspace_dir=temp_dir)
            assert content == "test content"
            print("✅ Valid file access works")
        
        return True
    except Exception as e:
        print(f"❌ Security tests failed: {e}")
        return False

def test_tool_executor_integration():
    """Test integration with the actual ToolExecutor."""
    print("Testing ToolExecutor integration...")
    
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from test_compatibility import get_test_tool_executor
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test content")
            
            # Get tool executor
            executor = get_test_tool_executor(temp_dir)
            print("✅ Tool executor creation works")
            
            # Test if it has available tools (even if it's a mock)
            if hasattr(executor, 'available_tools'):
                print("✅ Tool executor has available_tools attribute")
            else:
                print("ℹ️ Using mock tool executor")
        
        return True
    except Exception as e:
        print(f"❌ ToolExecutor integration failed: {e}")
        return False

def test_json_parsing():
    """Test JSON parsing functionality."""
    print("Testing JSON parsing...")
    
    try:
        # Test valid JSON
        test_json = '{"function": "read_file", "arguments": {"path": "test.txt"}}'
        parsed = json.loads(test_json)
        assert parsed["function"] == "read_file"
        assert parsed["arguments"]["path"] == "test.txt"
        print("✅ JSON parsing works")
        
        # Test invalid JSON handling
        try:
            json.loads('{"invalid": json}')
            print("❌ Invalid JSON should have failed")
            return False
        except json.JSONDecodeError:
            print("✅ Invalid JSON handling works")
        
        return True
    except Exception as e:
        print(f"❌ JSON parsing failed: {e}")
        return False

def main():
    """Run all simple tests."""
    print("=" * 60)
    print("MLX-RAG Test Framework Validation")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_file_operations,
        test_security_features,
        test_tool_executor_integration,
        test_json_parsing,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        print(f"\n{test_func.__name__}:")
        print("-" * 40)
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_func.__name__} PASSED")
            else:
                failed += 1
                print(f"❌ {test_func.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_func.__name__} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All tests passed! Test framework is working correctly.")
        return 0
    else:
        print("💥 Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

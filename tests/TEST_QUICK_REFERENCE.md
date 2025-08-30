# MLX-RAG Tool Calling Tests - Quick Reference

This is a quick reference guide for developers who need to run, understand, or extend the MLX-RAG tool calling tests.

## 🚀 Quick Start

### Run All Tests (Recommended)
```bash
./tests/run_integration_tests.py
```

### Run Individual Test Suites
```bash
# Tool execution tests
pytest tests/test_tool_execution.py -v

# API compatibility tests
pytest tests/test_openai_compatibility.py -v

# Security tests
pytest tests/test_security_sandboxing.py -v

# Error handling tests
pytest tests/test_error_handling.py -v
```

### Common Options
```bash
# Run with coverage report
./tests/run_integration_tests.py --coverage

# Run sequentially (for debugging)
./tests/run_integration_tests.py --sequential

# Keep test environment for inspection
./tests/run_integration_tests.py --no-cleanup

# System validation only
./tests/run_integration_tests.py --validate-only
```

## 📋 Test Suite Overview

| Test File | Purpose | Key Features |
|-----------|---------|--------------|
| `test_tool_execution.py` | Core functionality | LLM parsing, tool execution, workflows |
| `test_openai_compatibility.py` | API compliance | Request/response format, streaming |
| `test_security_sandboxing.py` | Security testing | Path traversal, limits, sanitization |
| `test_error_handling.py` | Error scenarios | Recovery, edge cases, resilience |
| `run_integration_tests.py` | Test orchestration | Parallel execution, reporting |

## 🔍 Test Categories

### ✅ What's Tested

**Core Functionality:**
- ✅ JSON, XML, and function-style LLM response parsing
- ✅ All tool operations (read, write, edit, list, search)
- ✅ Multi-tool workflows and realistic scenarios
- ✅ FastAPI integration and request processing

**Security & Safety:**
- ✅ Path traversal attack prevention
- ✅ Input sanitization and validation
- ✅ Resource limits and execution boundaries
- ✅ Workspace isolation and cleanup

**API Compatibility:**
- ✅ Complete OpenAI API format compliance
- ✅ Streaming response handling
- ✅ Error response formats
- ✅ Multi-turn conversation flows

**Error Handling:**
- ✅ File system errors and recovery
- ✅ Malformed LLM response handling
- ✅ Concurrent access and race conditions
- ✅ Edge cases and boundary conditions

## 🎯 Key Test Scenarios

### Tool Execution Flow
```python
def test_realistic_debugging_workflow(self):
    """End-to-end debugging scenario."""
    # 1. List project files
    files = list_directory(".", recursive=True)
    
    # 2. Read error-prone file
    content = read_file("buggy_code.py")
    
    # 3. Fix the bug
    edit_file("buggy_code.py", [{"search": "bug", "replace": "fix"}])
    
    # 4. Verify fix
    result = read_file("buggy_code.py")
    assert "fix" in result
```

### Security Testing
```python
def test_path_traversal_prevention(self):
    """Ensure path traversal attacks are blocked."""
    dangerous_paths = ["../../../etc/passwd", "..\\windows\\system32"]
    
    for path in dangerous_paths:
        with pytest.raises(SecurityError):
            read_file(path)
```

### API Compatibility
```python
def test_openai_request_format(self):
    """Verify OpenAI API request format compliance."""
    request = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [{"type": "function", "function": {...}}],
        "tool_choice": "auto"
    }
    
    # Should parse without errors
    parsed = ChatCompletionRequest(**request)
    assert parsed.model == "gpt-3.5-turbo"
```

## 🐛 Debugging Tests

### Run Specific Test
```bash
# Run single test method
pytest tests/test_tool_execution.py::TestLLMResponseParsing::test_json_parsing -v

# Run with debugger
pytest tests/test_tool_execution.py::TestClass::test_method --pdb -s

# Run with maximum verbosity
pytest tests/test_tool_execution.py -vvv
```

### Inspect Test Environment
```bash
# Keep workspace after tests
./tests/run_integration_tests.py --no-cleanup

# Find workspace location (logged during execution)
ls -la /tmp/mlx_rag_test_*

# Check test logs
tail -f integration_test.log
grep "ERROR" integration_test.log
```

### Common Debug Patterns
```python
def test_with_debugging(self):
    """Example test with debugging aids."""
    # Print workspace location
    print(f"Workspace: {self.temp_dir}")
    
    # Create test file for inspection
    test_file = os.path.join(self.temp_dir, "debug.txt")
    with open(test_file, "w") as f:
        f.write("Debug content")
    
    # Add breakpoint for interactive debugging
    import pdb; pdb.set_trace()
    
    # Your test code here...
```

## 📊 Test Results

### Success Indicators
```
✅ tool_execution completed in 12.34s
✅ openai_compatibility completed in 8.76s
✅ security_sandboxing completed in 15.23s
✅ error_handling completed in 9.87s

✅ ALL TESTS PASSED
```

### Failure Analysis
```
❌ FAIL security_sandboxing (15.23s)
  Tests: 35 | Passed: 33 | Failed: 2 | Skipped: 0
  Failures:
    - FAILED test_unicode_handling - UnicodeEncodeError in filename
    - FAILED test_memory_limits - Memory limit not properly enforced
```

### Report Files
- `integration_test_report.txt` - Complete test report
- `integration_test.log` - Detailed execution log  
- `*_report.json` - Individual suite JSON reports

## 🛠️ Adding New Tests

### Extend Existing Test Class
```python
class TestToolExecution:
    def test_new_feature(self):
        """Test new functionality."""
        # Setup test environment
        self.create_test_file("input.txt", "test content")
        
        # Execute functionality
        result = new_tool_function("input.txt", param="value")
        
        # Verify results
        assert result.success
        assert "expected" in result.output
```

### Create New Test Class
```python
class TestNewFeature:
    """Test new feature functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        # Additional setup...
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_feature_behavior(self):
        """Test specific behavior."""
        # Test implementation...
```

### Add to Integration Runner
```python
# In run_integration_tests.py, add to suite_files dict:
suite_files = {
    # ... existing suites ...
    "new_feature": "test_new_feature.py",
}

# Add to suites list:
suites = [
    # ... existing suites ...
    "new_feature"
]
```

## 🔧 Environment Setup

### Required Dependencies
```bash
pip install pytest>=7.0.0
pip install pytest-json-report>=1.5.0
pip install coverage>=6.0.0  # Optional
```

### Environment Variables
```bash
# Enable test mode
export MLX_RAG_TEST_MODE=1

# Custom workspace (optional)
export MLX_RAG_TEST_WORKSPACE=/tmp/custom_workspace

# Debug mode (optional)
export MLX_RAG_DEBUG=1
```

### Workspace Structure
```
/tmp/mlx_rag_test_xyz123/
├── fixtures/
│   ├── sample.txt
│   ├── data.json
│   ├── code.py
│   └── subdir/
│       └── nested_file.txt
└── workspace_specific_files...
```

## 🚨 Common Issues

### Permission Errors
```bash
# Make script executable
chmod +x tests/run_integration_tests.py

# Check file permissions
ls -la /tmp/mlx_rag_test_*
```

### Import Errors
```bash
# Ensure MLX-RAG is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

### Memory Issues
```bash
# Run sequentially to reduce memory usage
./tests/run_integration_tests.py --sequential

# Clear old test workspaces
rm -rf /tmp/mlx_rag_test_*
```

### Test Failures
```bash
# Run failed tests individually
pytest tests/test_security_sandboxing.py::TestClass::test_method -v

# Check detailed logs
grep -A 10 -B 10 "FAILED" integration_test.log

# Keep environment for inspection
./tests/run_integration_tests.py --no-cleanup
```

## 📈 Performance Expectations

### Execution Times
- **Individual suite**: 5-15 seconds
- **Full test suite**: 30-60 seconds (parallel)
- **Sequential execution**: 60-120 seconds

### Resource Usage
- **Memory**: <100MB peak usage
- **Disk**: <1GB temporary files
- **File handles**: <50 concurrent handles

### Success Criteria
- **Coverage**: >90% line coverage
- **Pass rate**: >95% test success
- **Performance**: <2 minutes total execution

## 🔄 CI/CD Integration

### Basic CI Script
```bash
#!/bin/bash
set -e

# Install dependencies
pip install -r requirements-test.txt

# Run tests with coverage
./tests/run_integration_tests.py --coverage

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ All tests passed"
else
    echo "❌ Tests failed"
    exit 1
fi
```

### GitHub Actions Example
```yaml
- name: Run MLX-RAG Tests
  run: |
    pip install -r requirements-test.txt
    ./tests/run_integration_tests.py --coverage
    
- name: Upload Test Reports
  uses: actions/upload-artifact@v2
  if: always()
  with:
    name: test-reports
    path: |
      integration_test_report.txt
      *_report.json
```

## 📚 Additional Resources

### Documentation Files
- `docs/TESTING.md` - Complete testing documentation
- `docs/TEST_COVERAGE_MATRIX.md` - Detailed coverage analysis
- `tests/README.md` - Test suite overview

### Test Files Location
- `tests/test_*.py` - Individual test suites
- `tests/run_integration_tests.py` - Test orchestrator
- `tests/TEST_QUICK_REFERENCE.md` - This file

### Getting Help
1. Check the detailed logs: `integration_test.log`
2. Review test reports: `integration_test_report.txt`
3. Inspect workspace: `--no-cleanup` option
4. Use debugger: `--pdb` with pytest
5. Check documentation: `docs/TESTING.md`

---

💡 **Pro Tip**: Start with `./tests/run_integration_tests.py --validate-only` to quickly check if your environment is set up correctly!

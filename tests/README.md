# MLX-RAG Tool Calling Test Suite

This directory contains a comprehensive test suite for the MLX-RAG tool calling functionality, covering all aspects of the system from basic tool execution to security, error handling, and OpenAI API compatibility.

## Test Structure

### Core Test Files

1. **`test_tool_execution.py`** - Tool execution and LLM response parsing tests
   - LLM response parsing (JSON, XML, function-like formats)
   - Tool call validation and execution
   - Realistic scenario testing
   - Integration with chat completions endpoint

2. **`test_openai_compatibility.py`** - OpenAI API compatibility tests
   - Request/response format validation
   - Tool calling conversation flows
   - Streaming response handling
   - Parameter validation and error handling

3. **`test_security_sandboxing.py`** - Security and sandboxing tests
   - Path traversal protection
   - Execution limits and resource management
   - Input sanitization
   - File permission security

4. **`test_error_handling.py`** - Error handling and edge case tests
   - Tool execution errors
   - LLM response parsing errors
   - Recovery scenarios
   - Edge cases and boundary conditions

5. **`run_integration_tests.py`** - Comprehensive integration test runner
   - Orchestrates all test suites
   - Parallel and sequential execution modes
   - Detailed reporting and validation
   - System integration validation

## Running Tests

### Quick Start

Run all tests with the integration test runner:

```bash
# Run all tests in parallel (recommended)
./tests/run_integration_tests.py

# Run tests sequentially
./tests/run_integration_tests.py --sequential

# Run only system validation
./tests/run_integration_tests.py --validate-only
```

### Individual Test Suites

Run individual test suites using pytest:

```bash
# Tool execution tests
pytest tests/test_tool_execution.py -v

# OpenAI compatibility tests
pytest tests/test_openai_compatibility.py -v

# Security and sandboxing tests
pytest tests/test_security_sandboxing.py -v

# Error handling tests
pytest tests/test_error_handling.py -v
```

### Advanced Options

```bash
# Generate coverage report
./tests/run_integration_tests.py --coverage

# Keep test environment for debugging
./tests/run_integration_tests.py --no-cleanup

# Specify custom test directory
./tests/run_integration_tests.py --test-dir /path/to/tests
```

## Test Coverage

### Tool Execution Testing
- ✅ LLM response parsing (JSON, XML, function calls)
- ✅ Multiple tool call handling
- ✅ Tool call validation and sanitization
- ✅ File system operations (read, write, edit, list, search)
- ✅ Workspace isolation
- ✅ Integration with FastAPI endpoints

### OpenAI API Compatibility
- ✅ Request format validation
- ✅ Response structure compliance
- ✅ Tool choice parameter variations
- ✅ Streaming response format
- ✅ Conversation flow handling
- ✅ Error response formats

### Security & Sandboxing
- ✅ Path traversal attack prevention
- ✅ Absolute path rejection
- ✅ Symbolic link protection
- ✅ File size and execution limits
- ✅ Input sanitization (filenames, content, queries)
- ✅ Resource management and cleanup

### Error Handling
- ✅ File system errors (permissions, not found, corruption)
- ✅ Malformed LLM responses
- ✅ Invalid function names and parameters
- ✅ Concurrent modification handling
- ✅ Recovery from partial failures
- ✅ Edge cases and boundary conditions

## Test Environment

### Prerequisites

```bash
# Required packages
pip install pytest
pip install pytest-json-report  # For detailed reporting
pip install coverage            # For coverage reports (optional)
```

### Test Workspace

Tests automatically create isolated temporary workspaces for each test run:
- Temporary directories with controlled file structures
- Sample files in various formats (text, JSON, YAML, Python)
- Nested directory structures for traversal testing
- Automatic cleanup after test completion

### Environment Variables

The test suite uses several environment variables:
- `MLX_RAG_TEST_MODE=1` - Enables test mode
- `MLX_RAG_TEST_WORKSPACE` - Temporary workspace directory

## Integration Test Features

### Parallel Execution
- Test suites can run in parallel for faster execution
- Automatic resource management and isolation
- Real-time progress reporting

### Comprehensive Reporting
- Detailed test results with timing information
- Failure and error reporting with context
- System integration validation results
- Coverage reports (when enabled)

### System Validation
- Server startup verification
- Tool registration validation
- API endpoint testing
- End-to-end tool execution flow
- Error handling verification

## Test Results and Reports

### Console Output
The integration test runner provides real-time feedback:
```
INFO - Running test suite: tool_execution
✅ tool_execution completed in 12.34s
INFO - Running test suite: openai_compatibility  
✅ openai_compatibility completed in 8.76s
...
```

### Detailed Reports
- `integration_test_report.txt` - Complete test execution report
- `integration_test.log` - Detailed execution log
- Individual JSON reports for each test suite

### Example Report Structure
```
================================================================================
MLX-RAG TOOL CALLING INTEGRATION TEST REPORT
================================================================================
Total Duration: 45.67 seconds
Test Suites Run: 4

SUMMARY:
  Total Tests: 127
  Passed: 125
  Failed: 2
  Skipped: 0
  Errors: 0
  Warnings: 3

SUITE RESULTS:
==================================================
✅ PASS tool_execution (12.34s)
  Tests: 32 | Passed: 32 | Failed: 0 | Skipped: 0
...
```

## Debugging Failed Tests

### Individual Test Debugging
```bash
# Run specific test with detailed output
pytest tests/test_tool_execution.py::TestToolExecution::test_specific_case -v -s

# Run with Python debugger
pytest tests/test_tool_execution.py::TestClass::test_method --pdb
```

### Workspace Inspection
```bash
# Keep test workspace for manual inspection
./tests/run_integration_tests.py --no-cleanup

# The workspace path will be logged during test execution
```

### Log Analysis
Check `integration_test.log` for detailed execution information including:
- Test environment setup
- Individual test execution details
- Error traces and debugging information
- Resource cleanup operations

## Extending the Test Suite

### Adding New Tests
1. Add test methods to existing test classes
2. Follow the established patterns for setup/teardown
3. Use descriptive test names and docstrings
4. Include both positive and negative test cases

### Adding New Test Suites
1. Create new test file following naming convention `test_*.py`
2. Add suite mapping to `run_integration_tests.py`
3. Ensure proper isolation and cleanup
4. Update this README with new test coverage

### Test Data and Fixtures
- Use temporary directories for file operations
- Clean up resources in teardown methods
- Use realistic test data that covers edge cases
- Ensure tests are deterministic and repeatable

## Continuous Integration

This test suite is designed to run in CI/CD environments:
- All dependencies are clearly specified
- Tests are isolated and deterministic
- Comprehensive reporting for automated analysis
- Configurable execution modes (parallel/sequential)
- Exit codes indicate overall success/failure

For CI integration, use:
```bash
./tests/run_integration_tests.py --coverage
# Exit code 0 = all tests passed
# Exit code 1 = some tests failed
```

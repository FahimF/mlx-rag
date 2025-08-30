# MLX-RAG Tool Calling Test Suite Documentation

This document provides comprehensive documentation for the MLX-RAG tool calling test suite, including test architecture, detailed test case descriptions, and usage guidelines.

## Table of Contents

- [Overview](#overview)
- [Test Architecture](#test-architecture)
- [Test Suites](#test-suites)
- [Test Case Details](#test-case-details)
- [Running Tests](#running-tests)
- [Test Reports and Analysis](#test-reports-and-analysis)
- [Extending Tests](#extending-tests)
- [Troubleshooting](#troubleshooting)

## Overview

The MLX-RAG tool calling test suite is a comprehensive testing framework designed to validate all aspects of the tool calling functionality, from basic tool execution to advanced security and error handling scenarios. The suite consists of 127+ individual test cases organized into 4 specialized test modules plus an integration test orchestrator.

### Key Testing Objectives

1. **Functional Correctness** - Ensure tools work as expected
2. **Security Compliance** - Validate security boundaries and protections
3. **API Compatibility** - Ensure OpenAI API compatibility
4. **Error Resilience** - Test error handling and recovery
5. **Performance** - Validate resource management and limits

## Test Architecture

### Component Overview

```
tests/
├── test_tool_execution.py      # Core functionality tests
├── test_openai_compatibility.py # API compatibility tests
├── test_security_sandboxing.py # Security and safety tests
├── test_error_handling.py      # Error scenarios and recovery
├── run_integration_tests.py    # Test orchestrator
└── README.md                   # Quick reference guide
```

### Test Infrastructure

#### Workspace Management
- **Isolated Environments**: Each test creates its own temporary workspace
- **Automatic Cleanup**: All test artifacts are automatically cleaned up
- **Mock File Systems**: Controlled file structures for testing
- **Security Boundaries**: Tests are confined to temporary directories

#### Test Execution Framework
- **Parallel Execution**: Test suites can run concurrently
- **Resource Management**: Memory and execution time limits
- **Comprehensive Reporting**: Detailed success/failure reporting
- **CI/CD Integration**: Exit codes and reports for automation

## Test Suites

## 1. Tool Execution Tests (`test_tool_execution.py`)

### Purpose
Tests the core tool execution functionality, including LLM response parsing, tool validation, and execution workflows.

### Test Classes

#### `TestLLMResponseParsing`
Tests parsing of various LLM response formats:

- **JSON Format Parsing**
  - Valid JSON tool calls
  - Multiple tool calls in single response
  - Nested JSON structures
  
- **XML Format Parsing**
  - XML-tagged function calls
  - Mixed XML/text responses
  - Malformed XML handling

- **Function-like Format Parsing**
  - Natural language function calls
  - Parameter extraction from text
  - Mixed format responses

#### `TestToolCallValidation`
Validates tool call structure and parameters:

- **Schema Validation**
  - Required field validation
  - Parameter type checking
  - Unknown parameter handling

- **Security Validation**
  - Malicious parameter detection
  - Path traversal prevention in parameters
  - Input sanitization

#### `TestToolExecution`
Tests actual tool execution with file system operations:

- **File Operations**
  - `read_file`: Reading various file types
  - `write_file`: Creating and modifying files
  - `edit_file`: Search and replace operations
  - `list_directory`: Directory listing with recursion
  - `search_files`: Content search with regex

- **Workspace Isolation**
  - Cross-workspace access prevention
  - Path resolution testing
  - Permission boundaries

#### `TestRealisticScenarios`
End-to-end scenarios simulating real usage:

- **Code Exploration Workflow**
  1. List project directories
  2. Read source files
  3. Search for specific patterns
  4. Analyze code structure

- **Debugging Workflow**
  1. Identify error locations
  2. Read relevant files
  3. Edit code to fix issues
  4. Verify changes

- **Documentation Workflow**
  1. Scan project files
  2. Extract documentation
  3. Generate summaries
  4. Update documentation files

#### `TestChatCompletionsIntegration`
Integration with FastAPI chat completions endpoint:

- **Request Processing**
  - Tool call extraction from requests
  - Parameter validation
  - Response formatting

- **Error Handling**
  - Invalid request handling
  - Tool execution error propagation
  - Graceful degradation

### Key Test Cases

```python
def test_json_tool_call_parsing(self):
    """Test parsing of JSON-formatted tool calls from LLM responses."""
    # Tests various JSON formats and edge cases

def test_multiple_tool_calls_execution(self):
    """Test execution of multiple tool calls in sequence."""
    # Validates parallel and sequential execution

def test_realistic_debugging_workflow(self):
    """Test a complete debugging workflow scenario."""
    # End-to-end scenario testing
```

## 2. OpenAI API Compatibility Tests (`test_openai_compatibility.py`)

### Purpose
Ensures complete compatibility with OpenAI's tool calling API format and conventions.

### Test Classes

#### `TestOpenAIRequestFormat`
Validates incoming request format compliance:

- **Basic Structure Validation**
  - Required fields presence
  - Data type validation
  - Schema compliance

- **Tool Definition Format**
  - Function schema validation
  - Parameter specification
  - Description requirements

- **Message Format Validation**
  - Role-based message structure
  - Tool call message format
  - Tool response message format

#### `TestOpenAIResponseFormat`
Validates outgoing response format compliance:

- **Chat Completion Response**
  - Response object structure
  - Choice array format
  - Usage statistics

- **Streaming Response Format**
  - Chunk-based streaming
  - Delta object structure
  - Stream termination

- **Tool Call Response Format**
  - Tool call object structure
  - Function call specification
  - Multiple tool call handling

#### `TestOpenAIErrorHandling`
Tests error response format compliance:

- **Error Response Structure**
  - Error object format
  - Error type classification
  - Error message formatting

- **HTTP Status Code Compliance**
  - 400 for bad requests
  - 404 for missing resources
  - 500 for server errors

#### `TestOpenAIConversationFlow`
Tests complete conversation scenarios:

- **Multi-turn Conversations**
  - User → Assistant → Tool → Assistant flow
  - Context maintenance across turns
  - State management

- **Parallel Tool Calls**
  - Multiple simultaneous tool calls
  - Result aggregation
  - Error handling in parallel execution

### Key Test Cases

```python
def test_tool_choice_variations(self):
    """Test different tool_choice parameter values."""
    # Tests "auto", "none", and specific function choices

def test_streaming_response_structure(self):
    """Test streaming response chunk format."""
    # Validates SSE format compliance

def test_tool_calling_conversation_flow(self):
    """Test complete multi-turn tool calling conversation."""
    # End-to-end conversation testing
```

## 3. Security and Sandboxing Tests (`test_security_sandboxing.py`)

### Purpose
Validates security boundaries and protection mechanisms against malicious inputs and attacks.

### Test Classes

#### `TestPathTraversalProtection`
Tests protection against path traversal attacks:

- **Basic Path Traversal**
  - `../` sequences
  - Absolute path attempts
  - Windows path traversal (`..\\`)

- **Encoded Path Traversal**
  - URL-encoded sequences (`%2e%2e`)
  - Double encoding
  - Mixed encoding schemes

- **Symbolic Link Protection**
  - Symlink traversal prevention
  - Hard link restrictions
  - Junction point protection (Windows)

- **Valid Path Testing**
  - Legitimate relative paths
  - Subdirectory access
  - File resolution validation

#### `TestExecutionLimits`
Tests resource limits and execution boundaries:

- **File Size Limits**
  - Large file handling
  - Memory usage limits
  - Read/write size restrictions

- **Directory Depth Limits**
  - Deep nesting prevention
  - Recursive operation limits
  - Path length restrictions

- **Concurrent Operation Limits**
  - Simultaneous operation limits
  - Resource contention handling
  - Thread safety validation

- **Timeout Protection**
  - Long operation timeouts
  - Resource cleanup on timeout
  - Graceful termination

#### `TestInputSanitization`
Tests input validation and sanitization:

- **Filename Sanitization**
  - Special character removal
  - Reserved name handling
  - Length restrictions

- **Content Sanitization**
  - Binary content handling
  - Character encoding validation
  - Malicious content detection

- **Query Sanitization**
  - Regex injection prevention
  - SQL injection protection
  - Command injection prevention

#### `TestResourceManagement`
Tests resource cleanup and management:

- **Memory Management**
  - Memory leak prevention
  - Large object handling
  - Garbage collection validation

- **File Handle Management**
  - File descriptor limits
  - Handle cleanup
  - Resource leak prevention

- **Temporary File Cleanup**
  - Automatic cleanup verification
  - Failed operation cleanup
  - System temp directory protection

### Key Test Cases

```python
def test_basic_path_traversal_attempts(self):
    """Test blocking of basic path traversal patterns."""
    # Tests various ../ patterns and absolute paths

def test_file_size_limits(self):
    """Test file size limitations for security."""
    # Validates memory and disk usage limits

def test_input_sanitization(self):
    """Test sanitization of user inputs."""
    # Validates malicious input filtering
```

## 4. Error Handling Tests (`test_error_handling.py`)

### Purpose
Tests error handling, recovery mechanisms, and edge case scenarios to ensure system resilience.

### Test Classes

#### `TestToolExecutionErrors`
Tests error handling in tool execution:

- **File System Errors**
  - File not found errors
  - Permission denied errors
  - Disk space errors
  - Corruption handling

- **Parameter Errors**
  - Missing required parameters
  - Invalid parameter types
  - Out-of-range values

- **Concurrent Modification**
  - File modification detection
  - Race condition handling
  - Lock contention

#### `TestLLMResponseParsingErrors`
Tests error handling in response parsing:

- **Malformed Responses**
  - Invalid JSON format
  - Incomplete XML tags
  - Mixed format errors

- **Invalid Function Names**
  - Unknown function calls
  - Restricted function names
  - Malicious function attempts

- **Parameter Errors**
  - Missing function parameters
  - Invalid parameter formats
  - Type mismatch errors

#### `TestRecoveryScenarios`
Tests recovery from various failure scenarios:

- **Partial Failure Recovery**
  - Mixed success/failure handling
  - State rollback mechanisms
  - Error propagation

- **Resource Recovery**
  - Memory pressure recovery
  - Disk space recovery
  - Network failure recovery

- **Corruption Recovery**
  - File corruption detection
  - Data integrity validation
  - Backup and recovery

#### `TestEdgeCases`
Tests boundary conditions and edge cases:

- **Empty Input Handling**
  - Empty files
  - Empty directories
  - Null inputs

- **Large Input Handling**
  - Very long filenames
  - Large file contents
  - Deep directory structures

- **Special Character Handling**
  - Unicode characters
  - Special symbols
  - Control characters

### Key Test Cases

```python
def test_nonexistent_file_error(self):
    """Test error handling for missing files."""
    # Validates appropriate error messages

def test_malformed_tool_call_responses(self):
    """Test handling of malformed LLM responses."""
    # Tests various malformation scenarios

def test_recovery_scenarios(self):
    """Test recovery from partial failures."""
    # Validates system resilience
```

## 5. Integration Test Runner (`run_integration_tests.py`)

### Purpose
Orchestrates all test suites and provides comprehensive system validation and reporting.

### Core Classes

#### `TestResults`
Tracks and aggregates test results:

- **Result Aggregation**
  - Success/failure counting
  - Timing information
  - Error categorization

- **Report Generation**
  - Detailed test reports
  - Summary statistics
  - Failure analysis

#### `IntegrationTestRunner`
Main test orchestration:

- **Test Environment Setup**
  - Workspace creation
  - Fixture generation
  - Environment configuration

- **Test Suite Execution**
  - Parallel execution management
  - Sequential fallback
  - Progress reporting

- **System Validation**
  - Server startup validation
  - Tool registration verification
  - End-to-end flow testing

### System Integration Validations

1. **Server Startup Test**
   - FastAPI application creation
   - Route registration
   - Dependency injection

2. **Tool Registration Test**
   - Available tool enumeration
   - Tool schema validation
   - Function mapping verification

3. **API Endpoint Test**
   - Health check endpoints
   - Chat completion endpoints
   - Error handling endpoints

4. **Tool Execution Flow Test**
   - Complete request processing
   - Tool execution pipeline
   - Response generation

5. **Error Handling Test**
   - Error propagation
   - Recovery mechanisms
   - Graceful degradation

## Running Tests

### Command Line Interface

#### Basic Usage
```bash
# Run all tests with default settings
./tests/run_integration_tests.py

# Run tests sequentially (for debugging)
./tests/run_integration_tests.py --sequential

# Run only system validation
./tests/run_integration_tests.py --validate-only
```

#### Advanced Options
```bash
# Generate coverage report
./tests/run_integration_tests.py --coverage

# Keep test workspace for inspection
./tests/run_integration_tests.py --no-cleanup

# Custom test directory
./tests/run_integration_tests.py --test-dir /custom/path

# Combine options
./tests/run_integration_tests.py --coverage --sequential --no-cleanup
```

#### Individual Test Suites
```bash
# Run specific test suite
pytest tests/test_tool_execution.py -v

# Run specific test class
pytest tests/test_tool_execution.py::TestLLMResponseParsing -v

# Run specific test case
pytest tests/test_tool_execution.py::TestLLMResponseParsing::test_json_format -v

# Run with debugging
pytest tests/test_tool_execution.py::TestClass::test_method --pdb -s
```

### Environment Setup

#### Required Dependencies
```bash
# Core testing dependencies
pip install pytest>=7.0.0
pip install pytest-json-report>=1.5.0

# Optional dependencies
pip install coverage>=6.0.0  # For coverage reports
pip install pytest-xdist     # For parallel test execution
pip install pytest-html     # For HTML reports
```

#### Environment Variables
```bash
# Test mode configuration
export MLX_RAG_TEST_MODE=1

# Custom workspace directory (optional)
export MLX_RAG_TEST_WORKSPACE=/tmp/custom_workspace

# Debug mode (optional)
export MLX_RAG_DEBUG=1
```

## Test Reports and Analysis

### Console Output
Real-time progress reporting during test execution:

```
INFO - Setting up test environment...
INFO - Test workspace created at: /tmp/mlx_rag_test_abc123
INFO - Running test suite: tool_execution
✅ tool_execution completed in 12.34s
INFO - Running test suite: openai_compatibility
✅ openai_compatibility completed in 8.76s
INFO - Running test suite: security_sandboxing
✅ security_sandboxing completed in 15.23s
INFO - Running test suite: error_handling
✅ error_handling completed in 9.87s
```

### Detailed Test Report
Generated as `integration_test_report.txt`:

```
================================================================================
MLX-RAG TOOL CALLING INTEGRATION TEST REPORT
================================================================================
Total Duration: 46.20 seconds
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

✅ PASS openai_compatibility (8.76s)
  Tests: 28 | Passed: 28 | Failed: 0 | Skipped: 0

❌ FAIL security_sandboxing (15.23s)
  Tests: 35 | Passed: 33 | Failed: 2 | Skipped: 0
  Failures:
    - FAILED test_unicode_handling - UnicodeEncodeError in filename
    - FAILED test_memory_limits - Memory limit not properly enforced

✅ PASS error_handling (9.87s)
  Tests: 32 | Passed: 32 | Failed: 0 | Skipped: 0

SYSTEM INTEGRATION VALIDATION:
==================================================
✅ PASS server_startup
✅ PASS tool_registration
✅ PASS api_endpoints
✅ PASS tool_execution_flow
✅ PASS error_handling

================================================================================
❌ SOME TESTS FAILED
================================================================================
```

### Coverage Reports
When `--coverage` flag is used:

```
Name                              Stmts   Miss  Cover   Missing
---------------------------------------------------------------
mlx_rag/__init__.py                   0      0   100%
mlx_rag/server.py                   245     12    95%   123-125, 178-180
mlx_rag/tools.py                    156      8    95%   89-92, 145-148
mlx_rag/tool_prompts.py              89      5    94%   67-69, 82-84
---------------------------------------------------------------
TOTAL                               490     25    95%
```

### JSON Reports
Individual test suite reports (e.g., `tool_execution_report.json`):

```json
{
  "summary": {
    "total": 32,
    "passed": 32,
    "failed": 0,
    "skipped": 0,
    "error": 0
  },
  "tests": [
    {
      "name": "test_json_tool_call_parsing",
      "outcome": "passed",
      "duration": 0.123,
      "setup": 0.001,
      "call": 0.122,
      "teardown": 0.000
    }
  ],
  "warnings": [],
  "duration": 12.34
}
```

## Extending Tests

### Adding New Test Cases

#### 1. Extend Existing Test Classes
```python
class TestToolExecution:
    def test_new_functionality(self):
        """Test description."""
        # Setup
        self.create_test_file("test.txt", "content")
        
        # Execute
        result = execute_tool("new_tool", {"param": "value"})
        
        # Verify
        assert result == expected_result
        assert self.file_exists("output.txt")
```

#### 2. Create New Test Classes
```python
class TestNewFeature:
    """Test new feature functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_feature_behavior(self):
        """Test specific feature behavior."""
        # Test implementation
        pass
```

### Adding New Test Suites

#### 1. Create Test File
Create `tests/test_new_suite.py` following the established patterns.

#### 2. Update Integration Runner
```python
# In run_integration_tests.py
suite_files = {
    "tool_execution": "test_tool_execution.py",
    "openai_compatibility": "test_openai_compatibility.py",
    "security_sandboxing": "test_security_sandboxing.py",
    "error_handling": "test_error_handling.py",
    "new_suite": "test_new_suite.py",  # Add new suite
}
```

#### 3. Update Test Lists
```python
suites = [
    "tool_execution",
    "openai_compatibility",
    "security_sandboxing", 
    "error_handling",
    "new_suite"  # Add to execution list
]
```

### Test Data and Fixtures

#### Creating Test Fixtures
```python
def create_test_workspace(self):
    """Create test workspace with fixtures."""
    # Create directory structure
    os.makedirs(self.temp_dir / "src")
    os.makedirs(self.temp_dir / "docs")
    
    # Create test files
    (self.temp_dir / "src" / "main.py").write_text(
        "def main():\n    print('Hello, World!')\n"
    )
    
    (self.temp_dir / "README.md").write_text(
        "# Test Project\nThis is a test project.\n"
    )
```

#### Using Parameterized Tests
```python
import pytest

@pytest.mark.parametrize("input,expected", [
    ("valid_input", "expected_output"),
    ("edge_case", "edge_output"),
    ("error_case", None),
])
def test_parameterized_behavior(self, input, expected):
    """Test with multiple parameter sets."""
    result = function_under_test(input)
    assert result == expected
```

## Troubleshooting

### Common Issues

#### Test Environment Setup Failures
```bash
# Check Python environment
python --version  # Should be 3.8+

# Check required packages
pip list | grep pytest
pip list | grep fastapi

# Install missing dependencies
pip install -r requirements-test.txt
```

#### Permission Errors
```bash
# Make integration script executable
chmod +x tests/run_integration_tests.py

# Check file permissions in test workspace
ls -la /tmp/mlx_rag_test_*
```

#### Memory or Resource Issues
```bash
# Run tests sequentially to reduce memory usage
./tests/run_integration_tests.py --sequential

# Clear system temporary files
rm -rf /tmp/mlx_rag_test_*

# Monitor resource usage
top -p $(pgrep python)
```

### Debugging Failed Tests

#### Running Individual Tests
```bash
# Run with maximum verbosity
pytest tests/test_tool_execution.py::TestClass::test_method -vvv

# Run with stdout/stderr capture disabled
pytest tests/test_tool_execution.py::TestClass::test_method -s

# Run with Python debugger
pytest tests/test_tool_execution.py::TestClass::test_method --pdb
```

#### Inspecting Test Workspace
```bash
# Keep workspace after test failure
./tests/run_integration_tests.py --no-cleanup

# Inspect workspace contents
ls -la /tmp/mlx_rag_test_*
cat /tmp/mlx_rag_test_*/fixtures/test_file.txt
```

#### Analyzing Log Output
```bash
# Check detailed logs
tail -f integration_test.log

# Filter specific test logs
grep "test_specific_case" integration_test.log

# Check error logs
grep -i "error\|fail" integration_test.log
```

### Performance Issues

#### Slow Test Execution
```bash
# Profile test execution
python -m cProfile -o profile.stats tests/run_integration_tests.py

# Analyze profile results
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('time').print_stats(20)"
```

#### Memory Usage
```bash
# Monitor memory usage during tests
watch -n 1 'ps aux | grep pytest | grep -v grep'

# Use memory profiler
pip install memory-profiler
python -m memory_profiler tests/run_integration_tests.py
```

### CI/CD Integration Issues

#### Exit Code Problems
```bash
# Test exit codes locally
./tests/run_integration_tests.py
echo $?  # Should be 0 for success, 1 for failure
```

#### Report Generation Issues
```bash
# Ensure pytest-json-report is installed
pip install pytest-json-report

# Verify JSON report generation
ls -la *_report.json
cat tool_execution_report.json | jq .summary
```

### Getting Help

#### Log Analysis
Check the following log files for detailed information:
- `integration_test.log` - Main execution log
- `*_report.json` - Individual suite reports
- `integration_test_report.txt` - Summary report

#### Community Resources
- Check existing issues and solutions
- Review test patterns in similar projects
- Consult FastAPI and pytest documentation

#### Debugging Checklist
1. ✅ Python version compatibility (3.8+)
2. ✅ All dependencies installed
3. ✅ File permissions correct
4. ✅ Sufficient disk space and memory
5. ✅ No conflicting processes
6. ✅ Environment variables set correctly
7. ✅ Test workspace accessible
8. ✅ Network connectivity (if required)

This comprehensive test documentation provides everything needed to understand, use, and extend the MLX-RAG tool calling test suite. The tests ensure robust, secure, and compatible tool calling functionality across all scenarios and edge cases.

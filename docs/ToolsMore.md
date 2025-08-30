# Chat Integration

New Tool Call Detection and Parsing Utilities

1. _parse_tool_calls_from_text(text: str) Function
This function detects and parses tool calls from LLM responses in multiple formats:

•  JSON-style function calls: {"function": "tool_name", "arguments": {...}}
•  XML-style tool calls: <tool_call function="tool_name" args='{"key": "value"}'/>  
•  Function call patterns: tool_name({"key": "value"})

Key features:
•  Uses regex patterns to detect different tool call formats
•  Validates JSON arguments before creating tool calls
•  Generates unique IDs for each tool call
•  Removes duplicate tool calls
•  Only processes known tool names to avoid false positives
•  Comprehensive logging for debugging

2. _create_system_prompt_with_tools(tools: List[Dict[str, Any]]) Function
Creates system prompts that instruct LLMs on how to use available tools:

•  Generates detailed tool descriptions with parameters
•  Includes parameter types, descriptions, and requirement status
•  Provides clear JSON format instructions for tool usage
•  Returns a helpful default prompt when no tools are provided

Usage in Chat Completions Endpoint

These utilities integrate seamlessly with your existing /v1/chat/completions endpoint. You can now:

1. Detect tool calls in responses: The _parse_tool_calls_from_text() function can be called on LLM responses to extract structured tool calls
2. Add tool support: Use the _create_system_prompt_with_tools() function to modify system prompts when tools are provided in requests
3. Handle tool choice: The existing Pydantic models already support tools and tool_choice parameters

Example Integration

Here's how you could use these utilities in your chat completions endpoint:

```python
# If tools are provided in the request
if request.tools:
    # Create enhanced system prompt
    system_prompt = _create_system_prompt_with_tools(request.tools)
    # Add or modify system message in messages
    
# After getting LLM response
response_text = "LLM generated response..."
detected_tool_calls = _parse_tool_calls_from_text(response_text)

if detected_tool_calls:
    # Add tool_calls to the response message
    response_message.tool_calls = detected_tool_calls
```

# System Prompts

1. Comprehensive Tool Prompt Module (tool_prompts.py)
•  Base System Prompts: Foundational templates for AI assistant behavior
•  Detailed Tool Templates: Specific guidance for each tool (list_directory, read_file, search_files, write_file, edit_file)
•  Usage Guidelines: 7 key principles for effective tool usage
•  Common Workflows: Pre-defined strategies for different task types:
•  Explore New Codebase
•  Debug Issues  
•  Add New Features
•  Code Refactoring
•  Error Handling Templates: Guidance for common failure scenarios

2. Enhanced Server Integration
•  Updated _create_system_prompt_with_tools(): Now uses the comprehensive prompt system
•  Contextual Prompt Generation: Analyzes user queries to provide relevant guidance
•  Tool Call Validation: Built-in format checking and error reporting

3. New API Endpoints for Testing & Demonstration

#### GET /v1/tools/prompt/demo
Demonstrates the tool prompt system capabilities:

```bash
# Basic demo
curl "http://localhost:8080/v1/tools/prompt/demo"

# Contextual demo
curl "http://localhost:8080/v1/tools/prompt/demo?user_query=I want to debug a Python error"
```

#### GET /v1/tools/prompt/generate
Generate custom tool prompts:

```bash
# Generate with custom tools
curl "http://localhost:8080/v1/tools/prompt/generate" \
  -G --data-urlencode 'tools_json=[{"type":"function","function":{"name":"custom_tool","description":"My tool"}}]'
```

4. Key Features

#### 🎯 Contextual Prompts
The system analyzes user queries and provides contextually relevant guidance:
•  Debug queries → Debug workflow + error handling tips
•  Exploration queries → Codebase exploration strategies  
•  Feature requests → Development best practices

#### 📚 Comprehensive Tool Documentation
Each tool includes:
•  When to use: Clear use case scenarios
•  Best practices: Expert guidance for effective usage
•  Examples: Real-world usage patterns with explanations
•  Parameter details: Complete API documentation

#### 🔄 Common Workflows 
Pre-built strategies for:
•  First-time codebase exploration
•  Systematic debugging approaches
•  Feature development workflows 
•  Code refactoring methodologies

#### 🛠️ Error Recovery
Built-in guidance for handling:
•  File not found errors
•  Permission issues
•  Search failures
•  Edit conflicts

5. Usage in Chat Completions

The enhanced system now automatically:
1. Detects tools in chat completion requests
2. Analyzes user queries for context
3. Generates appropriate prompts with relevant guidance
4. Provides structured tool calling instructions

6. Example Generated Prompt

When tools are provided, the system generates prompts like:

```
You are a helpful AI assistant with access to tools for exploring and modifying files within a code repository or documentation collection.

## Available Tools

### list_directory
**Description**: List files and directories in a specified path
**When to use**:
- When first exploring a new codebase
- To understand project structure and organization
**Best practices**:
- Start with the root directory to get overall structure
- Use recursive=true for deep exploration
**Examples**:
- First time exploring a project:
  {"function": "list_directory", "arguments": {"path": "."}}

[... comprehensive tool documentation continues ...]

## Tool Usage Guidelines
1. Always think before acting
2. Start broad, then narrow
3. Be systematic in exploration
[... continues with detailed guidance ...]
```

# Testing

1. ✅ Tool execution tests - test_tool_execution.py
2. ✅ OpenAI API compatibility tests - test_openai_compatibility.py
3. ✅ Security and sandboxing tests - test_security_sandboxing.py
4. ✅ Error handling tests - test_error_handling.py
5. ✅ Integration test script - run_integration_tests.py

📁 Delivered Test Files:

1. tests/test_tool_execution.py (456 lines)
•  LLM response parsing (JSON, XML, function-like formats)
•  Tool call validation and execution with mock workspace
•  Realistic workflow scenarios (code exploration, debugging)
•  Integration tests with FastAPI TestClient
•  Mixed format response handling

2. tests/test_openai_compatibility.py (520 lines)  
•  Complete OpenAI API format validation
•  Request/response structure compliance
•  Streaming response testing
•  Tool choice parameter variations
•  Conversation flow validation
•  Parameter validation and error handling

3. tests/test_security_sandboxing.py (640 lines)
•  Path traversal attack prevention (basic, encoded, symlinks)
•  Execution limits and resource management
•  File permission security testing
•  Input sanitization (filenames, content, queries)
•  Workspace isolation and cleanup
•  Security error handling consistency

4. tests/test_error_handling.py (680 lines)
•  Tool execution errors (permissions, file not found, corruption)
•  LLM response parsing errors and malformed inputs
•  Recovery scenarios and partial failure handling
•  Edge cases (empty files, unicode, long filenames)
•  Concurrent access and memory pressure testing
•  API-level error handling

5. tests/run_integration_tests.py (600+ lines, executable)
•  Comprehensive integration test orchestrator
•  Parallel and sequential execution modes
•  Detailed reporting with timing and statistics
•  System integration validation
•  Coverage report generation
•  Advanced debugging and cleanup options

6. tests/README.md (200+ lines)
•  Complete documentation of the test suite
•  Usage instructions and examples
•  Test coverage overview
•  Debugging guides and CI integration

🔧 Key Features Implemented:

Testing Infrastructure:
•  Isolated temporary workspaces for each test
•  Mock file system operations with security controls
•  FastAPI TestClient integration
•  Parallel test execution with thread safety
•  Comprehensive reporting and logging

Security Testing:
•  Path traversal protection (basic, URL-encoded, symlinks)
•  File size and execution limits
•  Input sanitization and validation
•  Resource management and cleanup
•  Permission-based access control

Compatibility Testing:
•  Full OpenAI API format compliance
•  Request/response structure validation
•  Streaming response handling
•  Tool choice parameter variations
•  Conversation flow testing

Error Handling:
•  Graceful error recovery
•  Malformed input handling  
•  Concurrent access scenarios
•  Edge case and boundary testing
•  Resource exhaustion handling

Integration & Automation:
•  End-to-end system validation
•  Automated test suite execution
•  CI/CD ready with proper exit codes
•  Coverage report generation
•  Detailed failure reporting

🚀 Ready to Use:

The test suite is immediately usable:

```bash
# Run all tests
./tests/run_integration_tests.py

# Run with coverage
./tests/run_integration_tests.py --coverage

# Individual test suites  
pytest tests/test_tool_execution.py -v
```


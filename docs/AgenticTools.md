# Agentic Tools Implementation

MLX-RAG now includes a comprehensive agentic tools system that enables Large Language Models (LLMs) to perform file-system operations within RAG collection source directories. This implementation provides OpenAI-compatible tool calling functionality with robust security and sandboxing.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Available Tools](#available-tools)
4. [Security and Sandboxing](#security-and-sandboxing)
5. [Integration](#integration)
6. [API Reference](#api-reference)
7. [Usage Examples](#usage-examples)
8. [Implementation Details](#implementation-details)
9. [Next Steps](#next-steps)

## Overview

The agentic tools system allows LLMs to:
- Explore file and directory structures within RAG collections
- Read and search through code and documentation files
- Create new files and edit existing ones
- Perform intelligent code analysis and modifications
- Maintain strict security boundaries within RAG collection directories

### Key Features

- **OpenAI-Compatible**: Fully compatible with OpenAI's function calling API
- **Secure Sandboxing**: All operations are restricted to RAG collection directories
- **Concurrent Execution**: Multiple tools can run simultaneously for better performance
- **Comprehensive Error Handling**: Detailed error reporting and recovery
- **Type Safety**: Full type annotations and Pydantic model validation
- **Extensible Architecture**: Easy to add new tools and capabilities

## Architecture

The implementation consists of three main components:

### 1. Tool Definitions (`agentic_tools.py`)

```
AgenticTool (Abstract Base Class)
├── ToolInputValidationError
├── ToolExecutionError
├── ToolSecurityError
└── Concrete Tools:
    ├── ListDirectoryTool
    ├── ReadFileTool
    ├── SearchFilesTool
    ├── WriteFileTool
    └── EditFileTool
```

### 2. Tool Execution Framework (`tool_executor.py`)

```
ToolExecutor
├── ToolExecutionResult
├── ToolExecutionError
└── Global Functions:
    ├── get_tool_executor()
    └── clear_tool_executor()
```

### 3. Server Integration (`server.py`)

```
OpenAI-Compatible Models:
├── Tool
├── ToolCall  
├── ToolChoice
└── Extended ChatCompletionMessage
```

## Available Tools

### 1. ListDirectoryTool

**Purpose**: List files and directories within the RAG collection

**Parameters**:
- `path` (optional): Relative path to list (defaults to root)
- `show_hidden` (optional): Include hidden files (default: false)
- `recursive` (optional): Recursively list subdirectories (default: false)

**Returns**: Array of file/directory information with names, types, sizes, and modification times

**Example**:
```json
{
  "name": "list_directory",
  "arguments": {
    "path": "src",
    "recursive": true
  }
}
```

### 2. ReadFileTool

**Purpose**: Read the contents of a file with automatic encoding detection

**Parameters**:
- `file_path` (required): Relative path to the file to read
- `encoding` (optional): Specific encoding to use (auto-detected if not provided)

**Returns**: File contents as a string with metadata about encoding and size

**Security**: 
- Validates file exists and is readable
- Prevents reading of binary files over 10MB
- Supports common text encodings (UTF-8, UTF-16, ISO-8859-1)

**Example**:
```json
{
  "name": "read_file",
  "arguments": {
    "file_path": "src/main.py"
  }
}
```

### 3. SearchFilesTool

**Purpose**: Search for patterns within files using regular expressions

**Parameters**:
- `pattern` (required): Regular expression pattern to search for
- `file_pattern` (optional): Glob pattern for files to search (default: "*")
- `max_results` (optional): Maximum number of results to return (default: 100)
- `case_sensitive` (optional): Whether search is case-sensitive (default: false)

**Returns**: Array of search results with file paths, line numbers, and matched content

**Example**:
```json
{
  "name": "search_files",
  "arguments": {
    "pattern": "class\\s+\\w+\\s*\\(",
    "file_pattern": "*.py",
    "max_results": 50
  }
}
```

### 4. WriteFileTool

**Purpose**: Create new files with specified content

**Parameters**:
- `file_path` (required): Relative path for the new file
- `content` (required): Content to write to the file
- `encoding` (optional): Text encoding to use (default: UTF-8)
- `create_directories` (optional): Create parent directories if needed (default: true)

**Returns**: Confirmation with file path, size, and encoding information

**Security**:
- Prevents overwriting existing files
- Validates file path and content
- Creates parent directories safely
- Limits file size to prevent abuse

**Example**:
```json
{
  "name": "write_file",
  "arguments": {
    "file_path": "docs/new_feature.md",
    "content": "# New Feature\n\nDescription of the new feature...",
    "create_directories": true
  }
}
```

### 5. EditFileTool

**Purpose**: Edit existing files using line-based operations

**Parameters**:
- `file_path` (required): Relative path to the file to edit
- `operations` (required): Array of edit operations to perform

**Edit Operations**:
- `insert`: Insert lines at a specific position
- `replace`: Replace a range of lines with new content
- `delete`: Delete a range of lines

**Returns**: Summary of changes made with before/after line counts

**Example**:
```json
{
  "name": "edit_file",
  "arguments": {
    "file_path": "src/config.py",
    "operations": [
      {
        "type": "replace",
        "start_line": 10,
        "end_line": 12,
        "content": ["# Updated configuration", "DEBUG = False"]
      }
    ]
  }
}
```

## Security and Sandboxing

### Path Validation

All tools implement comprehensive path validation:

```python
def _validate_path(self, path: str) -> str:
    """Validate and resolve a path within the collection."""
    # Convert to absolute path within collection
    collection_path = Path(self.collection_path)
    full_path = collection_path / path
    
    # Resolve path and check bounds
    resolved_path = full_path.resolve()
    
    # Ensure path is within collection directory
    if not str(resolved_path).startswith(str(collection_path.resolve())):
        raise ToolSecurityError(f"Path outside collection: {path}")
    
    return str(resolved_path)
```

### File Type Restrictions

- Binary files are handled carefully (size limits, type detection)
- Hidden files require explicit permission
- Temporary files and common build artifacts are filtered
- Maximum file sizes enforced to prevent memory issues

### Input Sanitization

- All file paths are normalized and validated
- Regular expressions are compiled and validated before use
- File contents are validated for encoding and size
- Operation parameters are strictly typed and validated

## Integration

### Tool Executor Integration

The `ToolExecutor` class manages tool lifecycle and execution:

```python
# Initialize with RAG collection path
executor = get_tool_executor("/path/to/rag/collection/source")

# Check available tools
if executor.has_available_tools():
    tools = executor.get_tools_for_openai_request()
    
# Execute tool calls
results = await executor.execute_tool_calls(tool_calls)
```

### Server Integration

Tools are automatically available when a RAG collection is active:

```python
# Get active RAG collection
active_collection = db.query(RAGCollection).filter(
    RAGCollection.is_active == True
).first()

# Initialize tool executor with collection path
if active_collection:
    tool_executor = get_tool_executor(active_collection.path)
```

### OpenAI API Compatibility

Tools are exposed through standard OpenAI chat completion parameters:

```json
{
  "model": "llama-3-8b",
  "messages": [...],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read the contents of a file",
        "parameters": {
          "type": "object",
          "properties": {
            "file_path": {
              "type": "string",
              "description": "Relative path to the file to read"
            }
          },
          "required": ["file_path"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

## API Reference

### ToolExecutor Class

#### Methods

- `get_available_tools() -> Dict[str, Dict[str, Any]]`
  - Returns OpenAI-compatible tool definitions

- `get_tools_for_openai_request() -> List[Dict[str, Any]]`
  - Returns tools formatted for OpenAI API requests

- `execute_tool_call(tool_call_id: str, function_name: str, arguments: Dict[str, Any]) -> ToolExecutionResult`
  - Execute a single tool call

- `execute_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[ToolExecutionResult]`
  - Execute multiple tool calls concurrently

- `has_available_tools() -> bool`
  - Check if any tools are available

- `update_collection_path(new_path: str)`
  - Update the RAG collection path and reinitialize tools

#### ToolExecutionResult

```python
class ToolExecutionResult:
    tool_call_id: str
    function_name: str
    success: bool
    result: Any
    error: str
    execution_time_ms: float
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI tool call result format"""
```

### AgenticTool Base Class

#### Abstract Methods

- `get_name() -> str` - Tool name identifier
- `get_description() -> str` - Human-readable description
- `get_parameters() -> Dict[str, Any]` - JSON Schema for parameters
- `execute_async(arguments: Dict[str, Any]) -> Any` - Execute tool logic

#### Helper Methods

- `get_openai_function_definition() -> Dict[str, Any]` - OpenAI-compatible definition
- `validate_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]` - Validate inputs

## Usage Examples

### Example 1: Reading and Analyzing Code

```python
# LLM requests to explore a codebase
tool_calls = [
    {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "list_directory",
            "arguments": {"path": "src", "recursive": true}
        }
    },
    {
        "id": "call_2", 
        "type": "function",
        "function": {
            "name": "search_files",
            "arguments": {
                "pattern": "def\\s+main\\s*\\(",
                "file_pattern": "*.py"
            }
        }
    }
]

# Execute tools
executor = get_tool_executor("/path/to/project")
results = await executor.execute_tool_calls(tool_calls)

# Results contain file listings and search matches
for result in results:
    if result.success:
        print(f"Tool {result.function_name}: {result.result}")
```

### Example 2: Creating Documentation

```python
# LLM generates documentation based on code analysis
tool_call = {
    "id": "call_1",
    "type": "function", 
    "function": {
        "name": "write_file",
        "arguments": {
            "file_path": "docs/api_reference.md",
            "content": "# API Reference\n\n## Overview\n...",
            "create_directories": true
        }
    }
}

result = await executor.execute_tool_call(
    tool_call["id"],
    tool_call["function"]["name"], 
    tool_call["function"]["arguments"]
)
```

### Example 3: Refactoring Code

```python
# LLM modifies code files
edit_operation = {
    "id": "call_1",
    "type": "function",
    "function": {
        "name": "edit_file", 
        "arguments": {
            "file_path": "src/utils.py",
            "operations": [
                {
                    "type": "insert",
                    "line": 1,
                    "content": ["import logging", "logger = logging.getLogger(__name__)"]
                },
                {
                    "type": "replace",
                    "start_line": 15,
                    "end_line": 17,
                    "content": ["    logger.info(f'Processing {item}')", "    return process_item(item)"]
                }
            ]
        }
    }
}
```

## Implementation Details

### Error Handling

The system implements three levels of error handling:

1. **Input Validation Errors**: Invalid parameters or malformed requests
2. **Security Errors**: Attempts to access files outside the sandbox
3. **Execution Errors**: File system operations that fail

```python
try:
    result = await tool.execute_async(arguments)
except ToolInputValidationError as e:
    # Invalid input parameters
    return ToolExecutionResult(success=False, error=f"Invalid input: {e}")
except ToolSecurityError as e:
    # Security violation
    return ToolExecutionResult(success=False, error=f"Security error: {e}")
except ToolExecutionError as e:
    # Execution failure
    return ToolExecutionResult(success=False, error=f"Execution error: {e}")
```

### Concurrent Execution

Tools support concurrent execution for better performance:

```python
# Create tasks for concurrent execution
tasks = []
for tool_call in tool_calls:
    task = executor.execute_tool_call(
        tool_call_id=tool_call["id"],
        function_name=tool_call["function"]["name"],
        arguments=tool_call["function"]["arguments"]
    )
    tasks.append(task)

# Execute all tasks concurrently  
results = await asyncio.gather(*tasks, return_exceptions=True)
```

### Memory Management

- File size limits prevent memory exhaustion
- Large directory listings are paginated
- Search results are limited and streamed
- Temporary files are cleaned up automatically

### Logging and Monitoring

Comprehensive logging is implemented throughout:

```python
logger.info(f"Executing tool call {tool_call_id}: {function_name}")
logger.debug(f"Tool arguments: {arguments}")
logger.info(f"Tool execution completed in {execution_time:.1f}ms")
```

## Testing and Validation

### Unit Tests

Each tool should have comprehensive unit tests covering:

- Valid input scenarios
- Invalid input handling
- Security boundary enforcement
- Error conditions and recovery
- Performance characteristics

### Integration Tests

Full end-to-end testing should cover:

- Tool executor initialization
- Multi-tool execution scenarios
- RAG collection integration
- OpenAI API compatibility
- Error propagation and handling

### Security Testing

Security testing should verify:

- Path traversal prevention
- File access restrictions
- Input sanitization effectiveness
- Resource usage limits
- Error information disclosure

## Next Steps

### Immediate Tasks (Required for Basic Functionality)

- [ ] **Complete Chat Completions Integration**
  - [ ] Modify `/v1/chat/completions` endpoint to detect tool calls in LLM responses
  - [ ] Implement tool execution flow in chat completions
  - [ ] Handle tool results and feed them back into the conversation
  - [ ] Support tool calling in both streaming and non-streaming modes

- [ ] **Add System Prompts and Instructions**
  - [ ] Create system prompt templates that teach LLMs when to use tools
  - [ ] Document tool capabilities and parameters for LLM understanding
  - [ ] Implement tool usage guidelines and best practices
  - [ ] Add examples of effective tool calling patterns

- [ ] **Basic Testing and Validation**
  - [ ] Test tool execution with real LLM responses
  - [ ] Validate OpenAI API compatibility
  - [ ] Test security boundaries and sandboxing
  - [ ] Verify error handling and recovery

### Enhanced Features (Future Development)

- [ ] **Advanced Tool Capabilities**
  - [ ] Add file comparison and diff tools
  - [ ] Implement code analysis tools (syntax checking, linting)
  - [ ] Add git integration tools (status, diff, commit)
  - [ ] Create project structure analysis tools

- [ ] **Performance Optimizations**
  - [ ] Implement caching for frequently accessed files
  - [ ] Add batch operations for multiple file modifications
  - [ ] Optimize search performance with indexing
  - [ ] Implement streaming for large file operations

- [ ] **Enhanced Security**
  - [ ] Add configurable security policies
  - [ ] Implement audit logging for all tool operations
  - [ ] Add rate limiting for tool execution
  - [ ] Create whitelist/blacklist file patterns

- [ ] **User Experience Improvements**
  - [ ] Add tool execution progress indicators
  - [ ] Implement undo/redo functionality for file modifications
  - [ ] Create tool execution history and replay
  - [ ] Add visual diff display for file changes

- [ ] **Integration Enhancements**
  - [ ] Add webhook notifications for tool execution
  - [ ] Implement tool execution metrics and analytics
  - [ ] Create tool execution APIs for external integrations
  - [ ] Add support for custom tool plugins

- [ ] **Documentation and Testing**
  - [ ] Create comprehensive unit test suite
  - [ ] Add integration tests for all tool combinations
  - [ ] Implement performance benchmarking
  - [ ] Create user documentation and tutorials
  - [ ] Add API documentation with interactive examples

### Long-term Vision

- [ ] **Multi-Collection Support**
  - [ ] Enable tools to work across multiple RAG collections
  - [ ] Implement collection-specific tool permissions
  - [ ] Add cross-collection file operations

- [ ] **Advanced AI Integration**
  - [ ] Implement tool usage learning and optimization
  - [ ] Add intelligent tool suggestion based on context
  - [ ] Create tool execution planning and orchestration
  - [ ] Implement collaborative multi-agent tool usage

---

*This implementation provides a solid foundation for agentic AI capabilities in MLX-RAG, enabling LLMs to intelligently interact with codebases and documentation within secure, well-defined boundaries.*

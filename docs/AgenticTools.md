# Agentic Tools Implementation

MLX-RAG now includes a comprehensive agentic tools system with **dual implementations** that enables Large Language Models (LLMs) to perform file-system operations within RAG collection source directories. The system offers both an **original implementation** and a **LangChain-integrated implementation**, providing OpenAI-compatible tool calling functionality with robust security and sandboxing.

## 🆕 LangChain Integration

**NEW**: MLX-RAG now supports LangChain's powerful agent framework alongside the original tool system, providing enhanced capabilities including:

- **Advanced Agent Capabilities**: Built-in ReAct agents, conversation memory, and sophisticated reasoning patterns
- **Ecosystem Integration**: Access to LangChain's extensive tool and integration ecosystem 
- **Enhanced Prompt Management**: LangChain's prompt templates and contextual prompt generation
- **Seamless Migration Path**: Both systems run side-by-side, allowing gradual migration
- **Maintained Compatibility**: Full OpenAI API compatibility preserved in both systems

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

MLX-RAG now offers **two complementary tool system implementations**:

### 🔄 Dual System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MLX-RAG Tool Systems                     │
├─────────────────────┬───────────────────────────────────────┤
│   Original System   │         LangChain System             │
├─────────────────────┼───────────────────────────────────────┤
│ • Direct execution  │ • Agent-based execution              │
│ • Simple tool calls │ • ReAct agents                       │
│ • Basic prompts     │ • Memory management                  │
│ • Fast & lightweight│ • Contextual reasoning               │
│                     │ • Ecosystem integration             │
└─────────────────────┴───────────────────────────────────────┘
              ↓                          ↓
        ┌──────────────┐         ┌──────────────┐
        │   Same File  │    ←→   │   Same File  │
        │  Operations  │         │  Operations  │
        └──────────────┘         └──────────────┘
              ↓                          ↓
        ┌────────────────────────────────────────┐
        │      Unified Security Sandbox          │
        │   • Path validation                    │
        │   • File type restrictions             │
        │   • Resource limits                    │
        └────────────────────────────────────────┘
```

### 🎯 When to Use Each System

**Use Original System when:**
- Simple, direct tool execution is needed
- Minimal dependencies are preferred
- Fast response times are critical
- Custom tool integration is required

**Use LangChain System when:**
- Complex reasoning patterns are needed
- Memory and context management are important
- Integration with LangChain ecosystem is desired
- Advanced agent capabilities are required

The implementation consists of the following main components:

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

## 🦜 LangChain System Deep Dive

### LangChain Architecture Components

The LangChain-integrated system provides advanced capabilities through several key components:

#### 1. LangChain Tool Wrappers

Each file system tool is wrapped as a LangChain `BaseTool` while maintaining full compatibility:

```python
class LangChainListDirectoryTool(BaseTool):
    name: str = "list_directory"
    description: str = "List files and directories in a specified path"
    
    class InputSchema(BaseModel):
        path: str = Field(default=".", description="Directory path to list")
        recursive: bool = Field(default=False, description="List recursively")
        pattern: Optional[str] = Field(None, description="File pattern filter")
    
    def _run(self, path: str = ".", recursive: bool = False, pattern: str = None) -> str:
        # Delegates to original AgenticTool implementation
        return self.original_tool.execute_sync({"path": path, "recursive": recursive, "pattern": pattern})
    
    async def _arun(self, path: str = ".", recursive: bool = False, pattern: str = None) -> str:
        # Async execution via original tool
        return await self.original_tool.execute_async({"path": path, "recursive": recursive, "pattern": pattern})
```

#### 2. LangChain Tool Executor

The `LangChainToolExecutor` provides enhanced tool management with agent capabilities:

```python
class LangChainToolExecutor:
    def __init__(self, collection_path: str):
        self.collection_path = collection_path
        self.tools = self._initialize_langchain_tools()
        self.llm = None  # Optional LLM for agent mode
        self.memory = None  # Optional conversation memory
    
    def create_react_agent(self, llm, memory=None) -> AgentExecutor:
        """Create a ReAct agent with tools and optional memory"""
        
    async def execute_single_tool_call_async(self, tool_call) -> ToolExecutionResult:
        """Execute a single tool call with enhanced error handling"""
        
    async def execute_multiple_tool_calls_async(self, tool_calls) -> List[ToolExecutionResult]:
        """Execute multiple tool calls concurrently"""
```

#### 3. Enhanced Prompt Management

LangChain integration includes sophisticated prompt templates:

```python
# Contextual prompt generation based on user query and conversation history
def generate_contextual_prompt(tools, user_query, conversation_history=None):
    """Generate context-aware system prompts for optimal tool usage"""
    
# ReAct agent prompt templates
def create_react_prompt_template(tools):
    """Create ReAct-style reasoning prompt templates"""
    
# Tool usage workflow templates  
def create_file_system_agent_prompt(tools):
    """Create specialized prompts for file system operations"""
```

### LangChain-Specific Features

#### 1. ReAct Agent Integration

```python
# Create a ReAct agent for sophisticated reasoning
langchain_executor = get_langchain_tool_executor("/path/to/collection")
agent = langchain_executor.create_react_agent(
    llm=ChatMLX(model="llama-3-8b-instruct"),
    memory=ConversationBufferMemory()
)

# Execute with reasoning capability
result = await agent.arun(
    "Analyze the project structure and create a summary document"
)
```

#### 2. Memory and Context Management

```python
# Conversation memory for multi-turn interactions
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    k=10,  # Remember last 10 exchanges
    return_messages=True
)

# Agent with persistent memory
agent = langchain_executor.create_react_agent(llm, memory=memory)

# Multi-turn conversation with context retention
result1 = await agent.arun("List the Python files in the project")
result2 = await agent.arun("Now read the main.py file we found") # Remembers context
```

#### 3. Advanced Tool Chaining

```python
# LangChain enables sophisticated tool chaining
from langchain.chains import SequentialChain

# Create a chain for complex file operations
analysis_chain = SequentialChain(
    chains=[
        # 1. List project structure
        SimpleSequentialChain([list_directory_tool]),
        # 2. Find key files
        SimpleSequentialChain([search_files_tool]), 
        # 3. Analyze and summarize
        SimpleSequentialChain([read_file_tool, write_file_tool])
    ]
)
```

### LangChain API Usage

#### Basic Tool Execution

```python
# Get LangChain tool executor
langchain_executor = get_langchain_tool_executor("/path/to/collection")

# Check available tools
if langchain_executor.has_available_tools():
    tools = langchain_executor.get_openai_tool_definitions()
    
# Execute tool calls (same format as original system)
tool_calls = [{
    "id": "call_1",
    "type": "function", 
    "function": {
        "name": "list_directory",
        "arguments": {"path": "src", "recursive": True}
    }
}]

results = await langchain_executor.execute_multiple_tool_calls_async(tool_calls)
```

#### Agent-Based Execution

```python
# Create and use a ReAct agent
agent = langchain_executor.create_react_agent(
    llm=get_llm("llama-3-8b-instruct"),
    memory=ConversationBufferMemory()
)

# Natural language instructions with reasoning
result = await agent.arun(
    "I need to understand this Python project. First explore the structure, "
    "then find the main entry points, and finally create a README with an overview."
)

print(result)  # Agent will use tools intelligently and provide reasoning
```

### Server API Integration

#### Using LangChain System via API

```bash
# Get tools using LangChain system
curl -X GET "http://localhost:8000/v1/tools?use_langchain=true"

# Execute tools using LangChain system  
curl -X POST "http://localhost:8000/v1/tools/execute?use_langchain=true" \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "list_directory",
    "arguments": {"path": "src", "recursive": true},
    "tool_call_id": "call_123"
  }'
```

#### Response Format

```json
{
  "success": true,
  "result": "Directory listing results...",
  "error": null,
  "tool_call_id": "call_123",
  "function_name": "list_directory",
  "system": "langchain"
}
```

### Migration Guide

#### Gradual Migration Strategy

1. **Phase 1**: Start using LangChain system for new features
2. **Phase 2**: Migrate complex workflows to LangChain agents
3. **Phase 3**: Leverage LangChain ecosystem integrations
4. **Phase 4**: Optional full migration when ready

#### Code Migration Examples

**Before (Original System):**
```python
# Direct tool execution
executor = get_tool_executor("/path")
results = await executor.execute_tool_calls(tool_calls)
```

**After (LangChain System):**
```python
# Enhanced execution with agent capabilities
executor = get_langchain_tool_executor("/path")
results = await executor.execute_multiple_tool_calls_async(tool_calls)
# OR use agent mode for complex reasoning
agent = executor.create_react_agent(llm)
result = await agent.arun("Complex natural language instruction")
```

### Performance Considerations

#### LangChain System Overhead

- **Startup Time**: Slightly higher due to LangChain initialization
- **Memory Usage**: Additional memory for agent state and memory
- **Execution Time**: Comparable for simple operations, faster for complex workflows
- **Benefits**: Advanced reasoning, memory, and ecosystem access

#### Optimization Tips

```python
# Reuse LangChain executors to avoid initialization overhead
executor = get_langchain_tool_executor("/path")
# Cache the executor for multiple operations

# Use async operations for better concurrency
results = await executor.execute_multiple_tool_calls_async(tool_calls)

# Configure memory limits for long conversations
memory = ConversationBufferWindowMemory(k=5)  # Limit to 5 recent exchanges
```

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

### Original System API

#### ToolExecutor Class

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

### LangChain System API

#### LangChainToolExecutor Class

**Purpose**: Advanced tool executor with agent capabilities and LangChain integration

**Initialization**:
```python
from agentic_tools import get_langchain_tool_executor

# Initialize with collection path
executor = get_langchain_tool_executor("/path/to/rag/collection")
```

#### Core Methods

- `get_available_tools() -> List[BaseTool]`
  - Returns list of LangChain BaseTool instances
  - Each tool maintains compatibility with original functionality

- `get_openai_tool_definitions() -> List[Dict[str, Any]]`
  - Returns OpenAI-compatible tool definitions
  - Compatible with both original and LangChain formats

- `execute_single_tool_call_async(tool_call: Dict) -> ToolExecutionResult`
  - Execute individual tool call with enhanced error handling
  - Supports both sync and async execution patterns

- `execute_multiple_tool_calls_async(tool_calls: List[Dict]) -> List[ToolExecutionResult]`
  - Concurrent execution of multiple tool calls
  - Optimized for LangChain agent workflows

- `has_available_tools() -> bool`
  - Check tool availability status
  - Validates LangChain tool initialization

#### Agent Creation Methods

- `create_react_agent(llm, memory=None, verbose=False) -> AgentExecutor`
  - Creates ReAct (Reasoning + Acting) agent with file system tools
  - Supports conversation memory and verbose logging
  - Returns configured `AgentExecutor` instance

**Example**:
```python
from langchain.memory import ConversationBufferMemory
from langchain_mlx import ChatMLX

# Create ReAct agent with memory
agent = executor.create_react_agent(
    llm=ChatMLX(model="llama-3-8b-instruct"),
    memory=ConversationBufferMemory(),
    verbose=True
)

# Execute with natural language instructions
result = await agent.arun(
    "Analyze the Python files in this project and create a summary"
)
```

#### LangChain Tool Wrapper Classes

Each original tool is wrapped as a LangChain `BaseTool`:

#### LangChainListDirectoryTool

```python
class LangChainListDirectoryTool(BaseTool):
    name: str = "list_directory"
    description: str = "List files and directories in a specified path within the RAG collection"
    
    class InputSchema(BaseModel):
        path: str = Field(default=".", description="Directory path to list")
        show_hidden: bool = Field(default=False, description="Include hidden files")
        recursive: bool = Field(default=False, description="List subdirectories recursively")
        pattern: Optional[str] = Field(None, description="File name pattern filter")
    
    def _run(self, path: str = ".", show_hidden: bool = False, 
             recursive: bool = False, pattern: str = None) -> str:
        """Synchronous execution"""
    
    async def _arun(self, path: str = ".", show_hidden: bool = False,
                    recursive: bool = False, pattern: str = None) -> str:
        """Asynchronous execution (recommended)"""
```

#### LangChainReadFileTool

```python
class LangChainReadFileTool(BaseTool):
    name: str = "read_file"
    description: str = "Read the contents of a text file with automatic encoding detection"
    
    class InputSchema(BaseModel):
        file_path: str = Field(description="Relative path to the file to read")
        encoding: Optional[str] = Field(None, description="Specific encoding (auto-detected if not specified)")
    
    def _run(self, file_path: str, encoding: str = None) -> str:
        """Synchronous file reading"""
    
    async def _arun(self, file_path: str, encoding: str = None) -> str:
        """Asynchronous file reading (recommended)"""
```

#### LangChainSearchFilesTool

```python
class LangChainSearchFilesTool(BaseTool):
    name: str = "search_files"
    description: str = "Search for patterns within files using regular expressions"
    
    class InputSchema(BaseModel):
        pattern: str = Field(description="Regular expression pattern to search for")
        file_pattern: str = Field(default="*", description="Glob pattern for files to search")
        max_results: int = Field(default=100, description="Maximum number of results")
        case_sensitive: bool = Field(default=False, description="Case-sensitive search")
    
    def _run(self, pattern: str, file_pattern: str = "*", 
             max_results: int = 100, case_sensitive: bool = False) -> str:
        """Synchronous search"""
    
    async def _arun(self, pattern: str, file_pattern: str = "*",
                    max_results: int = 100, case_sensitive: bool = False) -> str:
        """Asynchronous search (recommended)"""
```

#### LangChainWriteFileTool

```python
class LangChainWriteFileTool(BaseTool):
    name: str = "write_file"
    description: str = "Create a new file with specified content"
    
    class InputSchema(BaseModel):
        file_path: str = Field(description="Relative path for the new file")
        content: str = Field(description="Content to write to the file")
        encoding: str = Field(default="utf-8", description="Text encoding")
        create_directories: bool = Field(default=True, description="Create parent directories")
    
    def _run(self, file_path: str, content: str, 
             encoding: str = "utf-8", create_directories: bool = True) -> str:
        """Synchronous file creation"""
    
    async def _arun(self, file_path: str, content: str,
                    encoding: str = "utf-8", create_directories: bool = True) -> str:
        """Asynchronous file creation (recommended)"""
```

#### LangChainEditFileTool

```python
class LangChainEditFileTool(BaseTool):
    name: str = "edit_file"
    description: str = "Edit existing files using line-based operations"
    
    class InputSchema(BaseModel):
        file_path: str = Field(description="Relative path to the file to edit")
        operations: List[Dict[str, Any]] = Field(description="List of edit operations")
    
    def _run(self, file_path: str, operations: List[Dict[str, Any]]) -> str:
        """Synchronous file editing"""
    
    async def _arun(self, file_path: str, operations: List[Dict[str, Any]]) -> str:
        """Asynchronous file editing (recommended)"""
```

#### Advanced LangChain Features

#### Memory Integration

```python
# Conversation Buffer Memory
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

# Basic conversation memory
memory = ConversationBufferMemory(return_messages=True)

# Window memory (last N exchanges)
window_memory = ConversationBufferWindowMemory(
    k=5,  # Remember last 5 exchanges
    return_messages=True
)

# Use with agent
agent = executor.create_react_agent(llm, memory=memory)
```

#### Custom Prompt Templates

```python
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Create custom prompt for file system operations
file_system_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an AI assistant with access to file system tools. "
     "You can list directories, read files, search content, write new files, and edit existing files. "
     "Always validate file paths and handle errors gracefully. "
     "Use tools strategically and provide clear explanations of your actions."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Use custom prompt with agent
agent = executor.create_react_agent(llm, memory=memory, prompt=file_system_prompt)
```

#### Tool Result Processing

```python
# Enhanced tool result with LangChain metadata
class LangChainToolExecutionResult(ToolExecutionResult):
    """Extended result with LangChain-specific information"""
    langchain_metadata: Dict[str, Any] = None
    agent_reasoning: Optional[str] = None
    tool_chain: List[str] = None  # Track tool usage sequence
    
    def to_langchain_format(self) -> Dict[str, Any]:
        """Convert to LangChain-compatible format"""
        return {
            "output": self.result,
            "success": self.success,
            "error": self.error,
            "metadata": self.langchain_metadata or {},
            "execution_time_ms": self.execution_time_ms
        }
```

#### Async Execution Patterns

```python
# Recommended async patterns for LangChain integration

# Single tool execution
result = await executor.execute_single_tool_call_async({
    "id": "call_1",
    "type": "function",
    "function": {
        "name": "read_file",
        "arguments": {"file_path": "src/main.py"}
    }
})

# Multiple tool execution (concurrent)
results = await executor.execute_multiple_tool_calls_async([
    {"id": "call_1", "function": {"name": "list_directory", "arguments": {"path": "src"}}},
    {"id": "call_2", "function": {"name": "search_files", "arguments": {"pattern": "def main"}}}
])

# Agent-based execution (with reasoning)
result = await agent.arun("Analyze the project structure and summarize the main components")
```

#### Global Functions

```python
# LangChain-specific global functions

def get_langchain_tool_executor(collection_path: str) -> LangChainToolExecutor:
    """Get or create a LangChain tool executor for the specified collection"""
    
def clear_langchain_tool_executor():
    """Clear the cached LangChain tool executor"""
    
def get_langchain_tools_list(collection_path: str) -> List[BaseTool]:
    """Get list of available LangChain tools"""
    
def create_file_system_agent(llm, collection_path: str, **kwargs) -> AgentExecutor:
    """Convenience function to create a file system agent"""
```

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

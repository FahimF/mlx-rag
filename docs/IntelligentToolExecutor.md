# Intelligent Tool Executor Documentation

MLX-RAG's Intelligent Tool Executor is an advanced system that analyzes user queries and automatically executes relevant tools before generating responses. This capability is especially valuable for local language models that don't have native function calling support, as it bridges the gap by providing intelligent tool selection and execution.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Key Features](#key-features)
4. [Integration](#integration)
5. [Query Analysis](#query-analysis)
6. [Tool Detection Patterns](#tool-detection-patterns)
7. [API Reference](#api-reference)
8. [Usage Examples](#usage-examples)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)

## Overview

The Intelligent Tool Executor serves as a bridge between natural language queries and tool execution, automatically determining which tools to run based on user intent. Instead of requiring explicit function calls, users can express their needs in natural language, and the system will intelligently select and execute the appropriate tools.

### Problem Solved

Local language models often struggle with:
- **Function calling consistency**: May not reliably format function calls correctly
- **Tool selection**: Difficulty choosing the right tools for complex tasks
- **Multi-step workflows**: Poor coordination of multiple tools in sequence

The Intelligent Tool Executor addresses these limitations by:
- **Pre-execution analysis**: Analyzing queries before model processing
- **Smart tool selection**: Using regex patterns and heuristics to detect intent
- **Automatic execution**: Running tools and providing results as context
- **Enhanced prompting**: Creating better system prompts with tool results

## Architecture

```
User Query → Query Analysis → Tool Detection → Tool Execution → Context Enhancement → Model Response
     ↓              ↓               ↓                ↓                   ↓
Natural Language  Pattern       Selected Tools   Tool Results    Enhanced Prompt
                 Matching      with Arguments                   with Context
```

### Core Components

1. **IntelligentToolExecutor**: Main orchestration class
2. **Pattern Matching Engine**: Regex-based query analysis
3. **Argument Inference**: Smart parameter suggestion for tools
4. **Result Summarization**: Context generation from tool outputs
5. **Prompt Enhancement**: System prompt creation with tool context

## Key Features

### Smart Query Analysis
- **Regex Pattern Matching**: Sophisticated patterns for detecting tool needs
- **File/Directory Extraction**: Automatic extraction of file paths from queries
- **Intent Classification**: Categorizes queries into exploration, modification, or search intents

### Automatic Tool Execution
- **Pre-emptive Execution**: Runs tools before model processing
- **Argument Inference**: Intelligently suggests tool parameters
- **Multi-tool Coordination**: Can execute multiple tools for complex queries
- **Error Handling**: Graceful handling of tool execution failures

### Context Enhancement
- **Result Summarization**: Creates concise summaries of tool outputs
- **System Prompt Generation**: Enhanced prompts with tool context
- **Strategy Guidance**: Provides task-specific guidance for models

### Performance Optimizations
- **Duplicate Detection**: Prevents redundant tool executions
- **Efficient Processing**: Minimizes unnecessary computation
- **Logging Integration**: Comprehensive logging for debugging

## Integration

The Intelligent Tool Executor integrates with MLX-RAG's chat completions endpoint:

```python
from mlx_rag.intelligent_tool_executor import get_intelligent_tool_executor

# Initialize with a RAG collection
executor = get_intelligent_tool_executor("/path/to/rag/collection")

# Analyze and execute tools
tool_results, context = await executor.analyze_and_execute_tools(
    user_query="What files are in the src directory?",
    conversation_history=[]
)

# Generate enhanced system prompt
enhanced_prompt = executor.create_enhanced_system_prompt(
    tools=available_tools,
    user_query=user_query,
    tool_results=tool_results
)
```

### Server Integration

In the chat completions endpoint (`server.py`), the intelligent tool executor is used when:
1. Tools are available for the RAG collection
2. The model doesn't have native function calling support
3. The query suggests tool usage would be beneficial

## Query Analysis

### Pattern Categories

The system uses regex patterns to detect different types of user intents:

#### 1. Directory Exploration
**Triggers**: "list", "show", "explore", "browse", "what files", "directory structure"

**Examples**:
- "What files are in the src directory?"
- "Show me the project structure"
- "List all Python files"

#### 2. File Reading
**Triggers**: "read", "show", "display", "view", "examine", "what's in"

**Examples**:
- "Read the main.py file"
- "Show me the contents of config.yaml"
- "What's in the README?"

#### 3. Code Search
**Triggers**: "find", "search", "locate", "grep", "where is"

**Examples**:
- "Find all functions named 'process'"
- "Search for error handling code"
- "Where is the User class defined?"

#### 4. File Creation
**Triggers**: "create", "write", "make", "add", "generate"

**Examples**:
- "Create a new Python module"
- "Write a configuration file"
- "Add a README to the docs folder"

#### 5. File Modification
**Triggers**: "edit", "modify", "change", "update", "fix"

**Examples**:
- "Fix the bug in server.py"
- "Update the configuration"
- "Modify the import statement"

### File and Directory Extraction

The system automatically extracts file and directory references from queries:

#### File Pattern Detection
- **Extensions**: `file.py`, `config.json`, `README.md`
- **Quoted paths**: `"src/main.py"`, `'config/settings.yaml'`
- **Backtick paths**: `` `utils/helper.js` ``
- **Path structures**: `src/components/Button.tsx`

#### Directory Pattern Detection
- **Common directories**: `src`, `lib`, `app`, `components`, `utils`, `tests`, `docs`
- **Explicit references**: "in the src directory", "utils folder"
- **Quoted paths**: `"config/"`, `'src/components/'`

## Tool Detection Patterns

### List Directory Tool

**Pattern Examples**:
```regex
r'\b(?:list|show|see|explore|browse|what.*(?:files|directories|folders))\b.*\b(?:directory|folder|files|structure)\b'
r'\b(?:contents|files|directories)\b.*\b(?:in|of|inside)\b.*\b(?:directory|folder)\b'
r'\bls\b|\bdir\b|\btree\b.*structure'
```

**Argument Inference**:
- Defaults to current directory (`.`)
- Uses extracted directory if specified
- Detects file patterns (*.py, *.js, etc.)
- Identifies recursive requests

### Read File Tool

**Pattern Examples**:
```regex
r'\b(?:read|show|display|open|view|examine|check|see)\b.*\b(?:file|content|code)\b'
r'\b(?:what.*in|contents.*of|show.*me)\b.*\b(?:file|\.py|\.js|\.md|\.txt|\.json|\.yaml|\.yml)\b'
```

**Argument Inference**:
- Uses extracted file paths
- Falls back to common files (main.py, config.py, README.md)
- Returns None if no file can be determined

### Search Files Tool

**Pattern Examples**:
```regex
r'\b(?:find|search|locate|grep|look.*for)\b.*\b(?:in|across|through)\b.*\b(?:files|code|codebase)\b'
r'\b(?:where.*(?:defined|located|used)|find.*(?:function|class|variable|method))\b'
```

**Argument Inference**:
- Extracts quoted search terms
- Identifies function/class names
- Falls back to significant words
- Excludes common words (the, and, or, etc.)

### Special Scenarios

#### Project Exploration
**Triggers**: "What's in this project?", "Explore the codebase", "Project structure"

**Actions**:
1. List root directory
2. Read main files (main.py, app.py, __init__.py, README.md)

#### Functionality Inquiry
**Triggers**: "How does X work?", "What does Y do?", "Explain Z"

**Actions**:
1. Search for relevant terms
2. Read discovered files

#### Modification Without Specific File
**Triggers**: "Fix the bug", "Update the code", "Modify the implementation"

**Actions**:
1. List directory to show available files
2. Let user specify which file to modify

## API Reference

### IntelligentToolExecutor Class

#### Constructor

```python
def __init__(self, tool_executor=None):
    """
    Initialize the intelligent tool executor.
    
    Args:
        tool_executor: Optional ToolExecutor instance. If None, 
                      no tools will be executed.
    """
```

#### Main Methods

##### analyze_and_execute_tools()

```python
async def analyze_and_execute_tools(
    self, 
    user_query: str, 
    conversation_history: List[Dict[str, Any]] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Analyze user query and automatically execute relevant tools.
    
    Args:
        user_query: The user's request
        conversation_history: Previous conversation context (optional)
        
    Returns:
        Tuple of (tool_results, context_summary)
        - tool_results: List of executed tool results with metadata
        - context_summary: Formatted summary for system prompt
    """
```

**Example Usage**:
```python
executor = IntelligentToolExecutor(tool_executor)
results, context = await executor.analyze_and_execute_tools(
    "What Python files are in the src directory?"
)
print(f"Executed {len(results)} tools")
print(f"Generated context: {context}")
```

##### create_enhanced_system_prompt()

```python
def create_enhanced_system_prompt(
    self, 
    tools: List[Dict[str, Any]], 
    user_query: str,
    tool_results: List[Dict[str, Any]] = None
) -> str:
    """
    Create an enhanced system prompt that includes tool results and better instructions.
    
    Args:
        tools: Available tools for the model
        user_query: Original user query
        tool_results: Results from tool execution (optional)
        
    Returns:
        Enhanced system prompt string
    """
```

#### Internal Methods

##### _detect_required_tools()

Analyzes the user query and returns a list of tools that should be executed:

```python
def _detect_required_tools(self, user_query: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Detect which tools should be executed based on the user query.
    
    Returns:
        List of (tool_name, suggested_arguments) tuples
    """
```

##### _extract_file_references()

Extracts explicit file references from the user query:

```python
def _extract_file_references(self, query: str) -> List[str]:
    """Extract explicit file references from the query."""
```

**Detection Patterns**:
- `file.ext` - Simple filename with extension
- `path/file.ext` - File with path
- `"quoted/file.ext"` - Quoted file paths
- `` `backtick/file.ext` `` - Backtick-quoted paths

##### _extract_directory_references()

Extracts directory references from the user query:

```python
def _extract_directory_references(self, query: str) -> List[str]:
    """Extract directory references from the query."""
```

**Detection Patterns**:
- Common directories: `src`, `lib`, `app`, `components`, `utils`, `tests`, `docs`
- Contextual references: "src directory", "utils folder"
- Quoted paths: `"config/"`, `'src/components/'`

##### _suggest_tool_arguments()

Intelligently suggests arguments for detected tools:

```python
def _suggest_tool_arguments(
    self, 
    tool_name: str, 
    query: str, 
    explicit_files: List[str], 
    explicit_dirs: List[str]
) -> Optional[Dict[str, Any]]:
    """Suggest arguments for a tool based on the query context."""
```

**Logic by Tool**:
- **list_directory**: Uses extracted dirs or defaults to current directory
- **read_file**: Uses extracted files or infers from context (main.py, config.py)
- **search_files**: Extracts search terms from quotes or significant words
- **write_file**: Uses extracted file paths with placeholder content
- **edit_file**: Uses extracted file paths with minimal edit parameters

##### _summarize_tool_result()

Creates concise summaries of tool execution results:

```python
def _summarize_tool_result(self, tool_name: str, result: Any) -> str:
    """Create a concise summary of tool execution results."""
```

**Summary Formats**:
- **Directory listings**: Item counts, categories, and truncated lists
- **File reading**: Path, line counts, and content previews
- **Search results**: Match counts, affected files, and sample matches
- **File operations**: Success confirmations with paths

### Factory Function

```python
def get_intelligent_tool_executor(rag_collection_path: Optional[str] = None) -> IntelligentToolExecutor:
    """Get an intelligent tool executor for the given RAG collection."""
```

## Usage Examples

### Basic Usage

```python
from mlx_rag.intelligent_tool_executor import get_intelligent_tool_executor

# Initialize with RAG collection
executor = get_intelligent_tool_executor("/path/to/collection")

# Execute tools for exploration query
results, context = await executor.analyze_and_execute_tools(
    "What's in this project?"
)
# Automatically executes: list_directory, read main files

# Execute tools for search query
results, context = await executor.analyze_and_execute_tools(
    "Find all functions named 'process_data'"
)
# Automatically executes: search_files with query "process_data"
```

### Integration with Chat Completions

```python
async def enhanced_chat_completion(user_query: str, rag_collection_path: str):
    # Initialize intelligent executor
    intelligent_executor = get_intelligent_tool_executor(rag_collection_path)
    
    # Auto-execute tools based on query
    tool_results, context_summary = await intelligent_executor.analyze_and_execute_tools(
        user_query=user_query
    )
    
    # Create enhanced system prompt
    enhanced_system_prompt = intelligent_executor.create_enhanced_system_prompt(
        tools=available_tools,
        user_query=user_query,
        tool_results=tool_results
    )
    
    # Use enhanced prompt and context in model call
    response = await model.generate(
        messages=[
            {"role": "system", "content": enhanced_system_prompt},
            {"role": "user", "content": user_query}
        ]
    )
    
    return response, tool_results
```

### Query Examples and Tool Detection

#### Exploration Queries
```python
# "What files are in the src directory?"
# Detects: list_directory with path="src"

# "Show me the project structure"
# Detects: list_directory with path=".", recursive=True

# "What Python files are available?"
# Detects: list_directory with pattern="*.py"
```

#### Reading Queries
```python
# "Read the main.py file"
# Detects: read_file with path="main.py"

# "What's in the configuration?"
# Detects: read_file with path="config.py" (inferred)

# "Show me the contents of utils/helper.py"
# Detects: read_file with path="utils/helper.py"
```

#### Search Queries
```python
# "Find all functions named 'validate'"
# Detects: search_files with query="validate"

# "Where is the User class defined?"
# Detects: search_files with query="User"

# "Search for error handling code"
# Detects: search_files with query="error"
```

#### Special Scenarios
```python
# "What's in this project?"
# Detects: Multiple tools
# - list_directory with path="."
# - read_file for main.py, app.py, __init__.py, README.md

# "How does authentication work?"
# Detects: search_files with query="authentication"

# "Fix the bug in the code"
# Detects: list_directory to show available files
```

## Tool Detection Patterns

### Pattern Structure

Each tool has multiple regex patterns for robust detection:

```python
self.tool_patterns = {
    'list_directory': [
        r'\b(?:list|show|see|explore|browse|what.*(?:files|directories|folders))\b.*\b(?:directory|folder|files|structure)\b',
        r'\b(?:contents|files|directories)\b.*\b(?:in|of|inside)\b.*\b(?:directory|folder)\b',
        # ... more patterns
    ],
    # ... other tools
}
```

### Pattern Design Principles

1. **Word Boundaries**: Use `\b` to ensure whole word matching
2. **Flexible Ordering**: Patterns allow different word orders
3. **Synonym Support**: Include multiple ways to express the same intent
4. **Context Awareness**: Consider surrounding words and phrases

### Common Pattern Elements

- **Action Words**: list, show, read, find, search, create, edit
- **Object Words**: file, directory, folder, code, function, class
- **Context Words**: in, of, inside, for, with, containing

### File Extension Detection

The system recognizes file types in queries:
- `.py` files: Python source code
- `.js` files: JavaScript source code  
- `.md` files: Markdown documentation
- `.json` files: JSON configuration
- `.yaml`/`.yml` files: YAML configuration

## Configuration

### Logging Configuration

```python
import logging

# Enable debug logging for detailed analysis
logging.getLogger('mlx_rag.intelligent_tool_executor').setLevel(logging.DEBUG)

# Example log output:
# INFO: Detected tools to execute for query 'What files...': [('list_directory', {...})]
# INFO: Auto-executing tool: list_directory with args: {'path': '.'}
# INFO: Auto-executed 1 tools, generated 150 chars of context
```

### Pattern Customization

Extend or modify detection patterns:

```python
class CustomIntelligentToolExecutor(IntelligentToolExecutor):
    def __init__(self, tool_executor=None):
        super().__init__(tool_executor)
        
        # Add custom patterns
        self.tool_patterns['custom_tool'] = [
            r'\bcustom\b.*\baction\b',
            r'\bspecial\b.*\boperation\b'
        ]
        
        # Modify existing patterns
        self.tool_patterns['search_files'].append(
            r'\blocate\b.*\bcode\b.*\bpattern\b'
        )
```

### Result Summarization Customization

```python
def _summarize_tool_result(self, tool_name: str, result: Any) -> str:
    """Override to customize result summarization."""
    if tool_name == 'custom_tool':
        return f"Custom tool executed: {result}"
    return super()._summarize_tool_result(tool_name, result)
```

## Error Handling

### Tool Execution Failures

The system gracefully handles tool execution errors:

```python
try:
    result = await self.tool_executor.execute_tool_call(...)
    if result.success:
        # Process successful result
    else:
        logger.warning(f"Tool execution failed: {tool_name} - {result.error}")
except Exception as e:
    logger.error(f"Error auto-executing tool {tool_name}: {e}")
```

### Query Analysis Failures

- **No patterns matched**: Returns empty tool list
- **Invalid file/directory references**: Filters out invalid paths
- **Missing tool executor**: Returns empty results gracefully

### Graceful Degradation

When tool execution fails:
1. **Log the error**: Detailed error logging for debugging
2. **Continue processing**: Don't stop on individual tool failures
3. **Partial results**: Return results from successful tools
4. **Fallback behavior**: Model can still respond without tool context

## Performance Considerations

### Duplicate Prevention

The system prevents duplicate tool executions:

```python
# Remove duplicates while preserving order
seen = set()
unique_tools = []
for tool, args in tools_to_execute:
    key = (tool, str(sorted(args.items())))
    if key not in seen:
        seen.add(key)
        unique_tools.append((tool, args))
```

### Execution Metrics

Tool results include performance metrics:

```python
{
    'tool': 'list_directory',
    'arguments': {'path': '.'},
    'result': {...},
    'execution_time_ms': 45  # Execution time in milliseconds
}
```

### Memory Efficiency

- **Result Summarization**: Creates concise summaries instead of full results
- **Context Limitation**: Limits context length to prevent prompt overflow
- **Lazy Evaluation**: Only processes tools that match patterns

## Troubleshooting

### Common Issues

#### 1. No Tools Detected
**Symptoms**: Empty tool_results list
**Causes**: 
- Query patterns don't match
- No tool executor available
**Solutions**:
- Check regex patterns match your query style
- Verify tool executor is properly initialized
- Enable debug logging to see pattern matching

#### 2. Incorrect Tool Arguments
**Symptoms**: Tool execution failures
**Causes**:
- File paths don't exist
- Invalid argument inference
**Solutions**:
- Use explicit file paths in queries
- Check current working directory
- Verify file/directory references

#### 3. Tool Execution Timeout
**Symptoms**: Long response times
**Causes**:
- Large directory listings
- Complex search operations
**Solutions**:
- Use more specific queries
- Limit recursive operations
- Optimize tool implementations

### Debug Logging

Enable comprehensive logging:

```python
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('mlx_rag.intelligent_tool_executor')

# Example debug output:
# DEBUG: Extracted files: ['main.py', 'config.json']
# DEBUG: Extracted directories: ['src', 'utils']
# DEBUG: Pattern matched: list_directory
# DEBUG: Suggested arguments: {'path': 'src', 'pattern': '*.py'}
```

### Testing Tool Detection

Test pattern matching manually:

```python
executor = IntelligentToolExecutor()

# Test query analysis
tools = executor._detect_required_tools("What files are in src?")
print(f"Detected tools: {tools}")

# Test file extraction
files = executor._extract_file_references("Read main.py and config.json")
print(f"Extracted files: {files}")

# Test directory extraction  
dirs = executor._extract_directory_references("List files in the src directory")
print(f"Extracted directories: {dirs}")
```

## Best Practices

### Query Formulation

For best results, structure queries clearly:

**Good Examples**:
- "Read the main.py file"
- "List files in the src directory"
- "Find all functions named 'process'"
- "Show me the project structure"

**Less Optimal Examples**:
- "Tell me about that thing" (too vague)
- "Do something with files" (unclear intent)
- "Fix it" (no specific file reference)

### Integration Guidelines

1. **Initialize Early**: Create the executor once and reuse
2. **Handle Failures**: Always check tool execution results
3. **Context Management**: Use context summaries in system prompts
4. **Performance Monitoring**: Track execution times and success rates
5. **Error Logging**: Enable appropriate logging levels

### Extension Points

The system is designed for extensibility:

1. **New Tools**: Add patterns for custom tools
2. **Enhanced Patterns**: Improve existing pattern matching
3. **Custom Summarization**: Override result summarization
4. **Query Preprocessing**: Add query normalization
5. **Context Enhancement**: Extend system prompt generation

## Future Enhancements

Potential improvements to the Intelligent Tool Executor:

1. **Machine Learning Integration**: Train models to improve tool detection
2. **Conversation Context**: Use conversation history for better tool selection
3. **User Preferences**: Learn user-specific tool usage patterns
4. **Performance Optimization**: Cache frequent tool results
5. **Advanced Query Understanding**: Natural language processing improvements
6. **Tool Chaining**: Automatic multi-step tool workflows
7. **Result Filtering**: Intelligent filtering of tool results based on relevance

---

The Intelligent Tool Executor represents a significant advancement in making local language models more capable with tool usage, providing a seamless bridge between natural language queries and precise tool execution.

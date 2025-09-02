"""
LangChain-based Agentic Tools System for MLX-RAG

This module provides a LangChain-compatible framework for LLMs to interact with RAG 
collection source folders through various tools like file search, editing, creation, etc.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Type
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
import fnmatch
import difflib

from langchain.tools import BaseTool
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    properties: Optional[Dict[str, Any]] = None  # For object types


@dataclass
class ToolDefinition:
    """Definition of an agentic tool."""
    name: str
    description: str
    parameters: List[ToolParameter]
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop_def = {
                "type": param.type,
                "description": param.description
            }
            
            if param.enum:
                prop_def["enum"] = param.enum
            if param.properties:
                prop_def["properties"] = param.properties
                
            properties[param.name] = prop_def
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


@dataclass
class ToolExecutionResult:
    """Result of tool execution."""
    success: bool
    result: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AgenticTool(ABC):
    """Base class for agentic tools."""
    
    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Get the tool definition."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolExecutionResult:
        """Execute the tool with given parameters."""
        pass
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate parameters against the tool definition."""
        required_params = [p.name for p in self.definition.parameters if p.required]
        
        # Check required parameters
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Check parameter types (basic validation)
        param_types = {p.name: p.type for p in self.definition.parameters}
        for param_name, value in kwargs.items():
            if param_name in param_types:
                expected_type = param_types[param_name]
                if not self._validate_type(value, expected_type):
                    raise ValueError(f"Parameter {param_name} must be of type {expected_type}")
        
        return True
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate a single parameter type."""
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "number":
            return isinstance(value, (int, float))
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, list)
        elif expected_type == "object":
            return isinstance(value, dict)
        return True
    
    def get_openai_function_definition(self) -> Dict[str, Any]:
        """Convert tool definition to OpenAI function definition format."""
        definition = self.definition
        
        # Build properties and required parameters
        properties = {}
        required = []
        
        for param in definition.parameters:
            param_def = {
                "type": param.type,
                "description": param.description
            }
            
            # Add enum values if present
            if hasattr(param, 'enum') and param.enum:
                param_def["enum"] = param.enum
            
            properties[param.name] = param_def
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": definition.name,
            "description": definition.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


class FileSystemTool(AgenticTool):
    """Base class for file system tools with safety restrictions."""
    
    def __init__(self, collection_path: str):
        """Initialize with RAG collection source path."""
        self.collection_path = Path(collection_path).resolve()
        if not self.collection_path.exists():
            raise ValueError(f"Collection path does not exist: {collection_path}")
        if not self.collection_path.is_dir():
            raise ValueError(f"Collection path is not a directory: {collection_path}")
    
    def _validate_path(self, path: str) -> Path:
        """Validate that a path is within the collection directory."""
        target_path = (self.collection_path / path).resolve()
        
        # Ensure the path is within the collection directory
        try:
            target_path.relative_to(self.collection_path)
        except ValueError:
            raise PermissionError(f"Path '{path}' is outside the allowed collection directory")
        
        return target_path
    
    def _is_text_file(self, path: Path) -> bool:
        """Check if a file is likely a text file."""
        text_extensions = {
            '.txt', '.md', '.py', '.js', '.ts', '.html', '.css', '.json', '.yaml', '.yml',
            '.xml', '.csv', '.sql', '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat',
            '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.java', '.kt', '.scala',
            '.rs', '.go', '.php', '.rb', '.pl', '.r', '.swift', '.m', '.mm',
            '.vue', '.jsx', '.tsx', '.svelte', '.dart', '.lua', '.nim', '.zig'
        }
        
        return path.suffix.lower() in text_extensions


class ListDirectoryTool(FileSystemTool):
    """Tool to list files and directories in the collection."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_directory",
            description="List files and directories in the RAG collection source folder. Use this to explore the codebase structure.",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Relative path within the collection to list (default: root)",
                    required=False
                ),
                ToolParameter(
                    name="include_hidden",
                    type="boolean",
                    description="Whether to include hidden files and directories (default: false)",
                    required=False
                ),
                ToolParameter(
                    name="pattern",
                    type="string",
                    description="File pattern to match (glob pattern, e.g., '*.py')",
                    required=False
                )
            ]
        )
    
    async def execute(self, **kwargs) -> ToolExecutionResult:
        """Execute directory listing."""
        try:
            self.validate_parameters(**kwargs)
            
            path = kwargs.get("path", "")
            include_hidden = kwargs.get("include_hidden", False)
            pattern = kwargs.get("pattern", "*")
            
            target_path = self._validate_path(path)
            
            if not target_path.exists():
                return ToolExecutionResult(
                    success=False,
                    result=None,
                    error=f"Directory '{path}' does not exist"
                )
            
            if not target_path.is_dir():
                return ToolExecutionResult(
                    success=False,
                    result=None,
                    error=f"'{path}' is not a directory"
                )
            
            items = []
            for item in target_path.iterdir():
                # Skip hidden files if not requested
                if not include_hidden and item.name.startswith('.'):
                    continue
                
                # Apply pattern filter
                if not fnmatch.fnmatch(item.name, pattern):
                    continue
                
                rel_path = item.relative_to(self.collection_path)
                
                item_info = {
                    "name": item.name,
                    "path": str(rel_path),
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                    "modified": item.stat().st_mtime
                }
                
                if item.is_file():
                    item_info["is_text"] = self._is_text_file(item)
                
                items.append(item_info)
            
            # Sort by type (directories first), then by name
            items.sort(key=lambda x: (x["type"] != "directory", x["name"].lower()))
            
            return ToolExecutionResult(
                success=True,
                result={
                    "path": str(target_path.relative_to(self.collection_path)),
                    "items": items,
                    "total": len(items)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in list_directory tool: {e}")
            return ToolExecutionResult(
                success=False,
                result=None,
                error=str(e)
            )


class ReadFileTool(FileSystemTool):
    """Tool to read file contents."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_file",
            description="Read the contents of a file in the RAG collection. Use this to examine source code, configuration files, documentation, etc.",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Relative path to the file within the collection"
                ),
                ToolParameter(
                    name="start_line",
                    type="number",
                    description="Start reading from this line number (1-based, default: 1)",
                    required=False
                ),
                ToolParameter(
                    name="end_line",
                    type="number",
                    description="Stop reading at this line number (1-based, default: end of file)",
                    required=False
                )
            ]
        )
    
    async def execute(self, **kwargs) -> ToolExecutionResult:
        """Execute file reading."""
        try:
            self.validate_parameters(**kwargs)
            
            path = kwargs["path"]
            start_line = kwargs.get("start_line", 1)
            end_line = kwargs.get("end_line")
            
            target_path = self._validate_path(path)
            
            if not target_path.exists():
                return ToolExecutionResult(
                    success=False,
                    result=None,
                    error=f"File '{path}' does not exist"
                )
            
            if not target_path.is_file():
                return ToolExecutionResult(
                    success=False,
                    result=None,
                    error=f"'{path}' is not a file"
                )
            
            # Check if file is likely binary
            if not self._is_text_file(target_path):
                return ToolExecutionResult(
                    success=False,
                    result=None,
                    error=f"'{path}' appears to be a binary file"
                )
            
            try:
                with open(target_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Apply line range filtering
                total_lines = len(lines)
                start_idx = max(0, start_line - 1)  # Convert to 0-based
                end_idx = min(total_lines, end_line) if end_line else total_lines
                
                if start_idx >= total_lines:
                    return ToolExecutionResult(
                        success=False,
                        result=None,
                        error=f"Start line {start_line} is beyond the file length ({total_lines} lines)"
                    )
                
                selected_lines = lines[start_idx:end_idx]
                content = ''.join(selected_lines)
                
                return ToolExecutionResult(
                    success=True,
                    result={
                        "path": path,
                        "content": content,
                        "total_lines": total_lines,
                        "start_line": start_line,
                        "end_line": min(end_idx, total_lines),
                        "lines_read": len(selected_lines)
                    }
                )
                
            except UnicodeDecodeError:
                return ToolExecutionResult(
                    success=False,
                    result=None,
                    error=f"'{path}' contains non-UTF-8 content and cannot be read as text"
                )
            
        except Exception as e:
            logger.error(f"Error in read_file tool: {e}")
            return ToolExecutionResult(
                success=False,
                result=None,
                error=str(e)
            )


class SearchFilesTool(FileSystemTool):
    """Tool to search for text within files."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search_files",
            description="Search for text patterns within files in the RAG collection. Use this to find specific code, functions, or text content.",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Text to search for"
                ),
                ToolParameter(
                    name="path",
                    type="string",
                    description="Relative path to search within (default: entire collection)",
                    required=False
                ),
                ToolParameter(
                    name="file_pattern",
                    type="string",
                    description="File pattern to search within (glob pattern, e.g., '*.py')",
                    required=False
                ),
                ToolParameter(
                    name="case_sensitive",
                    type="boolean",
                    description="Whether the search should be case sensitive (default: false)",
                    required=False
                ),
                ToolParameter(
                    name="max_results",
                    type="number",
                    description="Maximum number of results to return (default: 50)",
                    required=False
                )
            ]
        )
    
    async def execute(self, **kwargs) -> ToolExecutionResult:
        """Execute file search."""
        try:
            self.validate_parameters(**kwargs)
            
            query = kwargs["query"]
            search_path = kwargs.get("path", "")
            file_pattern = kwargs.get("file_pattern", "*")
            case_sensitive = kwargs.get("case_sensitive", False)
            max_results = kwargs.get("max_results", 50)
            
            target_path = self._validate_path(search_path)
            
            if not target_path.exists():
                return ToolExecutionResult(
                    success=False,
                    result=None,
                    error=f"Search path '{search_path}' does not exist"
                )
            
            search_query = query if case_sensitive else query.lower()
            results = []
            
            # Walk through the directory tree
            for root, dirs, files in os.walk(target_path):
                root_path = Path(root)
                
                for file in files:
                    # Apply file pattern filter
                    if not fnmatch.fnmatch(file, file_pattern):
                        continue
                    
                    file_path = root_path / file
                    
                    # Skip non-text files
                    if not self._is_text_file(file_path):
                        continue
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        for line_num, line in enumerate(lines, 1):
                            search_line = line if case_sensitive else line.lower()
                            
                            if search_query in search_line:
                                rel_path = file_path.relative_to(self.collection_path)
                                
                                results.append({
                                    "file": str(rel_path),
                                    "line": line_num,
                                    "content": line.strip(),
                                    "context": self._get_context_lines(lines, line_num - 1, 2)
                                })
                                
                                if len(results) >= max_results:
                                    break
                    
                    except (UnicodeDecodeError, PermissionError):
                        # Skip files that can't be read
                        continue
                    
                    if len(results) >= max_results:
                        break
                
                if len(results) >= max_results:
                    break
            
            return ToolExecutionResult(
                success=True,
                result={
                    "query": query,
                    "results": results,
                    "total_found": len(results),
                    "search_path": search_path or "entire collection",
                    "file_pattern": file_pattern
                }
            )
            
        except Exception as e:
            logger.error(f"Error in search_files tool: {e}")
            return ToolExecutionResult(
                success=False,
                result=None,
                error=str(e)
            )
    
    def _get_context_lines(self, lines: List[str], center_idx: int, context_size: int) -> List[str]:
        """Get context lines around a match."""
        start = max(0, center_idx - context_size)
        end = min(len(lines), center_idx + context_size + 1)
        
        context_lines = []
        for i in range(start, end):
            prefix = ">>> " if i == center_idx else "    "
            context_lines.append(f"{prefix}{i+1}: {lines[i].rstrip()}")
        
        return context_lines


class WriteFileTool(FileSystemTool):
    """Tool to write/create files."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="write_file",
            description="Write content to a file in the RAG collection. This will create a new file or overwrite an existing one. Use with caution!",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Relative path to the file within the collection"
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Content to write to the file"
                ),
                ToolParameter(
                    name="create_dirs",
                    type="boolean",
                    description="Whether to create parent directories if they don't exist (default: false)",
                    required=False
                )
            ]
        )
    
    async def execute(self, **kwargs) -> ToolExecutionResult:
        """Execute file writing."""
        try:
            self.validate_parameters(**kwargs)
            
            path = kwargs["path"]
            content = kwargs["content"]
            create_dirs = kwargs.get("create_dirs", False)
            
            target_path = self._validate_path(path)
            
            # Check if parent directory exists
            parent_dir = target_path.parent
            if not parent_dir.exists():
                if create_dirs:
                    parent_dir.mkdir(parents=True, exist_ok=True)
                else:
                    return ToolExecutionResult(
                        success=False,
                        result=None,
                        error=f"Parent directory '{parent_dir.relative_to(self.collection_path)}' does not exist"
                    )
            
            # Check if file already exists and get backup
            backup_content = None
            if target_path.exists():
                try:
                    with open(target_path, 'r', encoding='utf-8') as f:
                        backup_content = f.read()
                except (UnicodeDecodeError, PermissionError):
                    backup_content = None
            
            # Write the new content
            try:
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return ToolExecutionResult(
                    success=True,
                    result={
                        "path": path,
                        "bytes_written": len(content.encode('utf-8')),
                        "lines_written": len(content.splitlines()),
                        "was_existing": backup_content is not None,
                        "backup_available": backup_content is not None
                    },
                    metadata={
                        "backup_content": backup_content  # For potential rollback
                    }
                )
                
            except PermissionError:
                return ToolExecutionResult(
                    success=False,
                    result=None,
                    error=f"Permission denied writing to '{path}'"
                )
            
        except Exception as e:
            logger.error(f"Error in write_file tool: {e}")
            return ToolExecutionResult(
                success=False,
                result=None,
                error=str(e)
            )


class EditFileTool(FileSystemTool):
    """Tool to edit files with line-based modifications."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="edit_file",
            description="Edit a file by replacing specific lines or sections. This is safer than overwriting entire files.",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Relative path to the file within the collection"
                ),
                ToolParameter(
                    name="start_line",
                    type="number",
                    description="Start line number to replace (1-based)"
                ),
                ToolParameter(
                    name="end_line",
                    type="number",
                    description="End line number to replace (1-based, inclusive)"
                ),
                ToolParameter(
                    name="new_content",
                    type="string",
                    description="New content to replace the selected lines with"
                )
            ]
        )
    
    async def execute(self, **kwargs) -> ToolExecutionResult:
        """Execute file editing."""
        try:
            self.validate_parameters(**kwargs)
            
            path = kwargs["path"]
            start_line = kwargs["start_line"]
            end_line = kwargs["end_line"]
            new_content = kwargs["new_content"]
            
            target_path = self._validate_path(path)
            
            if not target_path.exists():
                return ToolExecutionResult(
                    success=False,
                    result=None,
                    error=f"File '{path}' does not exist"
                )
            
            if not target_path.is_file():
                return ToolExecutionResult(
                    success=False,
                    result=None,
                    error=f"'{path}' is not a file"
                )
            
            # Validate line numbers
            if start_line < 1 or end_line < 1 or start_line > end_line:
                return ToolExecutionResult(
                    success=False,
                    result=None,
                    error="Invalid line numbers. Start and end must be >= 1 and start <= end"
                )
            
            try:
                # Read the original file
                with open(target_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                original_content = ''.join(lines)
                total_lines = len(lines)
                
                # Validate line range
                if start_line > total_lines:
                    return ToolExecutionResult(
                        success=False,
                        result=None,
                        error=f"Start line {start_line} is beyond file length ({total_lines} lines)"
                    )
                
                # Convert to 0-based indexing
                start_idx = start_line - 1
                end_idx = min(end_line, total_lines)
                
                # Prepare new content lines
                new_lines = new_content.split('\n')
                if not new_content.endswith('\n') and new_lines:
                    new_lines[-1] += '\n'  # Preserve line endings
                
                # Replace the lines
                modified_lines = lines[:start_idx] + [line + '\n' if not line.endswith('\n') else line for line in new_lines] + lines[end_idx:]
                modified_content = ''.join(modified_lines)
                
                # Write the modified content back
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                
                # Generate diff for review
                diff = list(difflib.unified_diff(
                    original_content.splitlines(keepends=True),
                    modified_content.splitlines(keepends=True),
                    fromfile=f"{path} (original)",
                    tofile=f"{path} (modified)",
                    lineterm=""
                ))
                
                return ToolExecutionResult(
                    success=True,
                    result={
                        "path": path,
                        "lines_replaced": end_idx - start_idx,
                        "new_lines_count": len(new_lines),
                        "total_lines_after": len(modified_lines),
                        "diff": ''.join(diff)
                    },
                    metadata={
                        "original_content": original_content  # For potential rollback
                    }
                )
                
            except UnicodeDecodeError:
                return ToolExecutionResult(
                    success=False,
                    result=None,
                    error=f"'{path}' contains non-UTF-8 content and cannot be edited"
                )
            
        except Exception as e:
            logger.error(f"Error in edit_file tool: {e}")
            return ToolExecutionResult(
                success=False,
                result=None,
                error=str(e)
            )


class AgenticToolManager:
    """Manager for agentic tools."""
    
    def __init__(self):
        self.tools: Dict[str, AgenticTool] = {}
    
    def register_tool(self, tool: AgenticTool):
        """Register a tool."""
        self.tools[tool.definition.name] = tool
        logger.info(f"Registered agentic tool: {tool.definition.name}")
    
    def get_tool(self, name: str) -> Optional[AgenticTool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI-format tool definitions."""
        return [tool.definition.to_openai_format() for tool in self.tools.values()]
    
    async def execute_tool(self, tool_name: str, **kwargs) -> ToolExecutionResult:
        """Execute a tool by name."""
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolExecutionResult(
                success=False,
                result=None,
                error=f"Tool '{tool_name}' not found"
            )
        
        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return ToolExecutionResult(
                success=False,
                result=None,
                error=str(e)
            )
    
    def create_file_system_tools(self, collection_path: str) -> List[AgenticTool]:
        """Create and register file system tools for a collection."""
        tools = [
            ListDirectoryTool(collection_path),
            ReadFileTool(collection_path),
            SearchFilesTool(collection_path),
            WriteFileTool(collection_path),
            EditFileTool(collection_path)
        ]
        
        for tool in tools:
            self.register_tool(tool)
        
        return tools


# LangChain-compatible tool wrappers

class ListDirectoryInput(BaseModel):
    """Input schema for ListDirectoryTool."""
    path: str = Field(default="", description="Relative path within the collection to list")
    include_hidden: bool = Field(default=False, description="Include hidden files and directories")
    pattern: str = Field(default="*", description="File pattern to match (glob pattern)")

class LangChainListDirectoryTool(BaseTool):
    """LangChain wrapper for ListDirectoryTool."""
    name: str = "list_directory"
    description: str = "List files and directories in the RAG collection source folder. Use this to explore the codebase structure."
    args_schema: Type[BaseModel] = ListDirectoryInput
    
    def __init__(self, collection_path: str, **kwargs):
        super().__init__(**kwargs)
        self._internal_tool = ListDirectoryTool(collection_path)
    
    def _run(self, path: str = "", include_hidden: bool = False, pattern: str = "*") -> str:
        """Synchronous implementation."""
        # Run async method synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self._internal_tool.execute(
                    path=path, include_hidden=include_hidden, pattern=pattern
                )
            )
            if result.success:
                return json.dumps(result.result, indent=2)
            else:
                raise ToolException(result.error or "Unknown error")
        finally:
            loop.close()
    
    async def _arun(self, path: str = "", include_hidden: bool = False, pattern: str = "*") -> str:
        """Asynchronous implementation."""
        result = await self._internal_tool.execute(
            path=path, include_hidden=include_hidden, pattern=pattern
        )
        if result.success:
            return json.dumps(result.result, indent=2)
        else:
            raise ToolException(result.error or "Unknown error")


class ReadFileInput(BaseModel):
    """Input schema for ReadFileTool."""
    path: str = Field(description="Relative path to the file within the collection")
    start_line: Optional[int] = Field(default=1, description="Start reading from this line number (1-based)")
    end_line: Optional[int] = Field(default=None, description="Stop reading at this line number (1-based)")

class LangChainReadFileTool(BaseTool):
    """LangChain wrapper for ReadFileTool."""
    name: str = "read_file"
    description: str = "Read the contents of a file in the RAG collection. Use this to examine source code, configuration files, documentation, etc."
    args_schema: Type[BaseModel] = ReadFileInput
    
    def __init__(self, collection_path: str, **kwargs):
        super().__init__(**kwargs)
        self._internal_tool = ReadFileTool(collection_path)
    
    def _run(self, path: str, start_line: int = 1, end_line: Optional[int] = None) -> str:
        """Synchronous implementation."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self._internal_tool.execute(
                    path=path, start_line=start_line, end_line=end_line
                )
            )
            if result.success:
                return json.dumps(result.result, indent=2)
            else:
                raise ToolException(result.error or "Unknown error")
        finally:
            loop.close()
    
    async def _arun(self, path: str, start_line: int = 1, end_line: Optional[int] = None) -> str:
        """Asynchronous implementation."""
        result = await self._internal_tool.execute(
            path=path, start_line=start_line, end_line=end_line
        )
        if result.success:
            return json.dumps(result.result, indent=2)
        else:
            raise ToolException(result.error or "Unknown error")


class SearchFilesInput(BaseModel):
    """Input schema for SearchFilesTool."""
    query: str = Field(description="Text to search for")
    path: str = Field(default="", description="Relative path to search within (default: entire collection)")
    file_pattern: str = Field(default="*", description="File pattern to search within (glob pattern)")
    case_sensitive: bool = Field(default=False, description="Whether the search should be case sensitive")
    max_results: int = Field(default=50, description="Maximum number of results to return")

class LangChainSearchFilesTool(BaseTool):
    """LangChain wrapper for SearchFilesTool."""
    name: str = "search_files"
    description: str = "Search for text patterns within files in the RAG collection. Use this to find specific code, functions, or text content."
    args_schema: Type[BaseModel] = SearchFilesInput
    
    def __init__(self, collection_path: str, **kwargs):
        super().__init__(**kwargs)
        self._internal_tool = SearchFilesTool(collection_path)
    
    def _run(self, query: str, path: str = "", file_pattern: str = "*", 
             case_sensitive: bool = False, max_results: int = 50) -> str:
        """Synchronous implementation."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self._internal_tool.execute(
                    query=query, path=path, file_pattern=file_pattern,
                    case_sensitive=case_sensitive, max_results=max_results
                )
            )
            if result.success:
                return json.dumps(result.result, indent=2)
            else:
                raise ToolException(result.error or "Unknown error")
        finally:
            loop.close()
    
    async def _arun(self, query: str, path: str = "", file_pattern: str = "*", 
                   case_sensitive: bool = False, max_results: int = 50) -> str:
        """Asynchronous implementation."""
        result = await self._internal_tool.execute(
            query=query, path=path, file_pattern=file_pattern,
            case_sensitive=case_sensitive, max_results=max_results
        )
        if result.success:
            return json.dumps(result.result, indent=2)
        else:
            raise ToolException(result.error or "Unknown error")


class WriteFileInput(BaseModel):
    """Input schema for WriteFileTool."""
    path: str = Field(description="Relative path to the file within the collection")
    content: str = Field(description="Content to write to the file")
    create_dirs: bool = Field(default=False, description="Whether to create parent directories if they don't exist")

class LangChainWriteFileTool(BaseTool):
    """LangChain wrapper for WriteFileTool."""
    name: str = "write_file"
    description: str = "Write content to a file in the RAG collection. This will create a new file or overwrite an existing one. Use with caution!"
    args_schema: Type[BaseModel] = WriteFileInput
    
    def __init__(self, collection_path: str, **kwargs):
        super().__init__(**kwargs)
        self._internal_tool = WriteFileTool(collection_path)
    
    def _run(self, path: str, content: str, create_dirs: bool = False) -> str:
        """Synchronous implementation."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self._internal_tool.execute(
                    path=path, content=content, create_dirs=create_dirs
                )
            )
            if result.success:
                return json.dumps(result.result, indent=2)
            else:
                raise ToolException(result.error or "Unknown error")
        finally:
            loop.close()
    
    async def _arun(self, path: str, content: str, create_dirs: bool = False) -> str:
        """Asynchronous implementation."""
        result = await self._internal_tool.execute(
            path=path, content=content, create_dirs=create_dirs
        )
        if result.success:
            return json.dumps(result.result, indent=2)
        else:
            raise ToolException(result.error or "Unknown error")


class EditFileInput(BaseModel):
    """Input schema for EditFileTool."""
    path: str = Field(description="Relative path to the file within the collection")
    start_line: int = Field(description="Start line number to replace (1-based)")
    end_line: int = Field(description="End line number to replace (1-based, inclusive)")
    new_content: str = Field(description="New content to replace the selected lines with")

class LangChainEditFileTool(BaseTool):
    """LangChain wrapper for EditFileTool."""
    name: str = "edit_file"
    description: str = "Edit a file by replacing specific lines or sections. This is safer than overwriting entire files."
    args_schema: Type[BaseModel] = EditFileInput
    
    def __init__(self, collection_path: str, **kwargs):
        super().__init__(**kwargs)
        self._internal_tool = EditFileTool(collection_path)
    
    def _run(self, path: str, start_line: int, end_line: int, new_content: str) -> str:
        """Synchronous implementation."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self._internal_tool.execute(
                    path=path, start_line=start_line, end_line=end_line, new_content=new_content
                )
            )
            if result.success:
                return json.dumps(result.result, indent=2)
            else:
                raise ToolException(result.error or "Unknown error")
        finally:
            loop.close()
    
    async def _arun(self, path: str, start_line: int, end_line: int, new_content: str) -> str:
        """Asynchronous implementation."""
        result = await self._internal_tool.execute(
            path=path, start_line=start_line, end_line=end_line, new_content=new_content
        )
        if result.success:
            return json.dumps(result.result, indent=2)
        else:
            raise ToolException(result.error or "Unknown error")


# LangChain Tool Factory
class LangChainToolFactory:
    """Factory for creating LangChain-compatible tools."""
    
    @staticmethod
    def create_file_system_tools(collection_path: str) -> List[BaseTool]:
        """Create all LangChain file system tools for a collection."""
        return [
            LangChainListDirectoryTool(collection_path),
            LangChainReadFileTool(collection_path),
            LangChainSearchFilesTool(collection_path),
            LangChainWriteFileTool(collection_path),
            LangChainEditFileTool(collection_path)
        ]
    
    @staticmethod
    def get_tool_by_name(tool_name: str, collection_path: str) -> Optional[BaseTool]:
        """Get a specific tool by name."""
        tool_map = {
            "list_directory": LangChainListDirectoryTool,
            "read_file": LangChainReadFileTool,
            "search_files": LangChainSearchFilesTool,
            "write_file": LangChainWriteFileTool,
            "edit_file": LangChainEditFileTool
        }
        
        tool_class = tool_map.get(tool_name)
        if tool_class:
            return tool_class(collection_path)
        return None


# Global tool manager instance
_tool_manager = None

def get_tool_manager() -> AgenticToolManager:
    """Get the global tool manager instance."""
    global _tool_manager
    if _tool_manager is None:
        _tool_manager = AgenticToolManager()
    return _tool_manager


def create_langchain_tools(collection_path: str) -> List[BaseTool]:
    """Create LangChain-compatible tools for the given collection path."""
    return LangChainToolFactory.create_file_system_tools(collection_path)

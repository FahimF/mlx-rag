"""
Tool execution framework for MLX-RAG agentic capabilities.

This module manages the execution of agentic tools with proper error handling,
validation, and sandboxing to ensure secure operation.
"""

import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import asyncio

from mlx_rag.agentic_tools import (
    AgenticTool, 
    ListDirectoryTool, 
    ReadFileTool, 
    SearchFilesTool, 
    WriteFileTool, 
    EditFileTool
)

logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""
    pass


class ToolExecutionResult:
    """Result of tool execution with metadata."""
    
    def __init__(
        self, 
        tool_call_id: str, 
        function_name: str, 
        success: bool, 
        result: Any = None, 
        error: str = None,
        execution_time_ms: float = 0
    ):
        self.tool_call_id = tool_call_id
        self.function_name = function_name
        self.success = success
        self.result = result
        self.error = error
        self.execution_time_ms = execution_time_ms
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI tool call result format."""
        if self.success:
            content = json.dumps(self.result) if self.result is not None else ""
        else:
            content = f"Error: {self.error}"
        
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": self.tool_call_id
        }


class ToolExecutor:
    """Manages execution of agentic tools with safety and validation."""
    
    def __init__(self, rag_collection_path: Optional[str] = None):
        """Initialize the tool executor.
        
        Args:
            rag_collection_path: Path to the RAG collection source directory.
                               If None, tools will not be available.
        """
        self.rag_collection_path = rag_collection_path
        self._available_tools: Dict[str, AgenticTool] = {}
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize available tools based on the collection path."""
        if not self.rag_collection_path:
            logger.warning("No RAG collection path provided, tools will not be available")
            return
        
        # Validate collection path exists and is accessible
        collection_path = Path(self.rag_collection_path)
        if not collection_path.exists():
            logger.error(f"RAG collection path does not exist: {self.rag_collection_path}")
            return
        
        if not collection_path.is_dir():
            logger.error(f"RAG collection path is not a directory: {self.rag_collection_path}")
            return
        
        # Initialize filesystem tools
        try:
            self._available_tools["list_directory"] = ListDirectoryTool(self.rag_collection_path)
            self._available_tools["read_file"] = ReadFileTool(self.rag_collection_path)
            self._available_tools["search_files"] = SearchFilesTool(self.rag_collection_path)
            self._available_tools["write_file"] = WriteFileTool(self.rag_collection_path)
            self._available_tools["edit_file"] = EditFileTool(self.rag_collection_path)
            
            logger.info(f"Initialized {len(self._available_tools)} agentic tools for collection: {self.rag_collection_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize agentic tools: {e}")
            self._available_tools.clear()
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get OpenAI-compatible tool definitions for all available tools.
        
        Returns:
            Dictionary mapping tool names to their OpenAI function definitions.
        """
        tool_definitions = {}
        
        for tool_name, tool in self._available_tools.items():
            tool_definitions[tool_name] = {
                "type": "function",
                "function": tool.get_openai_function_definition()
            }
        
        return tool_definitions
    
    def get_tools_for_openai_request(self) -> List[Dict[str, Any]]:
        """Get tools formatted for OpenAI chat completion request.
        
        Returns:
            List of tool definitions suitable for OpenAI API requests.
        """
        tools = []
        
        for tool_name, tool in self._available_tools.items():
            tools.append({
                "type": "function",
                "function": tool.get_openai_function_definition()
            })
        
        return tools
    
    async def execute_tool_call(
        self, 
        tool_call_id: str, 
        function_name: str, 
        arguments: Dict[str, Any]
    ) -> ToolExecutionResult:
        """Execute a single tool call.
        
        Args:
            tool_call_id: Unique identifier for this tool call
            function_name: Name of the function to execute
            arguments: Function arguments
            
        Returns:
            ToolExecutionResult with execution details
        """
        start_time = asyncio.get_event_loop().time()
        
        # Validate tool exists
        if function_name not in self._available_tools:
            error_msg = f"Unknown tool function: {function_name}"
            logger.error(error_msg)
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            return ToolExecutionResult(
                tool_call_id=tool_call_id,
                function_name=function_name,
                success=False,
                error=error_msg,
                execution_time_ms=execution_time
            )
        
        tool = self._available_tools[function_name]
        
        try:
            logger.info(f"Executing tool call {tool_call_id}: {function_name} with args {arguments}")
            
            # Execute the tool
            result = await tool.execute_async(arguments)
            
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            logger.info(f"Tool call {tool_call_id} completed successfully in {execution_time:.1f}ms")
            
            return ToolExecutionResult(
                tool_call_id=tool_call_id,
                function_name=function_name,
                success=True,
                result=result,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            error_msg = str(e)
            logger.error(f"Tool call {tool_call_id} failed after {execution_time:.1f}ms: {error_msg}", exc_info=True)
            
            return ToolExecutionResult(
                tool_call_id=tool_call_id,
                function_name=function_name,
                success=False,
                error=error_msg,
                execution_time_ms=execution_time
            )
    
    async def execute_tool_calls(
        self, 
        tool_calls: List[Dict[str, Any]]
    ) -> List[ToolExecutionResult]:
        """Execute multiple tool calls concurrently.
        
        Args:
            tool_calls: List of tool call dictionaries with id, type, and function fields
            
        Returns:
            List of ToolExecutionResult objects in the same order as input
        """
        if not tool_calls:
            return []
        
        logger.info(f"Executing {len(tool_calls)} tool calls")
        
        # Create tasks for concurrent execution
        tasks = []
        for tool_call in tool_calls:
            if tool_call.get("type") != "function":
                logger.warning(f"Skipping non-function tool call: {tool_call}")
                continue
            
            function_data = tool_call.get("function", {})
            function_name = function_data.get("name")
            arguments_str = function_data.get("arguments", "{}")
            
            # Parse arguments
            try:
                arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool call arguments: {arguments_str}")
                # Create a failed result for invalid JSON
                async def create_error_result():
                    return ToolExecutionResult(
                        tool_call_id=tool_call.get("id", str(uuid.uuid4())),
                        function_name=function_name or "unknown",
                        success=False,
                        error=f"Invalid JSON arguments: {str(e)}"
                    )
                
                tasks.append(create_error_result())
                continue
            
            # Create execution task
            task = self.execute_tool_call(
                tool_call_id=tool_call.get("id", str(uuid.uuid4())),
                function_name=function_name,
                arguments=arguments
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions from gather
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Tool execution task {i} raised exception: {result}")
                # Create an error result
                tool_call = tool_calls[i] if i < len(tool_calls) else {}
                processed_results.append(ToolExecutionResult(
                    tool_call_id=tool_call.get("id", str(uuid.uuid4())),
                    function_name=tool_call.get("function", {}).get("name", "unknown"),
                    success=False,
                    error=f"Task execution failed: {str(result)}"
                ))
            else:
                processed_results.append(result)
        
        logger.info(f"Completed {len(processed_results)} tool executions")
        return processed_results
    
    def has_available_tools(self) -> bool:
        """Check if any tools are available for execution."""
        return len(self._available_tools) > 0
    
    def get_collection_path(self) -> Optional[str]:
        """Get the current RAG collection path."""
        return self.rag_collection_path
    
    def update_collection_path(self, new_path: str):
        """Update the RAG collection path and reinitialize tools.
        
        Args:
            new_path: New path to the RAG collection source directory
        """
        logger.info(f"Updating tool executor collection path from {self.rag_collection_path} to {new_path}")
        self.rag_collection_path = new_path
        self._available_tools.clear()
        self._initialize_tools()


# Global instance for the server to use
_global_tool_executor: Optional[ToolExecutor] = None


def get_tool_executor(rag_collection_path: Optional[str] = None) -> ToolExecutor:
    """Get or create the global tool executor instance.
    
    Args:
        rag_collection_path: Path to update the executor with (if provided)
        
    Returns:
        The global ToolExecutor instance
    """
    global _global_tool_executor
    
    # Create if doesn't exist
    if _global_tool_executor is None:
        _global_tool_executor = ToolExecutor(rag_collection_path)
    
    # Update path if provided and different
    elif rag_collection_path and rag_collection_path != _global_tool_executor.get_collection_path():
        _global_tool_executor.update_collection_path(rag_collection_path)
    
    return _global_tool_executor


def clear_tool_executor():
    """Clear the global tool executor instance."""
    global _global_tool_executor
    _global_tool_executor = None

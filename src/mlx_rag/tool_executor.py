"""
LangChain-integrated Tool execution framework for MLX-RAG agentic capabilities.

This module manages the execution of agentic tools using LangChain's framework
while maintaining OpenAI API compatibility and secure operation.
"""

import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import asyncio
from functools import wraps

from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.agent import Agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.schema import AgentAction, AgentFinish

from mlx_rag.agentic_tools import (
    AgenticTool, 
    ListDirectoryTool, 
    ReadFileTool, 
    SearchFilesTool, 
    WriteFileTool, 
    EditFileTool,
    LangChainToolFactory,
    create_langchain_tools,
    ToolExecutionResult
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
            tool_result = await tool.execute(**arguments)
            
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            logger.info(f"Tool call {tool_call_id} completed in {execution_time:.1f}ms")
            
            return ToolExecutionResult(
                tool_call_id=tool_call_id,
                function_name=function_name,
                success=tool_result.success,
                result=tool_result.result,
                error=tool_result.error,
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
            
            # Parse arguments with robust JSON parsing
            try:
                arguments = self._parse_arguments_robust(arguments_str) if isinstance(arguments_str, str) else arguments_str
            except (json.JSONDecodeError, ValueError) as e:
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
    
    def _parse_arguments_robust(self, arguments_str: str) -> Dict[str, Any]:
        """Parse JSON arguments with robust handling of unescaped quotes and control characters.
        
        This method attempts multiple strategies to parse JSON that may contain
        unescaped single quotes, literal newlines, or other common formatting issues.
        
        Args:
            arguments_str: JSON string to parse
            
        Returns:
            Parsed dictionary
            
        Raises:
            json.JSONDecodeError: If all parsing strategies fail
        """
        import re
        
        # Strategy 1: Try direct JSON parsing first
        try:
            return json.loads(arguments_str)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Fix literal newlines and control characters
        try:
            # Replace literal newlines with escaped newlines
            fixed_str = arguments_str.replace('\n', '\\n')
            fixed_str = fixed_str.replace('\r', '\\r')
            fixed_str = fixed_str.replace('\t', '\\t')
            
            return json.loads(fixed_str)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Fix literal newlines AND unescaped single quotes
        try:
            # Start with the string from Strategy 2
            fixed_str = arguments_str.replace('\n', '\\n')
            fixed_str = fixed_str.replace('\r', '\\r')
            fixed_str = fixed_str.replace('\t', '\\t')
            
            # Find string values (content between double quotes) and escape single quotes within them
            def escape_quotes_in_strings(match):
                content = match.group(1)
                # Escape single quotes that aren't already escaped
                escaped = re.sub(r"(?<!\\\\)'", r"\\'", content)
                return f'"{escaped}"'
            
            # Apply escaping to content within double quotes
            # This regex handles escaped quotes within the string content
            fixed_str = re.sub(r'"([^"\\\\]*(?:\\\\.[^"\\\\]*)*)"', escape_quotes_in_strings, fixed_str)
            
            return json.loads(fixed_str)
        except (json.JSONDecodeError, re.error):
            pass
        
        # Strategy 4: More aggressive fixing - handle various quote issues
        try:
            # Start with control character fixes
            fixed = arguments_str.replace('\n', '\\n')
            fixed = fixed.replace('\r', '\\r')
            fixed = fixed.replace('\t', '\\t')
            
            # Handle cases where single quotes are used instead of double quotes for string values
            # But be careful not to break already-quoted content
            
            # First, temporarily replace already-escaped single quotes to protect them
            temp_placeholder = "__TEMP_ESCAPED_QUOTE__"
            fixed = fixed.replace("\\'", temp_placeholder)
            
            # Now replace unescaped single quotes with escaped ones
            fixed = fixed.replace("'", "\\'")            
            
            # Restore the originally escaped quotes
            fixed = fixed.replace(temp_placeholder, "\\'")            
            
            return json.loads(fixed)
        except (json.JSONDecodeError, re.error):
            pass
        
        # Strategy 5: Try to parse as if it's a Python literal (last resort)
        try:
            import ast
            # This is dangerous but might work for simple cases
            # Convert the JSON-like string to a Python literal
            python_like = arguments_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')
            result = ast.literal_eval(python_like)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError):
            pass
        
        # If all strategies fail, raise a descriptive error
        # Try to identify the specific issue
        error_details = []
        if '\n' in arguments_str:
            error_details.append('contains literal newlines (use \\n instead)')
        if "'" in arguments_str:
            error_details.append('contains unescaped single quotes')
        
        error_msg = f"Could not parse JSON arguments with any strategy: {arguments_str[:100]}{'...' if len(arguments_str) > 100 else ''}"
        if error_details:
            error_msg += f" Issues detected: {', '.join(error_details)}"
            
        raise json.JSONDecodeError(error_msg, arguments_str, 0)


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


class LangChainToolExecutor:
    """LangChain-integrated tool executor with agent capabilities."""
    
    def __init__(self, rag_collection_path: Optional[str] = None, use_memory: bool = False):
        """Initialize the LangChain tool executor.
        
        Args:
            rag_collection_path: Path to the RAG collection source directory
            use_memory: Whether to use conversation memory for the agent
        """
        self.rag_collection_path = rag_collection_path
        self.use_memory = use_memory
        self._langchain_tools: List[BaseTool] = []
        self._memory = ConversationBufferMemory(return_messages=True) if use_memory else None
        self._agent_executor = None
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize LangChain tools based on the collection path."""
        if not self.rag_collection_path:
            logger.warning("No RAG collection path provided, LangChain tools will not be available")
            return
        
        # Validate collection path exists and is accessible
        collection_path = Path(self.rag_collection_path)
        if not collection_path.exists():
            logger.error(f"RAG collection path does not exist: {self.rag_collection_path}")
            return
        
        if not collection_path.is_dir():
            logger.error(f"RAG collection path is not a directory: {self.rag_collection_path}")
            return
        
        try:
            # Create LangChain tools
            self._langchain_tools = create_langchain_tools(self.rag_collection_path)
            logger.info(f"Initialized {len(self._langchain_tools)} LangChain tools for collection: {self.rag_collection_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangChain tools: {e}")
            self._langchain_tools.clear()
    
    def get_langchain_tools(self) -> List[BaseTool]:
        """Get the list of LangChain tools."""
        return self._langchain_tools.copy()
    
    def get_tools_for_openai_request(self) -> List[Dict[str, Any]]:
        """Get tools formatted for OpenAI chat completion request."""
        tools = []
        for tool in self._langchain_tools:
            # Convert LangChain tool to OpenAI format
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            
            # Extract parameters from args_schema if available
            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema = tool.args_schema.schema()
                if 'properties' in schema:
                    tool_def["function"]["parameters"]["properties"] = schema['properties']
                if 'required' in schema:
                    tool_def["function"]["parameters"]["required"] = schema['required']
            
            tools.append(tool_def)
        
        return tools
    
    def create_react_agent(self, llm_wrapper=None) -> Optional[AgentExecutor]:
        """Create a ReAct agent with the available tools.
        
        Args:
            llm_wrapper: LLM wrapper for the agent (optional)
            
        Returns:
            AgentExecutor if successful, None otherwise
        """
        if not self._langchain_tools:
            logger.warning("No tools available for creating ReAct agent")
            return None
        
        if not llm_wrapper:
            logger.warning("No LLM wrapper provided for ReAct agent")
            return None
        
        try:
            # Create ReAct prompt template
            react_prompt = PromptTemplate.from_template(
                """You are a helpful assistant that can interact with files in a RAG collection.
                
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}"""
            )
            
            # Create the agent
            agent = create_react_agent(
                llm=llm_wrapper,
                tools=self._langchain_tools,
                prompt=react_prompt
            )
            
            # Create agent executor
            self._agent_executor = AgentExecutor(
                agent=agent,
                tools=self._langchain_tools,
                memory=self._memory,
                verbose=True,
                max_iterations=10,
                handle_parsing_errors=True
            )
            
            logger.info("Created ReAct agent with file system tools")
            return self._agent_executor
            
        except Exception as e:
            logger.error(f"Failed to create ReAct agent: {e}")
            return None
    
    async def execute_with_agent(self, query: str, llm_wrapper=None) -> Dict[str, Any]:
        """Execute a query using the LangChain agent.
        
        Args:
            query: User query to execute
            llm_wrapper: LLM wrapper for the agent
            
        Returns:
            Dict with execution results
        """
        if not self._agent_executor and llm_wrapper:
            self._agent_executor = self.create_react_agent(llm_wrapper)
        
        if not self._agent_executor:
            return {
                "success": False,
                "error": "No agent executor available",
                "result": None
            }
        
        try:
            logger.info(f"Executing query with LangChain agent: {query}")
            
            # Execute the query
            result = await self._agent_executor.ainvoke({"input": query})
            
            return {
                "success": True,
                "result": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", []),
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error executing query with agent: {e}")
            return {
                "success": False,
                "error": str(e),
                "result": None
            }
    
    def execute_single_tool(
        self, 
        tool_name: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a single tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool arguments
            
        Returns:
            Dict with execution results
        """
        # Find the tool
        tool = None
        for t in self._langchain_tools:
            if t.name == tool_name:
                tool = t
                break
        
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "result": None
            }
        
        try:
            logger.info(f"Executing single tool: {tool_name} with args: {kwargs}")
            
            # Execute the tool
            result = tool.run(**kwargs)
            
            return {
                "success": True,
                "result": result,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "result": None
            }
    
    async def execute_single_tool_async(
        self, 
        tool_name: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a single tool by name asynchronously.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool arguments
            
        Returns:
            Dict with execution results
        """
        # Find the tool
        tool = None
        for t in self._langchain_tools:
            if t.name == tool_name:
                tool = t
                break
        
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "result": None
            }
        
        try:
            logger.info(f"Executing single tool async: {tool_name} with args: {kwargs}")
            
            # Execute the tool asynchronously
            result = await tool.arun(**kwargs)
            
            return {
                "success": True,
                "result": result,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name} async: {e}")
            return {
                "success": False,
                "error": str(e),
                "result": None
            }
    
    def has_available_tools(self) -> bool:
        """Check if any tools are available for execution."""
        return len(self._langchain_tools) > 0
    
    def get_collection_path(self) -> Optional[str]:
        """Get the current RAG collection path."""
        return self.rag_collection_path
    
    def update_collection_path(self, new_path: str):
        """Update the RAG collection path and reinitialize tools.
        
        Args:
            new_path: New path to the RAG collection source directory
        """
        logger.info(f"Updating LangChain tool executor collection path from {self.rag_collection_path} to {new_path}")
        self.rag_collection_path = new_path
        self._langchain_tools.clear()
        self._agent_executor = None
        self._initialize_tools()


# Global instances
_global_langchain_tool_executor: Optional[LangChainToolExecutor] = None


def get_langchain_tool_executor(rag_collection_path: Optional[str] = None) -> LangChainToolExecutor:
    """Get or create the global LangChain tool executor instance.
    
    Args:
        rag_collection_path: Path to update the executor with (if provided)
        
    Returns:
        The global LangChainToolExecutor instance
    """
    global _global_langchain_tool_executor
    
    # Create if doesn't exist
    if _global_langchain_tool_executor is None:
        _global_langchain_tool_executor = LangChainToolExecutor(rag_collection_path)
    
    # Update path if provided and different
    elif rag_collection_path and rag_collection_path != _global_langchain_tool_executor.get_collection_path():
        _global_langchain_tool_executor.update_collection_path(rag_collection_path)
    
    return _global_langchain_tool_executor


def clear_langchain_tool_executor():
    """Clear the global LangChain tool executor instance."""
    global _global_langchain_tool_executor
    _global_langchain_tool_executor = None

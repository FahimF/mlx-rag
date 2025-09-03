import os
import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model
from typing import Any, Dict, Type, List, Optional
import traceback
import re
import json

# model = 'Qwen2.5-Coder-7B-Instruct-MLX-4bit';
model = 'Qwen3-Coder-30B-A3B-Instruct-4bit';

def parse_json_robust(json_str: str) -> dict:
    """Parse JSON with robust handling of unescaped quotes and control characters."""
    
    # Strategy 1: Try direct JSON parsing first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Fix literal newlines and control characters
    try:
        fixed_str = json_str.replace('\n', '\\n')
        fixed_str = fixed_str.replace('\r', '\\r') 
        fixed_str = fixed_str.replace('\t', '\\t')
        return json.loads(fixed_str)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Fix literal newlines AND unescaped single quotes
    try:
        fixed_str = json_str.replace('\n', '\\n')
        fixed_str = fixed_str.replace('\r', '\\r')
        fixed_str = fixed_str.replace('\t', '\\t')
        
        # First, temporarily replace already-escaped single quotes to protect them
        temp_placeholder = "__TEMP_ESCAPED_QUOTE__"
        fixed_str = fixed_str.replace("\\'", temp_placeholder)
        
        # Now replace unescaped single quotes with escaped ones
        fixed_str = fixed_str.replace("'", "\\'")            
        
        # Restore the originally escaped quotes
        fixed_str = fixed_str.replace(temp_placeholder, "\\'")            
        
        return json.loads(fixed_str)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: More advanced regex-based quote fixing within string values
    try:
        fixed_str = json_str.replace('\n', '\\n')
        fixed_str = fixed_str.replace('\r', '\\r')
        fixed_str = fixed_str.replace('\t', '\\t')
        
        # Use regex to find and fix quotes within string values more precisely
        import re
        
        # Pattern to find content within double quotes that might contain unescaped single quotes
        def fix_quotes_in_match(match):
            full_match = match.group(0)  # The full "..." including quotes
            content = match.group(1)     # Just the content between quotes
            
            # Escape any unescaped single quotes in the content
            # But preserve already escaped ones
            fixed_content = re.sub(r"(?<!\\\\)'", r"\\'", content)
            return f'"{fixed_content}"'
        
        # Apply the fix to all string values
        fixed_str = re.sub(r'"([^"]*(?:\\\\.[^"]*)*?)"', fix_quotes_in_match, fixed_str)
        
        return json.loads(fixed_str)
    except (json.JSONDecodeError, re.error):
        pass
    
    # Strategy 5: Character by character analysis and reconstruction
    try:
        # Try a more aggressive approach - rebuild the JSON
        rebuilt = json_str
        rebuilt = rebuilt.replace('\n', '\\n')
        rebuilt = rebuilt.replace('\r', '\\r')
        rebuilt = rebuilt.replace('\t', '\\t')
        
        # Replace ALL single quotes with escaped single quotes
        # This is aggressive but might work
        rebuilt = rebuilt.replace("'", "\\'") 
        
        return json.loads(rebuilt)
        
    except json.JSONDecodeError:
        pass
    
    # If all strategies fail, raise the original error with more info
    error_details = []
    if '\n' in json_str:
        error_details.append('contains literal newlines')
    if "'" in json_str:
        error_details.append('contains unescaped single quotes')
    if '\t' in json_str:
        error_details.append('contains literal tabs')
    
    error_msg = f"Could not parse JSON: {json_str[:100]}{'...' if len(json_str) > 100 else ''}"
    if error_details:
        error_msg += f" ({', '.join(error_details)})"
        
    raise json.JSONDecodeError(error_msg, json_str, 0)

# Define Pydantic models at module level to avoid scope issues
class ListDirectoryArgs(BaseModel):
    path: Optional[str] = Field(default="", description="Relative path within the collection to list")
    include_hidden: Optional[bool] = Field(default=False, description="Whether to include hidden files")
    pattern: Optional[str] = Field(default="", description="File pattern to match")

class ReadFileArgs(BaseModel):
    path: str = Field(description="Relative path to the file within the collection")
    start_line: Optional[int] = Field(default=1, description="Start reading from this line number")
    end_line: Optional[int] = Field(default=None, description="Stop reading at this line number")

class SearchFilesArgs(BaseModel):
    query: str = Field(description="Text to search for")
    path: Optional[str] = Field(default="", description="Relative path to search within")
    file_pattern: Optional[str] = Field(default="", description="File pattern to search within")
    case_sensitive: Optional[bool] = Field(default=False, description="Whether the search should be case sensitive")
    max_results: Optional[int] = Field(default=50, description="Maximum number of results to return")

class WriteFileArgs(BaseModel):
    path: str = Field(description="Relative path to the file within the collection")
    content: str = Field(description="Content to write to the file")
    create_dirs: Optional[bool] = Field(default=False, description="Whether to create parent directories")

class EditFileArgs(BaseModel):
    path: str = Field(description="Relative path to the file within the collection")
    start_line: int = Field(description="Start line number to replace (1-based)")
    end_line: int = Field(description="End line number to replace (1-based, inclusive)")
    new_content: str = Field(description="New content to replace the selected lines with")

class ManualAgent:
    """
    Simple manual agent that doesn't rely on LangChain's streaming mechanisms.
    """
    def __init__(self, llm, tools: List[StructuredTool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.tool_descriptions = self._create_tool_descriptions()
    
    def _create_tool_descriptions(self) -> str:
        descriptions = []
        for tool_name, tool in self.tools.items():
            # Get the tool's schema
            schema = tool.args_schema.model_json_schema() if tool.args_schema else {}
            properties = schema.get('properties', {})
            
            param_desc = []
            for param_name, param_info in properties.items():
                param_type = param_info.get('type', 'string')
                param_description = param_info.get('description', '')
                param_desc.append(f"  - {param_name} ({param_type}): {param_description}")
            
            params_str = "\n".join(param_desc) if param_desc else "  - No parameters required"
            descriptions.append(f"**{tool_name}**: {tool.description}\nParameters:\n{params_str}")
        
        return "\n\n".join(descriptions)
    
    def _extract_json_parameters(self, text: str):
        """Extract JSON parameters with proper brace matching."""
        # Find the start of parameters
        params_start = re.search(r'PARAMETERS:\s*', text)
        if not params_start:
            return None
        
        start_pos = params_start.end()
        if start_pos >= len(text) or text[start_pos] != '{':
            return None
        
        # Count braces to find the complete JSON object
        brace_count = 0
        in_string = False
        escape_next = False
        end_pos = start_pos
        
        for i, char in enumerate(text[start_pos:], start_pos):
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\' and in_string:
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
        
        if brace_count == 0:
            json_str = text[start_pos:end_pos]
            # Create a match-like object
            class JSONMatch:
                def group(self, n):
                    return json_str if n == 1 else None
            return JSONMatch()
        
        return None
    
    def run(self, user_input: str, max_iterations: int = 5) -> str:
        """
        Run the agent with manual tool calling logic.
        """
        conversation_history = []
        
        system_prompt = f"""You are a helpful assistant for managing a RAG collection. You have access to the following tools:

{self.tool_descriptions}

When you need to use a tool, respond in this exact format:
TOOL_CALL: tool_name
PARAMETERS: {{"param1": "value1", "param2": "value2"}}

If you don't need to use any tools, just provide a direct answer.

User question: {user_input}"""

        conversation_history.append(system_prompt)
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Get LLM response
            try:
                current_prompt = "\n\n".join(conversation_history)
                response = self.llm.invoke(current_prompt)
                llm_response = response.content.strip()
                print(f"LLM Response: {llm_response}")
                
                # Check if LLM wants to use a tool
                tool_call_match = re.search(r'TOOL_CALL:\s*(\w+)', llm_response)
                
                # Use a more sophisticated regex to capture complete JSON with nested braces
                params_match = self._extract_json_parameters(llm_response)
                
                if tool_call_match:
                    tool_name = tool_call_match.group(1)
                    
                    if tool_name in self.tools:
                        # Parse parameters
                        try:
                            if params_match:
                                params_str = params_match.group(1)
                                params = parse_json_robust(params_str)
                            else:
                                params = {}
                            
                            print(f"Executing tool: {tool_name} with params: {params}")
                            
                            # Execute the tool
                            tool_result = self.tools[tool_name].run(params)
                            print(f"Tool result: {tool_result}")
                            
                            # Add tool execution to conversation
                            conversation_history.append(f"Assistant: {llm_response}")
                            conversation_history.append(f"Tool Result: {tool_result}")
                            conversation_history.append("Please provide your final answer based on the tool result:")
                            
                        except Exception as tool_error:
                            error_msg = f"Error executing tool {tool_name}: {tool_error}"
                            print(error_msg)
                            conversation_history.append(f"Assistant: {llm_response}")
                            conversation_history.append(f"Tool Error: {error_msg}")
                            conversation_history.append("Please provide an answer based on the available information:")
                    else:
                        error_msg = f"Unknown tool: {tool_name}. Available tools: {list(self.tools.keys())}"
                        print(error_msg)
                        conversation_history.append(f"Assistant: {llm_response}")
                        conversation_history.append(f"Error: {error_msg}")
                        conversation_history.append("Please use one of the available tools or provide a direct answer:")
                else:
                    # No tool call detected, this should be the final answer
                    print(f"Final answer: {llm_response}")
                    return llm_response
                    
            except Exception as e:
                print(f"Error in iteration {iteration + 1}: {e}")
                return f"Error occurred: {e}"
        
        return "Max iterations reached without a final answer."

def run_dynamic_tool_agent():
    """
    Fetches tool schemas from a server's /tools endpoint,
    converts them to LangChain format, and runs a manual agent.
    """
    os.environ["OPENAI_API_KEY"] = "sk-not-required"
    server_base_url = "http://localhost:8000"
    tools_endpoint_url = f"{server_base_url}/v1/tools"
    
    print(f"Attempting to fetch tools from {tools_endpoint_url}")
    try:
        response = requests.get(tools_endpoint_url)
        response.raise_for_status()
        raw_tools_data = response.json()
        raw_tools_list = raw_tools_data.get("tools", [])
        if not raw_tools_list:
            print("No tools found on the server.")
            return
        print(f"Successfully fetched {len(raw_tools_list)} tool schemas.")
                
        langchain_tools = []
        for raw_tool in raw_tools_list:
            try:
                raw_function_schema = raw_tool["function"]
                tool_name = raw_function_schema["name"]
                tool_description = raw_function_schema.get("description", f"Tool: {tool_name}")
                
                # Create real function that calls the server
                def create_real_func(name: str, server_url: str):
                    def real_tool_func(**kwargs: Any) -> str:
                        print(f"\n[Tool Call] Executing tool: '{name}' with args: {kwargs}")
                        try:
                            # Make actual API call to your server's execute endpoint
                            tool_endpoint = f"{server_url}/v1/tools/execute"
                            
                            # Format the request according to your server's expectations
                            request_data = {
                                "function_name": name,
                                "arguments": kwargs,  # Send as dict, server will handle JSON conversion
                                "tool_call_id": f"call_{name}_{hash(str(kwargs)) % 10000}"
                            }
                            
                            print(f"Calling {tool_endpoint} with data: {request_data}")
                            response = requests.post(tool_endpoint, json=request_data)
                            response.raise_for_status()
                            result_data = response.json()
                            
                            # Extract the actual result based on your server's response format
                            if result_data.get('success', False):
                                return result_data.get('result', 'Tool executed successfully but no result returned')
                            else:
                                error_msg = result_data.get('error', 'Unknown error occurred')
                                return f"Tool execution failed: {error_msg}"
                                
                        except requests.exceptions.RequestException as e:
                            return f"HTTP error executing tool '{name}': {e}"
                        except Exception as e:
                            return f"Error executing tool '{name}': {e}"
                    
                    real_tool_func.__name__ = name 
                    return real_tool_func
                
                real_func = create_real_func(tool_name, server_base_url)
                
                # Get the appropriate Pydantic model for this tool
                tool_model_map = {
                    "list_directory": ListDirectoryArgs,
                    "read_file": ReadFileArgs,
                    "search_files": SearchFilesArgs,
                    "write_file": WriteFileArgs,
                    "edit_file": EditFileArgs
                }
                
                pydantic_model = tool_model_map.get(tool_name)
                if not pydantic_model:
                    print(f"Unknown tool: {tool_name}, skipping...")
                    continue
                
                print(f"Using predefined model for {tool_name}: {pydantic_model}")
                
                # Create StructuredTool
                langchain_tool = StructuredTool.from_function(
                    func=real_func,
                    name=tool_name,
                    description=tool_description,
                    args_schema=pydantic_model
                )
                langchain_tools.append(langchain_tool)
                print(f"Successfully created tool: {tool_name}")
                
            except Exception as tool_error:
                print(f"Error creating tool {raw_tool.get('function', {}).get('name', 'Unknown')}: {tool_error}")
                continue
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching tools from the server: {e}")
        print(f"Please ensure your server is running and the '/tools' endpoint is accessible at {tools_endpoint_url}")
        return
    except Exception as e:
        print(f"Error processing tool schemas: {e}")
        traceback.print_exc()
        return

    # Initialize LLM
    try:
        print(f"Initializing LLM with model: {model}")
        llm = ChatOpenAI(
            model=model,
            base_url=f"{server_base_url}/v1",
            temperature=0,
            api_key="sk-not-required",
            streaming=False,
            max_retries=1,
            max_completion_tokens=8000,
            request_timeout=60,
        )
        
        # Test the LLM connection
        print("Testing LLM connection...")
        test_response = llm.invoke("Hello")
        print(f"✓ LLM test successful: {test_response.content[:50]}...")
        
    except Exception as llm_error:
        print(f"✗ LLM initialization failed: {llm_error}")
        traceback.print_exc()
        return

    # Use our manual agent instead of LangChain's problematic agents
    print("\n" + "="*50)
    print("Running manual agent...")
    print("="*50)
    
    try:
        manual_agent = ManualAgent(llm, langchain_tools)
        user_query = "Modify /Users/fahim/Code/Flutter/Booker/lib/main.dart to space out the action buttons vertically and display the picked books folder and Calibre library path next to the relevant buttons."
        
        print(f"Running query: {user_query}")
        result = manual_agent.run(user_query)
        print(f"\n✓ Final result: {result}")
        
    except Exception as e:
        print(f"✗ Error running manual agent: {e}")
        traceback.print_exc()

def create_pydantic_model_from_openai_schema(parameters_schema: Dict[str, Any], model_name: str) -> Type[BaseModel]:
    """
    This function is now deprecated - we use predefined models instead.
    """
    # This function is no longer used but kept for compatibility
    class DummyArgs(BaseModel):
        pass
    return DummyArgs

def get_python_type_from_json_schema(schema: Dict[str, Any]) -> Type:
    """
    Convert JSON schema type to Python type.
    """
    schema_type = schema.get("type", "string")
    
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    
    return type_mapping.get(schema_type, str)

if __name__ == "__main__":
    run_dynamic_tool_agent()
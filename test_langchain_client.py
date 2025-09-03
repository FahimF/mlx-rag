#!/usr/bin/env python3
"""
LangChain client test for mlx-rag server.

This test uses LangChain's OpenAI integration to communicate with the mlx-rag server
at http://localhost:8000, testing the file modification workflow with proper loop detection.
"""

import asyncio
import logging
from typing import Dict, Any, List
from collections import Counter
import json

# LangChain imports - using community package for OpenAI compatibility
import httpx
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler

# Simple OpenAI-compatible LLM wrapper
class MLXChatWrapper:
    """Simple wrapper to make requests to MLX-RAG server with LangChain-like interface."""
    
    def __init__(self, base_url: str, api_key: str = "dummy", model: str = "Qwen3-Coder-30B-A3B-Instruct-4bit", **kwargs):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.temperature = kwargs.get('temperature', 0.3)
        self.max_tokens = kwargs.get('max_tokens', 2000)
        
    async def ainvoke(self, messages, callbacks=None):
        """Async invoke method compatible with LangChain interface."""
        
        # Convert LangChain messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                openai_messages.append({"role": "assistant", "content": msg.content})
        
        # Get tools from server
        async with httpx.AsyncClient() as client:
            try:
                # Get available tools
                tools_response = await client.get(f"{self.base_url}/v1/tools")
                tools = tools_response.json().get("tools", [])
                
                # Prepare request
                request_data = {
                    "model": self.model,
                    "messages": openai_messages,
                    "tools": tools,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                }
                
                # Make request
                print(f"[DEBUG] Making request to {self.base_url}/v1/chat/completions")
                print(f"[DEBUG] Model: {self.model}")
                print(f"[DEBUG] Tools count: {len(tools)}")
                
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                
                print(f"[DEBUG] Response status: {response.status_code}")
                
                if response.status_code != 200:
                    error_text = response.text
                    print(f"[DEBUG] Error response: {error_text}")
                    raise Exception(f"Request failed with status {response.status_code}: {error_text}")
                
                data = response.json()
                print(f"[DEBUG] Response received, data keys: {list(data.keys()) if data else 'None'}")
                
                # Handle tool calls if present
                if 'choices' in data and len(data['choices']) > 0:
                    choice = data['choices'][0]
                    message = choice.get('message', {})
                    
                    # Check for tool calls
                    if 'tool_calls' in message and message['tool_calls']:
                        if callbacks:
                            for callback in callbacks:
                                for tool_call in message['tool_calls']:
                                    if hasattr(callback, 'on_tool_start'):
                                        callback.on_tool_start(
                                            {"name": tool_call['function']['name']},
                                            tool_call['function']['arguments']
                                        )
                        
                        # Execute tool calls
                        tool_results = []
                        for tool_call in message['tool_calls']:
                            tool_request = {
                                "tool_call_id": tool_call['id'],
                                "function_name": tool_call['function']['name'],
                                "arguments": tool_call['function']['arguments']
                            }
                            
                            tool_response = await client.post(
                                f"{self.base_url}/v1/tools/execute",
                                json=tool_request,
                                headers={"Content-Type": "application/json"}
                            )
                            
                            if tool_response.status_code == 200:
                                tool_result = tool_response.json()
                                result_content = tool_result.get('result', tool_result.get('error', ''))
                                if isinstance(result_content, (dict, list)):
                                    result_content = json.dumps(result_content)
                                
                                tool_results.append({
                                    "tool_call_id": tool_call['id'],
                                    "role": "tool",
                                    "content": result_content
                                })
                                
                                if callbacks:
                                    for callback in callbacks:
                                        if hasattr(callback, 'on_tool_end'):
                                            callback.on_tool_end(result_content)
                        
                        # Continue conversation with tool results
                        continue_messages = openai_messages + [message] + tool_results
                        continue_request = {
                            "model": self.model,
                            "messages": continue_messages,
                            "tools": tools,  # Include tools in follow-up
                            "max_tokens": self.max_tokens,
                            "temperature": self.temperature
                        }
                        
                        continue_response = await client.post(
                            f"{self.base_url}/v1/chat/completions",
                            json=continue_request,
                            headers={"Content-Type": "application/json"}
                        )
                        
                        if continue_response.status_code == 200:
                            continue_data = continue_response.json()
                            if 'choices' in continue_data and len(continue_data['choices']) > 0:
                                final_message = continue_data['choices'][0]['message']
                                return AIMessage(content=final_message.get('content', ''))
                    
                    # No tool calls, return regular response
                    return AIMessage(content=message.get('content', ''))
                
                return AIMessage(content="No response received")
                
            except Exception as e:
                error_msg = f"Exception in MLXChatWrapper: {str(e)}"
                print(f"🚨 {error_msg}")
                return AIMessage(content=error_msg)
    
    async def astream(self, messages):
        """Simple streaming implementation."""
        result = await self.ainvoke(messages)
        # Split into chunks for streaming simulation
        content = result.content
        chunk_size = 50
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i+chunk_size]
            yield AIMessage(content=chunk)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoopDetectionCallbackHandler(BaseCallbackHandler):
    """Callback handler to detect and track tool usage patterns."""
    
    def __init__(self, max_same_file_reads: int = 3):
        self.max_same_file_reads = max_same_file_reads
        self.tool_calls = []
        self.read_counter = Counter()
        self.consecutive_reads = 0
        self.loop_detected = False
        
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when tool starts execution."""
        tool_name = serialized.get("name", "unknown")
        
        # Try to extract file path from input for read operations
        file_path = "unknown"
        if tool_name == "read_file" and "main.dart" in input_str:
            file_path = "main.dart"
            
        self.tool_calls.append({
            "tool": tool_name,
            "input": input_str,
            "file_path": file_path
        })
        
        # Track reads
        if tool_name == "read_file":
            self.read_counter[file_path] += 1
            self.consecutive_reads += 1
            
            if self.read_counter[file_path] > self.max_same_file_reads:
                self.loop_detected = True
                logger.warning(f"🚨 Loop detected: {tool_name} called {self.read_counter[file_path]} times on {file_path}")
        else:
            self.consecutive_reads = 0
            
        print(f"  🔧 Tool: {tool_name} | File: {file_path} | Reads: {self.read_counter[file_path]}")
        
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when tool ends execution."""
        # Check if it was an edit tool
        if "lines_replaced" in output or "bytes_written" in output:
            print(f"  🎉 EDIT DETECTED in tool output!")
            
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of tool usage."""
        return {
            "total_calls": len(self.tool_calls),
            "read_counter": dict(self.read_counter),
            "consecutive_reads": self.consecutive_reads,
            "loop_detected": self.loop_detected,
            "tools_used": [call["tool"] for call in self.tool_calls]
        }

async def test_langchain_client():
    """Test using LangChain client to communicate with mlx-rag server."""
    
    print("🧪 Testing LangChain client with mlx-rag server...")
    print("📡 Server: http://localhost:8000")
    
    # Initialize MLX client wrapper
    client = MLXChatWrapper(
        base_url="http://localhost:8000",
        api_key="dummy",  # Server might not require auth
        model="Qwen3-Coder-30B-A3B-Instruct-4bit",  # The server will use whatever model is loaded
        temperature=0.3,
        max_tokens=2000
    )
    
    # Setup callback handler for loop detection
    callback_handler = LoopDetectionCallbackHandler(max_same_file_reads=2)
    
    print("✅ Initialized LangChain OpenAI client")
    
    # Prepare the messages
    system_message = SystemMessage(content="""
You are a code editor assistant with access to file system tools. Your task is to modify files based on user requests.

When asked to modify files:
1. Find the file using search tools if needed
2. Read the file ONCE to understand its current structure
3. Use edit_file tool to make the requested changes
4. Do NOT read the same file multiple times
5. Make actual file modifications, don't just show code examples

Available tools will be provided automatically. Use them to complete the requested modifications.
    """.strip())
    
    human_message = HumanMessage(content="Modify main.dart file to space out the action buttons vertically and display the picked books folder and Calibre library path next to the relevant buttons.")
    
    messages = [system_message, human_message]
    
    print("\n💬 Sending request to server...")
    print(f"📝 Prompt: {human_message.content}")
    
    try:
        # Make the request with callback for loop detection
        response = await client.ainvoke(
            messages,
            callbacks=[callback_handler]
        )
        
        print("\n📥 Received response from server")
        print(f"📄 Response type: {type(response)}")
        print(f"📄 Full response content: {response.content}")
        
        # Check if response contains tool calls or mentions of tools
        response_lower = response.content.lower()
        if any(tool in response_lower for tool in ['read_file', 'edit_file', 'search_files']):
            print("✅ Response mentions tool usage")
        else:
            print("⚠️ Response doesn't seem to mention tool usage")
            
        # Get callback summary
        summary = callback_handler.get_summary()
        print("\n📊 Tool Usage Summary:")
        print(f"  - Total tool calls: {summary['total_calls']}")
        print(f"  - File read attempts: {summary['read_counter']}")
        print(f"  - Tools used: {summary['tools_used']}")
        print(f"  - Loop detected: {summary['loop_detected']}")
        
        # Check for successful edit indicators
        response_content = response.content
        edit_indicators = ["edited", "modified", "changed", "updated", "lines_replaced"]
        
        found_indicators = [indicator for indicator in edit_indicators 
                          if indicator in response_content.lower()]
        
        if found_indicators:
            print(f"🎉 Edit indicators found: {found_indicators}")
        else:
            print("❌ No clear edit indicators in response")
            
        if summary['loop_detected']:
            print("🚨 LOOP DETECTED - Model got stuck in repetitive tool usage")
        
        return {
            "success": True,
            "response": response.content,
            "tool_summary": summary,
            "edit_detected": bool(found_indicators),
            "loop_detected": summary['loop_detected']
        }
        
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "tool_summary": callback_handler.get_summary()
        }

async def test_langchain_streaming():
    """Test streaming response to see tool calls in real-time."""
    
    print("\n🔄 Testing streaming response...")
    
    client = MLXChatWrapper(
        base_url="http://localhost:8000",
        api_key="dummy",
        model="Qwen3-Coder-30B-A3B-Instruct-4bit",
        temperature=0.3
    )
    
    messages = [
        SystemMessage(content="You are a helpful assistant with file editing tools. Use tools to complete user requests."),
        HumanMessage(content="Please read main.dart and show me its structure.")
    ]
    
    try:
        print("📡 Starting streaming request...")
        
        response_chunks = []
        async for chunk in client.astream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                response_chunks.append(chunk.content)
                print(f"📤 Chunk: {chunk.content}", end="", flush=True)
        
        print(f"\n✅ Streaming complete. Received {len(response_chunks)} chunks.")
        
        full_response = ''.join(response_chunks)
        tool_mentions = sum(1 for tool in ['read_file', 'edit_file', 'search_files'] 
                           if tool in full_response.lower())
        
        print(f"🔧 Tool mentions in response: {tool_mentions}")
        
        return {
            "success": True,
            "chunks": len(response_chunks),
            "tool_mentions": tool_mentions
        }
        
    except Exception as e:
        print(f"❌ Streaming failed: {e}")
        return {"success": False, "error": str(e)}

async def test_with_explicit_tool_instruction():
    """Test with very explicit tool usage instructions."""
    
    print("\n🎯 Testing with explicit tool instructions...")
    
    client = MLXChatWrapper(
        base_url="http://localhost:8000",
        api_key="dummy",
        model="Qwen3-Coder-30B-A3B-Instruct-4bit",
        temperature=0.2  # Lower temperature for more focused responses
    )
    
    # More explicit system message
    system_message = SystemMessage(content="""
You are a code editor. You have access to file system tools. 

IMPORTANT: You MUST use tools to complete requests. Do not just provide code examples.

Workflow for file modification:
1. Use search_files or read_file to find and examine the target file
2. Use edit_file to make the actual changes to the file  
3. Confirm the changes were made

DO NOT:
- Read the same file multiple times
- Provide code examples without making actual edits
- Give up if you can't find the file immediately

Tools available: read_file, edit_file, search_files, list_directory, write_file
    """.strip())
    
    human_message = HumanMessage(content="""
TASK: Modify main.dart file to space out the action buttons vertically and display folder paths.

INSTRUCTIONS:
1. Find main.dart using tools  
2. Read it ONCE to understand the current layout
3. Use edit_file tool to change button layout from horizontal to vertical
4. Add text showing folder paths next to buttons
5. Confirm the edit was successful

DO THIS NOW using the available tools.
    """.strip())
    
    messages = [system_message, human_message]
    
    callback_handler = LoopDetectionCallbackHandler(max_same_file_reads=2)
    
    try:
        print("📤 Sending explicit instruction request...")
        
        response = await client.ainvoke(
            messages,
            callbacks=[callback_handler]
        )
        
        print(f"\n📥 Response received ({len(response.content)} chars)")
        print(f"📝 First 300 chars: {response.content[:300]}...")
        
        # Analysis
        response_lower = response.content.lower()
        
        # Check for tool usage mentions
        tool_usage = {
            'read_file': 'read_file' in response_lower,
            'edit_file': 'edit_file' in response_lower,  
            'search_files': 'search_files' in response_lower
        }
        
        print("\n🔍 Tool Usage Analysis:")
        for tool, mentioned in tool_usage.items():
            print(f"  - {tool}: {'✅ mentioned' if mentioned else '❌ not mentioned'}")
        
        # Check for action vs description
        action_words = ['edited', 'modified', 'changed', 'updated', 'replaced']
        description_words = ['would', 'should', 'could', 'need to', 'you can']
        
        actions = sum(1 for word in action_words if word in response_lower)
        descriptions = sum(1 for word in description_words if word in response_lower)
        
        print(f"\n📊 Response Analysis:")
        print(f"  - Action words: {actions}")
        print(f"  - Description words: {descriptions}")
        print(f"  - Action/Description ratio: {actions/(descriptions+1):.2f}")
        
        summary = callback_handler.get_summary()
        print(f"  - Tool calls detected: {summary['total_calls']}")
        print(f"  - Loop detected: {summary['loop_detected']}")
        
        # Determine if the response indicates successful tool usage
        likely_tool_usage = (
            actions > descriptions and 
            any(tool_usage.values()) and 
            summary['total_calls'] > 0
        )
        
        print(f"\n{'🎉' if likely_tool_usage else '❌'} Likely tool usage: {likely_tool_usage}")
        
        return {
            "success": True,
            "likely_tool_usage": likely_tool_usage,
            "action_words": actions,
            "description_words": descriptions,
            "tool_calls": summary['total_calls'],
            "loop_detected": summary['loop_detected']
        }
        
    except Exception as e:
        print(f"❌ Explicit instruction test failed: {e}")
        return {"success": False, "error": str(e)}

async def main():
    """Run all LangChain client tests."""
    
    print("🚀 Running LangChain client tests for mlx-rag server...\n")
    
    # Test 1: Basic functionality
    result1 = await test_langchain_client()
    
    # Test 2: Streaming (to see real-time tool calls)  
    result2 = await test_langchain_streaming()
    
    # Test 3: Explicit instructions
    result3 = await test_with_explicit_tool_instruction()
    
    # Summary
    print("\n" + "="*60)
    print("🏁 TEST SUMMARY")
    print("="*60)
    
    print(f"Test 1 (Basic): {'✅ Success' if result1.get('success') else '❌ Failed'}")
    if result1.get('success'):
        print(f"  - Edit detected: {result1.get('edit_detected', False)}")
        print(f"  - Loop detected: {result1.get('loop_detected', False)}")
    
    print(f"Test 2 (Streaming): {'✅ Success' if result2.get('success') else '❌ Failed'}")
    if result2.get('success'):
        print(f"  - Chunks received: {result2.get('chunks', 0)}")
        print(f"  - Tool mentions: {result2.get('tool_mentions', 0)}")
    
    print(f"Test 3 (Explicit): {'✅ Success' if result3.get('success') else '❌ Failed'}")
    if result3.get('success'):
        print(f"  - Likely tool usage: {result3.get('likely_tool_usage', False)}")
        print(f"  - Tool calls: {result3.get('tool_calls', 0)}")
        print(f"  - Loop detected: {result3.get('loop_detected', False)}")
    
    print("\n💡 Key Insights:")
    if any(result.get('loop_detected', False) for result in [result1, result3] if result.get('success')):
        print("  - ⚠️ Loop behavior detected - model getting stuck in repetitive reads")
    
    if any(result.get('edit_detected', False) or result.get('likely_tool_usage', False) 
           for result in [result1, result3] if result.get('success')):
        print("  - ✅ Model appears to be using tools for actual edits")
    else:
        print("  - ❌ Model may be providing text responses instead of using tools")

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
LangChain OpenAI client test for mlx-rag server.

This test uses the official langchain-openai integration to communicate with the mlx-rag server
at http://localhost:8000, testing the file modification workflow with proper loop detection.
"""

import asyncio
import logging
from typing import Dict, Any, List
from collections import Counter
import json

# LangChain imports using the official OpenAI integration
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_server_tools():
    """Fetches available tools from the server."""
    import httpx
    async with httpx.AsyncClient() as http_client:
        try:
            tools_response = await http_client.get("http://localhost:8000/v1/tools")
            if tools_response.status_code == 200:
                server_tools = tools_response.json().get('tools', [])
                print(f"📋 Found {len(server_tools)} tools on server")
                return server_tools
            else:
                print(f"⚠️ Could not fetch tools from server: {tools_response.status_code}")
                return []
        except Exception as e:
            print(f"⚠️ Error fetching tools: {e}")
            return []

class ToolUsageCallbackHandler(BaseCallbackHandler):
    """Enhanced callback handler to track tool usage patterns and detect loops."""
    
    def __init__(self, max_same_file_reads: int = 3):
        self.max_same_file_reads = max_same_file_reads
        self.tool_calls = []
        self.read_counter = Counter()
        self.consecutive_reads = 0
        self.loop_detected = False
        self.successful_edits = []
        
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when tool execution starts."""
        tool_name = serialized.get("name", "unknown")
        
        # Extract file path for file operations
        file_path = "unknown"
        if "main.dart" in str(input_str):
            file_path = "main.dart"
        elif "path" in str(input_str):
            # Try to extract path from arguments
            try:
                args = json.loads(str(input_str))
                file_path = args.get("path", "unknown")
            except:
                pass
            
        self.tool_calls.append({
            "tool": tool_name,
            "input": input_str,
            "file_path": file_path
        })
        
        # Track read operations for loop detection
        if tool_name == "read_file":
            self.read_counter[file_path] += 1
            self.consecutive_reads += 1
            
            if self.read_counter[file_path] > self.max_same_file_reads:
                self.loop_detected = True
                logger.warning(f"🚨 Loop detected: {tool_name} called {self.read_counter[file_path]} times on {file_path}")
        else:
            self.consecutive_reads = 0
            
        print(f"  🔧 Tool: {tool_name} | File: {file_path} | Reads: {self.read_counter.get(file_path, 0)}")
        
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when tool execution completes."""
        # Check for successful edit operations
        output_str = str(output)
        if "lines_replaced" in output_str or "bytes_written" in output_str:
            print(f"  🎉 EDIT DETECTED in tool output!")
            try:
                result_data = json.loads(output_str)
                if "lines_replaced" in result_data:
                    self.successful_edits.append({
                        "type": "edit",
                        "lines": result_data["lines_replaced"],
                        "file": result_data.get("file_path", "unknown")
                    })
                elif "bytes_written" in result_data:
                    self.successful_edits.append({
                        "type": "write", 
                        "bytes": result_data["bytes_written"],
                        "file": result_data.get("file_path", "unknown")
                    })
            except:
                self.successful_edits.append({"type": "unknown", "raw": output_str[:100]})
    
    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Called when tool execution fails."""
        print(f"  ❌ Tool error: {error}")
        
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of tool usage."""
        return {
            "total_calls": len(self.tool_calls),
            "read_counter": dict(self.read_counter),
            "consecutive_reads": self.consecutive_reads,
            "loop_detected": self.loop_detected,
            "tools_used": [call["tool"] for call in self.tool_calls],
            "successful_edits": self.successful_edits,
            "unique_tools": list(set(call["tool"] for call in self.tool_calls))
        }

async def test_basic_langchain_openai():
    """Test basic functionality using LangChain OpenAI client."""
    
    print("🧪 Testing with LangChain OpenAI client...")
    print("📡 Server: http://localhost:8000")
    
    # First, get available tools from the server
    server_tools = await get_server_tools()
    
    # Initialize the LangChain OpenAI client pointing to our local server
    client = ChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy",  # mlx-rag server doesn't require real API key
        model="Qwen3-Coder-30B-A3B-Instruct-4bit",  # Use the correct model name
        temperature=0,
        max_tokens=20000,
        timeout=60.0
    )
    
    openai_tools = server_tools
    
    if openai_tools:
        print(f"✅ Loaded {len(openai_tools)} tools to be passed to the server")
    else:
        print("⚠️ No tools available - using client without tools")
    
    # Setup callback handler for monitoring tool usage
    callback_handler = ToolUsageCallbackHandler(max_same_file_reads=2)
    
    print("✅ Initialized LangChain OpenAI client")
    
    # Create the messages for file modification task
    messages = [
        SystemMessage(content="""
You are a code editor assistant with access to file system tools. Your task is to modify files based on user requests.

When asked to modify files:
1. Use search_files or read_file to locate and examine the target file
2. Read the file ONCE to understand its current structure  
3. Use edit_file tool to make the requested changes
4. Do NOT read the same file multiple times
5. Make actual file modifications, don't just show code examples

You have access to these tools: read_file, edit_file, search_files, list_directory, write_file
Use them to complete the requested modifications efficiently.
        """.strip()),
        
        HumanMessage(content="Modify main.dart file to space out the action buttons vertically and display the picked books folder and Calibre library path next to the relevant buttons.")
    ]
    
    print("\n💬 Sending request to server...")
    print(f"📝 Task: Modify main.dart for better button layout")
    
    try:
        # Make the request with callback monitoring
        response = await client.ainvoke(
            messages,
            tools=openai_tools,
            config={"callbacks": [callback_handler]}
        )
        
        print(f"\n📥 Received response ({len(response.content)} chars)")
        print(f"📄 Response preview: {response.content[:300]}...")
        
        # Analyze the response
        response_lower = response.content.lower()
        
        tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []
        
        # Check for tool usage indicators
        tool_indicators = {
            'read_file': 'read_file' in response_lower,
            'edit_file': 'edit_file' in response_lower,
            'search_files': 'search_files' in response_lower,
            'tool_call': len(tool_calls) > 0
        }
        
        print("\n🔍 Response Analysis:")
        for indicator, found in tool_indicators.items():
            print(f"  - {indicator}: {'✅ mentioned' if found else '❌ not mentioned'}")
        
        # Check for action vs planning language
        action_words = ['modified', 'changed', 'updated', 'edited', 'replaced', 'added']
        planning_words = ['will', 'would', 'should', 'need to', 'going to']
        
        actions = sum(1 for word in action_words if word in response_lower)
        planning = sum(1 for word in planning_words if word in response_lower)
        
        print(f"\n📊 Language Analysis:")
        print(f"  - Action words: {actions}")
        print(f"  - Planning words: {planning}")
        print(f"  - Action/Planning ratio: {actions/(planning+1):.2f}")
        
        # Get tool usage summary
        summary = callback_handler.get_summary()
        summary['total_calls'] = len(tool_calls)
        summary['unique_tools'] = list(set(call['name'] for call in tool_calls))
        
        print(f"\n🔧 Tool Usage Summary:")
        print(f"  - Total tool calls: {summary['total_calls']}")
        print(f"  - Unique tools used: {summary['unique_tools']}")
        print(f"  - File read attempts: {summary['read_counter']}")
        print(f"  - Successful edits: {len(summary['successful_edits'])}")
        print(f"  - Loop detected: {summary['loop_detected']}")
        
        # Show successful edits
        if summary['successful_edits']:
            print(f"\n🎉 Successful Edits:")
            for edit in summary['successful_edits']:
                if edit['type'] == 'edit':
                    print(f"  - Edited {edit['lines']} lines in {edit['file']}")
                elif edit['type'] == 'write':
                    print(f"  - Wrote {edit['bytes']} bytes to {edit['file']}")
        
        # Determine overall success
        tool_usage_success = summary['total_calls'] > 0
        edit_success = len(summary['successful_edits']) > 0
        no_loop = not summary['loop_detected']
        
        overall_success = tool_usage_success and no_loop
        
        print(f"\n{'🎉' if overall_success else '❌'} Overall Assessment:")
        print(f"  - Used tools: {'✅' if tool_usage_success else '❌'}")
        print(f"  - Made edits: {'✅' if edit_success else '❌'}")
        print(f"  - No loops: {'✅' if no_loop else '❌'}")
        print(f"  - Success: {'✅' if overall_success else '❌'}")
        
        return {
            "success": True,
            "overall_success": overall_success,
            "tool_calls": summary['total_calls'],
            "successful_edits": len(summary['successful_edits']),
            "loop_detected": summary['loop_detected'],
            "response_length": len(response.content),
            "action_planning_ratio": actions/(planning+1)
        }
        
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "tool_summary": callback_handler.get_summary()
        }

async def test_streaming_with_tools():
    """Test streaming responses while monitoring tool usage."""
    
    print("\n🔄 Testing streaming with tool monitoring...")
    
    server_tools = await get_server_tools()
    
    # Initialize client with streaming enabled
    client = ChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy",
        model="Qwen3-Coder-30B-A3B-Instruct-4bit",
        temperature=0.3,
        streaming=True
    )
    
    callback_handler = ToolUsageCallbackHandler()
    
    messages = [
        SystemMessage(content="You are a helpful assistant with file editing tools. Use tools to complete requests efficiently."),
        HumanMessage(content="Please read main.dart and show me its current structure, then make improvements to the button layout.")
    ]
    
    try:
        print("📡 Starting streaming request...")
        
        # Collect streaming response
        full_response = ""
        chunk_count = 0
        tool_calls = []
        
        async for chunk in client.astream(
            messages,
            tools=server_tools,
            config={"callbacks": [callback_handler]}
        ):
            if chunk.content:
                full_response += chunk.content
                chunk_count += 1
                print(".", end="", flush=True)
            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)

        
        print(f"\n✅ Streaming complete: {chunk_count} chunks, {len(full_response)} total chars")
        
        # Analyze streaming results
        summary = callback_handler.get_summary()
        summary['total_calls'] = len(tool_calls)
        tool_mentions = sum(1 for tool in ['read_file', 'edit_file', 'search_files'] 
                           if tool in full_response.lower())
        
        print(f"🔧 Streaming Results:")
        print(f"  - Tool calls detected: {summary['total_calls']}")
        print(f"  - Tool mentions in text: {tool_mentions}")
        print(f"  - Successful edits: {len(summary['successful_edits'])}")
        
        return {
            "success": True,
            "chunks": chunk_count,
            "response_length": len(full_response),
            "tool_calls": summary['total_calls'],
            "tool_mentions": tool_mentions
        }
        
    except Exception as e:
        print(f"❌ Streaming failed: {e}")
        return {"success": False, "error": str(e)}

async def test_explicit_tool_forcing():
    """Test with very explicit instructions to force tool usage."""
    
    print("\n🎯 Testing with explicit tool forcing...")
    
    server_tools = await get_server_tools()
    
    client = ChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy", 
        model="Qwen3-Coder-30B-A3B-Instruct-4bit",
        temperature=0.1  # Lower temperature for more focused responses
    )
    
    callback_handler = ToolUsageCallbackHandler(max_same_file_reads=1)
    
    # Very explicit and directive system message
    messages = [
        SystemMessage(content="""
IMPORTANT: You are a file editor that MUST use tools. Do not provide examples or descriptions.

REQUIRED WORKFLOW:
1. Use read_file to examine main.dart (do this ONLY ONCE)
2. Use edit_file to make the requested changes
3. Confirm the changes were made

DO NOT:
- Read the same file multiple times 
- Provide code examples without editing
- Explain what you would do - just do it

TOOLS AVAILABLE: read_file, edit_file, search_files, list_directory, write_file

You MUST use these tools to complete the task. No exceptions.
        """.strip()),
        
        HumanMessage(content="""
TASK: Modify main.dart file NOW.

SPECIFIC CHANGES REQUIRED:
1. Change button layout from horizontal (Row) to vertical (Column)
2. Add spacing between buttons  
3. Display folder paths next to relevant buttons

EXECUTE THIS IMMEDIATELY using the available tools. Do not explain - just execute.
        """.strip())
    ]
    
    try:
        print("📤 Sending explicit directive...")
        
        response = await client.ainvoke(
            messages,
            tools=server_tools,
            config={"callbacks": [callback_handler]}
        )
        
        print(f"📥 Response received ({len(response.content)} chars)")
        
        # Detailed analysis for explicit test
        summary = callback_handler.get_summary()
        response_lower = response.content.lower()
        tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []
        
        # Check for different types of responses
        response_types = {
            'tool_usage': len(tool_calls) > 0,
            'code_example': 'widget' in response_lower and 'flutter' in response_lower,
            'explanation': any(word in response_lower for word in ['first', 'then', 'next', 'step']),
            'action_taken': any(word in response_lower for word in ['modified', 'changed', 'updated'])
        }
        
        print(f"\n🔍 Response Type Analysis:")
        for resp_type, detected in response_types.items():
            print(f"  - {resp_type}: {'✅' if detected else '❌'}")
        
        summary['total_calls'] = len(tool_calls)
        summary['unique_tools'] = list(set(call['name'] for call in tool_calls))
        
        print(f"\n🔧 Tool Execution Results:")
        print(f"  - Tools called: {summary['total_calls']}")
        print(f"  - Unique tools: {summary['unique_tools']}")  
        print(f"  - Edits made: {len(summary['successful_edits'])}")
        print(f"  - Loop detected: {summary['loop_detected']}")
        
        # Success criteria for explicit test
        used_tools = summary['total_calls'] > 0
        made_edits = len(summary['successful_edits']) > 0
        no_excessive_reads = not summary['loop_detected']
        
        explicit_success = used_tools and no_excessive_reads
        
        print(f"\n{'🎉' if explicit_success else '❌'} Explicit Test Results:")
        print(f"  - Followed instructions: {'✅' if used_tools else '❌'}")
        print(f"  - Made actual edits: {'✅' if made_edits else '❌'}")
        print(f"  - Efficient execution: {'✅' if no_excessive_reads else '❌'}")
        
        return {
            "success": True,
            "explicit_success": explicit_success,
            "used_tools": used_tools,
            "made_edits": made_edits,
            "tool_calls": summary['total_calls'],
            "response_types": response_types
        }
        
    except Exception as e:
        print(f"❌ Explicit test failed: {e}")
        return {"success": False, "error": str(e)}

async def main():
    """Run comprehensive LangChain OpenAI tests."""
    
    print("🚀 LangChain OpenAI Integration Tests")
    print("=====================================")
    print("Testing mlx-rag server tool usage via LangChain OpenAI client\n")
    
    # Install langchain-openai if not already installed
    print("📦 Ensuring langchain-openai is available...")
    try:
        import langchain_openai
        print("✅ langchain-openai is available")
    except ImportError:
        print("❌ langchain-openai not found. Run: uv add langchain-openai")
        return
    
    # Run all tests
    print("\n" + "="*50)
    print("TEST 1: Basic Tool Usage")
    print("="*50)
    result1 = await test_basic_langchain_openai()
    
    print("\n" + "="*50)  
    print("TEST 2: Streaming with Tools")
    print("="*50)
    result2 = await test_streaming_with_tools()
    
    print("\n" + "="*50)
    print("TEST 3: Explicit Tool Forcing")
    print("="*50)
    result3 = await test_explicit_tool_forcing()
    
    # Comprehensive summary
    print("\n" + "="*60)
    print("🏁 COMPREHENSIVE TEST SUMMARY")
    print("="*60)
    
    print(f"\n📊 Test Results:")
    print(f"  Test 1 (Basic): {'✅ Success' if result1.get('success') else '❌ Failed'}")
    if result1.get('success'):
        print(f"    - Tool calls: {result1.get('tool_calls', 0)}")
        print(f"    - Successful edits: {result1.get('successful_edits', 0)}")
        print(f"    - Overall success: {'✅' if result1.get('overall_success') else '❌'}")
    
    print(f"  Test 2 (Streaming): {'✅ Success' if result2.get('success') else '❌ Failed'}")
    if result2.get('success'):
        print(f"    - Chunks received: {result2.get('chunks', 0)}")
        print(f"    - Tool calls: {result2.get('tool_calls', 0)}")
    
    print(f"  Test 3 (Explicit): {'✅ Success' if result3.get('success') else '❌ Failed'}")
    if result3.get('success'):
        print(f"    - Used tools: {'✅' if result3.get('used_tools') else '❌'}")
        print(f"    - Made edits: {'✅' if result3.get('made_edits') else '❌'}")
        print(f"    - Explicit success: {'✅' if result3.get('explicit_success') else '❌'}")
    
    # Overall assessment
    successful_tests = sum(1 for result in [result1, result2, result3] if result.get('success'))
    tool_usage_detected = any(
        result.get('tool_calls', 0) > 0 
        for result in [result1, result2, result3] 
        if result.get('success')
    )
    edits_made = any(
        result.get('successful_edits', 0) > 0 or result.get('made_edits', False)
        for result in [result1, result2, result3]
        if result.get('success')
    )
    
    print(f"\n🎯 Key Findings:")
    print(f"  - Tests completed: {successful_tests}/3")
    print(f"  - Tool usage detected: {'✅' if tool_usage_detected else '❌'}")
    print(f"  - File edits made: {'✅' if edits_made else '❌'}")
    
    if tool_usage_detected:
        print(f"  - ✅ LangChain OpenAI integration is working")
        print(f"  - ✅ Model successfully uses tools via LangChain")
    else:
        print(f"  - ❌ Tool usage not detected - possible integration issues")
    
    if edits_made:
        print(f"  - 🎉 SUCCESS: Model can modify files through LangChain!")
    else:
        print(f"  - ⚠️  Model uses tools but may not be making actual edits")
    
    print(f"\n💡 Conclusion:")
    if tool_usage_detected and not edits_made:
        print("  - The read loop issue is confirmed: model reads but doesn't edit")
    elif tool_usage_detected and edits_made:
        print("  - LangChain integration successfully bypasses read loop issues")
    else:
        print("  - Further investigation needed for tool integration")

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Debug script to compare Chat interface vs test_simple.py behavior.
This will help identify why the Chat interface shows raw text instead of executing tools.
"""

import asyncio
import httpx
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"
MODEL_NAME = "Qwen3-Coder-30B-A3B-Instruct-4bit"
TEST_PROMPT = "Modify main.dart to space out the action buttons vertically and display the picked books folder and Calibre library path next to the relevant buttons."

async def check_server_status():
    """Check if the server is running and responsive."""
    print("🔍 CHECKING SERVER STATUS")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # Health check
            response = await client.get(f"{BASE_URL}/health")
            print(f"✅ Server Health: {response.status_code}")
            
            # Models check
            response = await client.get(f"{BASE_URL}/v1/manager/models")
            if response.status_code == 200:
                models = response.json().get('models', [])
                target_model = next((m for m in models if m.get('name') == MODEL_NAME), None)
                if target_model:
                    print(f"✅ Target Model Found: {target_model.get('status', 'unknown')}")
                else:
                    print(f"❌ Target Model '{MODEL_NAME}' not found")
                    return False
            
            # Tools check
            response = await client.get(f"{BASE_URL}/v1/tools")
            if response.status_code == 200:
                tools_data = response.json()
                tools_count = len(tools_data.get('tools', []))
                print(f"✅ Tools Available: {tools_count}")
                if tools_count == 0:
                    print("❌ No tools available - this will cause the issue!")
                    return False
            else:
                print(f"❌ Tools endpoint failed: {response.status_code}")
                return False
                
            # RAG collections check
            response = await client.get(f"{BASE_URL}/v1/rag/collections")
            if response.status_code == 200:
                collections = response.json().get('collections', [])
                active_collection = next((c for c in collections if c.get('is_active')), None)
                if active_collection:
                    print(f"✅ Active RAG Collection: {active_collection.get('name')}")
                else:
                    print("❌ No active RAG collection")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Server check failed: {e}")
            return False

async def test_chat_interface_request():
    """Test the exact request the Chat interface makes."""
    print("\n🌐 TESTING CHAT INTERFACE REQUEST")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Get tools (like Chat interface does)
        tools_response = await client.get(f"{BASE_URL}/v1/tools")
        if tools_response.status_code != 200:
            print("❌ Failed to get tools")
            return False
            
        tools_data = tools_response.json()
        tools = tools_data.get('tools', [])
        print(f"📋 Tools fetched: {len(tools)}")
        
        if len(tools) == 0:
            print("❌ No tools available - Chat interface will not use tools!")
            return False
        
        # Step 2: Build request exactly like Chat interface
        messages = [
            {
                "role": "system",
                "content": f"""You have access to a RAG collection named "Booker" containing source code and documentation. You have access to the following tools to interact with the codebase:
{chr(10).join([f"- {tool['function']['name']}: {tool['function']['description']}" for tool in tools])}

IMPORTANT: When a user asks about modifying, reading, or working with files, you MUST use the appropriate tools (like read_file, search_files, list_directory, etc.) to examine the codebase first, then provide specific solutions. Do not make assumptions about code you haven't seen."""
            },
            {
                "role": "user",
                "content": TEST_PROMPT
            }
        ]
        
        request_body = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.7,
            "stream": False,
            "tools": tools,
            "tool_choice": "auto"
        }
        
        print(f"📤 Sending request to /v1/chat/completions")
        print(f"   Model: {MODEL_NAME}")
        print(f"   Tools: {len(tools)}")
        print(f"   Stream: {request_body['stream']}")
        
        try:
            response = await client.post(
                f"{BASE_URL}/v1/chat/completions",
                headers={'Content-Type': 'application/json'},
                json=request_body
            )
            
            print(f"📥 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                choice = data.get('choices', [{}])[0]
                message = choice.get('message', {})
                tool_calls = message.get('tool_calls', [])
                
                print(f"🔍 Response Analysis:")
                print(f"   - Finish Reason: {choice.get('finish_reason', 'None')}")
                print(f"   - Has Content: {bool(message.get('content'))}")
                print(f"   - Tool Calls: {len(tool_calls)}")
                
                if tool_calls:
                    print(f"✅ Model made {len(tool_calls)} tool calls:")
                    for i, call in enumerate(tool_calls):
                        func_name = call.get('function', {}).get('name', 'Unknown')
                        print(f"   {i+1}. {func_name}")
                    return True
                else:
                    print("❌ Model did not make tool calls")
                    print(f"   Content: {message.get('content', 'None')[:200]}...")
                    return False
                    
            else:
                print(f"❌ Request failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return False

async def test_simple_script_approach():
    """Test the approach used by test_simple.py script."""
    print("\n🧪 TESTING test_simple.py APPROACH")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Get tools
        tools_response = await client.get(f"{BASE_URL}/v1/tools")
        tools = tools_response.json().get('tools', []) if tools_response.status_code == 200 else []
        
        # Create manual agent-like request (simpler system prompt)
        messages = [
            {
                "role": "user",
                "content": "Please read the main.dart file first"
            }
        ]
        
        request_body = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.3,
            "stream": False,
            "tools": tools,
            "tool_choice": "auto"
        }
        
        print(f"📤 Sending simplified request")
        
        try:
            response = await client.post(
                f"{BASE_URL}/v1/chat/completions",
                json=request_body
            )
            
            if response.status_code == 200:
                data = response.json()
                choice = data.get('choices', [{}])[0]
                message = choice.get('message', {})
                tool_calls = message.get('tool_calls', [])
                
                print(f"🔍 Simple Request Analysis:")
                print(f"   - Tool Calls: {len(tool_calls)}")
                
                if tool_calls:
                    print(f"✅ Simple approach works: {len(tool_calls)} tool calls")
                    return True
                else:
                    print("❌ Simple approach also fails")
                    return False
            else:
                print(f"❌ Simple request failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Exception in simple test: {e}")
            return False

async def compare_approaches():
    """Compare what's different between the two approaches."""
    print("\n🔍 DETAILED COMPARISON")
    print("=" * 50)
    
    # Check if it's a model-specific issue
    print("Possible causes for Chat interface not working:")
    print("1. Different model behavior with complex system prompts")
    print("2. RAG collection not properly configured")
    print("3. Tools not available when Chat interface checks")
    print("4. Model not properly loaded")
    print("5. Different temperature/parameters affecting tool calling")
    
    server_ok = await check_server_status()
    if not server_ok:
        print("\n❌ DIAGNOSIS: Server configuration issue")
        return
    
    chat_works = await test_chat_interface_request()
    simple_works = await test_simple_script_approach()
    
    print(f"\n📊 RESULTS:")
    print(f"   - Chat Interface Request: {'✅ Works' if chat_works else '❌ Fails'}")
    print(f"   - Simple Script Request: {'✅ Works' if simple_works else '❌ Fails'}")
    
    if not chat_works and not simple_works:
        print("\n🚨 DIAGNOSIS: Tool calling is broken entirely")
        print("   - Check model supports function calling")
        print("   - Check system prompt format")
        print("   - Check if RAG collection is active")
    elif not chat_works and simple_works:
        print("\n🚨 DIAGNOSIS: Chat interface system prompt too complex")
        print("   - Try simplifying the system prompt in chat.js")
        print("   - Check if model struggles with long prompts")
    elif chat_works and not simple_works:
        print("\n🎉 DIAGNOSIS: Both should work - recheck Chat interface")
    else:
        print("\n🎉 DIAGNOSIS: Everything works - issue might be intermittent")

async def main():
    """Main diagnostic function."""
    print(f"🔍 MLX-RAG CHAT VS TEST_SIMPLE DIAGNOSTIC")
    print(f"Time: {datetime.now()}")
    print(f"Target: {BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)
    
    await compare_approaches()
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print("1. Ensure the Booker RAG collection is active")
    print("2. Ensure the model is fully loaded")
    print("3. Check browser console for Chat interface errors")
    print("4. Try refreshing the Chat page")
    print("5. Try creating a new chat session")

if __name__ == "__main__":
    asyncio.run(main())

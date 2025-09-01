#!/usr/bin/env python3
"""
Test script to replicate the exact scenario from the screenshot.

From the screenshot:
- Model: mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
- RAG Collection: Booker (active)
- User query: "What is the main.dart file?"
- Tools should be available and used
"""

import asyncio
import httpx
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

async def test_exact_scenario():
    """Test the exact scenario from the screenshot."""
    
    print("🧪 Testing exact scenario from screenshot...")
    print(f"📡 Using server: {BASE_URL}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        
        # Step 1: Check if server is running
        try:
            response = await client.get(f"{BASE_URL}/health")
            if response.status_code != 200:
                print("❌ Server not running or not healthy")
                return
            print("✅ Server is healthy")
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return
        
        # Step 2: Check models - look for the exact model from screenshot
        print("\n📋 Checking available models...")
        try:
            response = await client.get(f"{BASE_URL}/v1/manager/models")
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                print(f"Found {len(models)} models:")
                
                target_model = None
                for model in models:
                    print(f"  - {model['name']} ({model['status']})")
                    # Look for the exact model from screenshot first
                    if model['name'] == "Qwen3-Coder-30B-A3B-Instruct-4bit":
                        target_model = model
                        print(f"    ✅ Found exact model from screenshot: {model['name']} (Status: {model['status']})")
                        break
                    # Fallback to any Qwen coder model
                    elif "qwen" in model['name'].lower() and "coder" in model['name'].lower():
                        target_model = model
                        print(f"    📋 Found Qwen coder model: {model['name']} (Status: {model['status']})")
                
                if not target_model:
                    print("❌ No Qwen coder model found, will use first available model")
                    # Fallback to first available model
                    if models:
                        target_model = models[0]
                        print(f"    📋 Using fallback model: {target_model['name']}")
                    else:
                        print("❌ No models available at all")
                        return
            else:
                print(f"❌ Failed to get models: {response.status_code}")
                return
        except Exception as e:
            print(f"❌ Error checking models: {e}")
            return
        
        # Step 3: Check RAG collections - look for "mlx-rag"
        print("\n📚 Checking RAG collections...")
        try:
            response = await client.get(f"{BASE_URL}/v1/rag/collections")
            if response.status_code == 200:
                data = response.json()
                collections = data.get("collections", [])
                print(f"Found {len(collections)} collections:")
                
                target_collection = None
                for collection in collections:
                    print(f"  - {collection['name']} ({'active' if collection['is_active'] else 'inactive'})")
                    if collection['is_active']:  # Use any active collection
                        target_collection = collection
                        print(f"    ✅ Found active collection: {collection['name']}")
                        break
                
                if not target_collection:
                    print("❌ No active RAG collection found")
                    return
            else:
                print(f"❌ Failed to get collections: {response.status_code}")
                return
        except Exception as e:
            print(f"❌ Error checking collections: {e}")
            return
        
        # Step 4: Check available tools
        print("\n🔧 Checking available tools...")
        try:
            response = await client.get(f"{BASE_URL}/v1/tools")
            if response.status_code == 200:
                data = response.json()
                tools = data.get("tools", [])
                print(f"Found {len(tools)} tools:")
                for tool in tools:
                    if 'function' in tool:
                        print(f"  - {tool['function']['name']}: {tool['function']['description']}")
                    else:
                        print(f"  - {tool}")
                
                if not tools:
                    print("❌ No tools available")
                    return
                else:
                    print("✅ Tools are available")
            else:
                print(f"❌ Failed to get tools: {response.status_code}")
                return
        except Exception as e:
            print(f"❌ Error checking tools: {e}")
            return
        
        # Step 5: Load the model if it's not loaded
        if target_model['status'] != 'loaded':
            print(f"\n🔄 Loading model {target_model['name']}...")
            try:
                response = await client.post(f"{BASE_URL}/v1/models/{target_model['name']}/load")
                if response.status_code == 200:
                    print("✅ Model loaded successfully")
                else:
                    print(f"❌ Failed to load model: {response.status_code} - {response.text}")
                    return
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                return
        else:
            print(f"\n✅ Model {target_model['name']} is already loaded")
        
        # Step 6: Make the exact chat completion request from the screenshot
        print("\n💬 Making chat completion request...")
        
        # This matches the scenario but uses the available model
        request_data = {
            "model": target_model['name'],
            "messages": [
                {
                    "role": "user",
                    "content": "Modify main.dart file to space out the action buttons vertically and display the picked books folder and Calibre library path next to the relevant buttons."
                }
            ],
            "tools": tools,  # Use the tools we discovered
            "max_tokens": 2048,
            "temperature": 0.7
        }
        
        print(f"📤 Request: {json.dumps(request_data, indent=2)}")
        
        try:
            response = await client.post(
                f"{BASE_URL}/v1/chat/completions",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"\n📥 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"📄 Response: {json.dumps(data, indent=2)}")
                
                # Check if the response includes tool calls
                if 'choices' in data and len(data['choices']) > 0:
                    choice = data['choices'][0]
                    message = choice.get('message', {})
                    
                    if 'tool_calls' in message and message['tool_calls']:
                        print("✅ Response contains tool calls!")
                        for i, tool_call in enumerate(message['tool_calls']):
                            print(f"  Tool Call {i+1}:")
                            print(f"    Function: {tool_call['function']['name']}")
                            print(f"    Arguments: {tool_call['function']['arguments']}")
                    else:
                        print("❌ No tool calls in response")
                        if 'content' in message:
                            print(f"📝 Text response: {message['content']}")
                
            else:
                print(f"❌ Request failed: {response.text}")
                
        except Exception as e:
            print(f"❌ Error making request: {e}")
            return
        
        print("\n🏁 Test completed!")

if __name__ == "__main__":
    asyncio.run(test_exact_scenario())

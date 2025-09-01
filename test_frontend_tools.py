#!/usr/bin/env python3
"""
Test script to verify frontend tool calling behavior by simulating the exact frontend requests.
This will help us identify if the issue is in the frontend JavaScript logic.
"""

import asyncio
import httpx
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

async def test_frontend_tool_behavior():
    """Test the exact frontend tool calling behavior."""
    
    print("🧪 Testing frontend tool calling behavior...")
    print(f"📡 Using server: {BASE_URL}")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        
        # Step 1: Get available tools (like frontend does)
        print("\n🔧 Getting available tools...")
        try:
            response = await client.get(f"{BASE_URL}/v1/tools")
            if response.status_code == 200:
                tools_data = response.json()
                tools = tools_data.get("tools", [])
                print(f"✅ Found {len(tools)} tools:")
                for tool in tools:
                    print(f"  - {tool['function']['name']}: {tool['function']['description']}")
                
                if not tools:
                    print("❌ No tools available - frontend won't make tool calls")
                    return
            else:
                print(f"❌ Failed to get tools: {response.status_code}")
                return
        except Exception as e:
            print(f"❌ Error getting tools: {e}")
            return
        
        # Step 2: Make the exact request the frontend makes
        print("\n💬 Making chat completion request with tools (simulating frontend)...")
        
        # This mimics exactly what the frontend JavaScript does
        request_data = {
            "model": "Qwen3-Coder-30B-A3B-Instruct-4bit",
            "messages": [
                {
                    "role": "system",
                    "content": f"""You have access to a RAG collection named "Booker" containing source code and documentation. You have access to the following tools to interact with the codebase:
{chr(10).join([f"- {tool['function']['name']}: {tool['function']['description']}" for tool in tools])}

IMPORTANT: When a user asks about modifying, reading, or working with files, you MUST use the appropriate tools (like read_file, search_files, list_directory, etc.) to examine the codebase first, then provide specific solutions. Do not make assumptions about code you haven't seen."""
                },
                {
                    "role": "user",
                    "content": "What is the main.dart file?"
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.7,
            "stream": False,  # Frontend uses non-streaming for tool calls
            "tools": tools,
            "tool_choice": "auto"
        }
        
        print(f"📤 Request (abbreviated): Model={request_data['model']}, Tools={len(tools)}, Stream={request_data['stream']}")
        
        try:
            # Step 3: Send initial request
            response = await client.post(
                f"{BASE_URL}/v1/chat/completions",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"📥 Response Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ Request failed: {response.text}")
                return
            
            response_data = response.json()
            choice = response_data.get('choices', [{}])[0]
            message = choice.get('message', {})
            
            content = message.get('content', '')
            tool_calls = message.get('tool_calls', [])
            
            print(f"🔍 Response analysis:")
            print(f"  - Has content: {content is not None}")
            print(f"  - Content length: {len(content) if content else 0}")
            print(f"  - Has tool_calls: {tool_calls is not None and len(tool_calls) > 0}")
            print(f"  - Tool calls count: {len(tool_calls)}")
            print(f"  - Finish reason: {choice.get('finish_reason')}")
            
            # Step 4: Handle tool calls (if any)
            if message.get('tool_calls') and len(message['tool_calls']) > 0:
                print(f"\n🔧 Model requested {len(message['tool_calls'])} tool calls!")
                
                # Execute each tool call (like frontend does)
                tool_results = []
                for i, tool_call in enumerate(message['tool_calls']):
                    print(f"\n  Tool Call {i+1}:")
                    print(f"    Function: {tool_call['function']['name']}")
                    print(f"    Arguments: {tool_call['function']['arguments']}")
                    
                    try:
                        # Execute tool (like frontend does)
                        tool_response = await client.post(
                            f"{BASE_URL}/v1/tools/execute",
                            json={
                                "tool_call_id": tool_call['id'],
                                "function_name": tool_call['function']['name'],
                                "arguments": tool_call['function']['arguments']
                            }
                        )
                        
                        if tool_response.status_code != 200:
                            print(f"    ❌ Tool execution failed: {tool_response.status_code}")
                            tool_results.append({
                                "tool_call_id": tool_call['id'],
                                "role": "tool",
                                "content": json.dumps({"error": f"Tool execution failed: {tool_response.status_code}"})
                            })
                        else:
                            tool_result = tool_response.json()
                            print(f"    ✅ Tool executed successfully")
                            print(f"    📄 Result (first 200 chars): {str(tool_result.get('result', ''))[:200]}...")
                            
                            result_content = json.dumps(tool_result.get('result')) if tool_result.get('success') else json.dumps({"error": tool_result.get('error')})
                            tool_results.append({
                                "tool_call_id": tool_call['id'],
                                "role": "tool",
                                "content": result_content
                            })
                    
                    except Exception as e:
                        print(f"    ❌ Error executing tool: {e}")
                        tool_results.append({
                            "tool_call_id": tool_call['id'],
                            "role": "tool",
                            "content": json.dumps({"error": str(e)})
                        })
                
                # Step 5: Make follow-up request with tool results (like frontend does)
                print(f"\n🔄 Making follow-up request with {len(tool_results)} tool results...")
                
                follow_up_messages = request_data['messages'].copy()
                
                # Add assistant's tool call message
                follow_up_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": message['tool_calls']
                })
                
                # Add tool results
                follow_up_messages.extend(tool_results)
                
                follow_up_request = {
                    "model": request_data['model'],
                    "messages": follow_up_messages,
                    "max_tokens": request_data['max_tokens'],
                    "temperature": request_data['temperature'],
                    "stream": False
                    # No tools in follow-up request
                }
                
                try:
                    follow_up_response = await client.post(
                        f"{BASE_URL}/v1/chat/completions",
                        json=follow_up_request
                    )
                    
                    if follow_up_response.status_code == 200:
                        follow_up_data = follow_up_response.json()
                        final_choice = follow_up_data.get('choices', [{}])[0]
                        final_message = final_choice.get('message', {})
                        final_content = final_message.get('content', '')
                        
                        print(f"🎉 Final response received!")
                        print(f"📝 Response length: {len(final_content)} characters")
                        print(f"📖 Final response (first 500 chars):")
                        print(final_content[:500] + ("..." if len(final_content) > 500 else ""))
                        
                        print("\n✅ COMPLETE TOOL CALLING WORKFLOW SUCCESSFUL!")
                        print("The frontend logic should work perfectly!")
                        
                    else:
                        print(f"❌ Follow-up request failed: {follow_up_response.status_code}")
                        print(follow_up_response.text)
                
                except Exception as e:
                    print(f"❌ Error in follow-up request: {e}")
            
            else:
                print("❌ NO TOOL CALLS in response!")
                if message.get('content'):
                    print(f"📝 Text response instead: {message['content'][:200]}...")
                    print("\n🚨 This indicates the model is NOT making tool calls when it should!")
                
        except Exception as e:
            print(f"❌ Error making request: {e}")
            return
        
        print("\n🏁 Frontend tool behavior test completed!")

if __name__ == "__main__":
    asyncio.run(test_frontend_tool_behavior())

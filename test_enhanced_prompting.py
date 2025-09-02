#!/usr/bin/env python3
"""
Enhanced test with better prompting to encourage the LLM to actually use editing tools.
"""

import asyncio
import httpx
import json

BASE_URL = "http://localhost:8000"

async def test_enhanced_prompting():
    """Test with enhanced prompting to encourage tool usage."""
    
    print("🧪 Testing enhanced prompting for tool usage...")
    print(f"📡 Using server: {BASE_URL}")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        
        # Step 1: Check server and get model/tools
        try:
            # Health check
            response = await client.get(f"{BASE_URL}/health")
            if response.status_code != 200:
                print("❌ Server not healthy")
                return
            print("✅ Server is healthy")
            
            # Get model
            response = await client.get(f"{BASE_URL}/v1/manager/models")
            models = response.json().get("models", [])
            target_model = next((m for m in models if "Mistral-7B-Instruct-v0.3-8bit" in m['name']), None)
            if not target_model:
                target_model = next((m for m in models if m['status'] == 'loaded'), models[0] if models else None)
            if not target_model:
                print("❌ No model available")
                return
            print(f"✅ Using model: {target_model['name']}")
            
            # Get tools
            response = await client.get(f"{BASE_URL}/v1/tools")
            tools = response.json().get("tools", [])
            if not tools:
                print("❌ No tools available")
                return
            print(f"✅ {len(tools)} tools available")
            
        except Exception as e:
            print(f"❌ Setup error: {e}")
            return
        
        # Step 2: Enhanced prompting with explicit instructions
        print("\n💬 Making enhanced chat completion request...")
        
        # More explicit instructions that encourage tool usage
        enhanced_request = {
            "model": target_model['name'],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful coding assistant with access to file system tools. "
                        "When asked to modify code files, you should:\n"
                        "1. ALWAYS use the available tools to explore and read files first\n"  
                        "2. ACTUALLY MODIFY the files using edit_file or write_file tools\n"
                        "3. Don't just provide code examples - make the actual changes\n"
                        "4. Use exact file paths from directory listings\n"
                        "5. Read the existing code before making modifications\n\n"
                        "Available tools: list_directory, read_file, search_files, write_file, edit_file"
                    )
                },
                {
                    "role": "user", 
                    "content": (
                        "I need you to modify the main.dart file in the current project. "
                        "Please:\n"
                        "1. First explore the project to find main.dart\n"
                        "2. Read the current content of main.dart\n"
                        "3. ACTUALLY modify the file to space out the action buttons vertically\n"
                        "4. Add display of picked books folder path and Calibre library path next to the relevant buttons\n\n"
                        "Please use the file editing tools to make these changes - don't just show me code examples."
                    )
                }
            ],
            "tools": tools,
            "max_tokens": 3000,
            "temperature": 0.3  # Lower temperature for more focused responses
        }
        
        print(f"📤 Enhanced Request with explicit instructions")
        
        try:
            # Make the initial request
            response = await client.post(
                f"{BASE_URL}/v1/chat/completions",
                json=enhanced_request,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"\n📥 Response Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ Request failed: {response.text}")
                return
                
            data = response.json()
            
            # Process the conversation through multiple turns if needed
            messages = enhanced_request["messages"]
            turn_count = 0
            max_turns = 5
            
            while turn_count < max_turns:
                turn_count += 1
                print(f"\n🔄 Turn {turn_count}")
                
                if 'choices' not in data or len(data['choices']) == 0:
                    print("❌ No response choices")
                    break
                    
                choice = data['choices'][0]
                message = choice.get('message', {})
                
                # Add assistant's response to conversation
                messages.append(message)
                
                # Check for tool calls
                if 'tool_calls' in message and message['tool_calls']:
                    print(f"✅ Found {len(message['tool_calls'])} tool calls")
                    
                    # Execute each tool call
                    tool_results = []
                    for i, tool_call in enumerate(message['tool_calls']):
                        print(f"  🔧 Executing: {tool_call['function']['name']}")
                        print(f"    Args: {tool_call['function']['arguments']}")
                        
                        tool_request = {
                            "tool_call_id": tool_call['id'],
                            "function_name": tool_call['function']['name'],
                            "arguments": tool_call['function']['arguments']
                        }
                        
                        try:
                            tool_response = await client.post(
                                f"{BASE_URL}/v1/tools/execute",
                                json=tool_request,
                                headers={"Content-Type": "application/json"}
                            )
                            
                            if tool_response.status_code == 200:
                                tool_result = tool_response.json()
                                print(f"    ✅ Success: {tool_result.get('success', 'Unknown')}")
                                
                                # Show key parts of result
                                if tool_result.get('success'):
                                    result_content = tool_result.get('result', '')
                                    if isinstance(result_content, dict):
                                        if 'items' in result_content:  # Directory listing
                                            print(f"      Found {len(result_content['items'])} items")
                                        elif 'content' in result_content:  # File content
                                            content_preview = result_content['content'][:100] + "..." if len(result_content['content']) > 100 else result_content['content']
                                            print(f"      Content preview: {content_preview}")
                                        elif 'bytes_written' in result_content:  # File write
                                            print(f"      Wrote {result_content['bytes_written']} bytes")
                                else:
                                    print(f"      Error: {tool_result.get('error', 'Unknown error')}")
                                
                                # Format result for conversation
                                result_content = tool_result.get('result', tool_result.get('error', ''))
                                if isinstance(result_content, dict) or isinstance(result_content, list):
                                    result_content = json.dumps(result_content)
                                
                                tool_results.append({
                                    "tool_call_id": tool_call['id'],
                                    "role": "tool", 
                                    "content": result_content
                                })
                            else:
                                print(f"    ❌ Tool failed: {tool_response.text}")
                                tool_results.append({
                                    "tool_call_id": tool_call['id'],
                                    "role": "tool",
                                    "content": f"Error: {tool_response.text}"
                                })
                                
                        except Exception as e:
                            print(f"    ❌ Exception: {e}")
                            tool_results.append({
                                "tool_call_id": tool_call['id'],
                                "role": "tool",
                                "content": f"Exception: {e}"
                            })
                    
                    # Add tool results to conversation
                    messages.extend(tool_results)
                    
                    # Continue conversation with tool results
                    continue_request = {
                        "model": target_model['name'],
                        "messages": messages,
                        "tools": tools,
                        "max_tokens": 3000,
                        "temperature": 0.3
                    }
                    
                    try:
                        response = await client.post(
                            f"{BASE_URL}/v1/chat/completions",
                            json=continue_request,
                            headers={"Content-Type": "application/json"}
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            continue  # Continue the loop
                        else:
                            print(f"❌ Continue request failed: {response.text}")
                            break
                            
                    except Exception as e:
                        print(f"❌ Continue request exception: {e}")
                        break
                
                else:
                    # No more tool calls, show final response
                    if 'content' in message and message['content']:
                        print("🎉 Final Response:")
                        print(message['content'])
                    else:
                        print("❌ No content in final response")
                    break
            
            if turn_count >= max_turns:
                print(f"⚠️ Reached maximum turns ({max_turns})")
                
        except Exception as e:
            print(f"❌ Request exception: {e}")
            return
        
        print("\n🏁 Enhanced test completed!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_prompting())

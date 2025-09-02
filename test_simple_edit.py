#!/usr/bin/env python3
"""
Simple test to encourage LLM to use editing tools with very explicit instructions.
"""

import asyncio
import httpx
import json

BASE_URL = "http://localhost:8000"

async def test_simple_edit():
    """Test with simple, explicit instructions for editing."""
    
    print("🧪 Testing simple edit approach...")
    print(f"📡 Using server: {BASE_URL}")
    
    async with httpx.AsyncClient(timeout=180.0) as client:
        
        # Step 1: Setup
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
            target_model = next((m for m in models if "Qwen3-Coder-30B-A3B-Instruct-4bit" in m['name']), None)
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
        
        # Step 2: Very explicit request
        print("\n💬 Making simple, explicit request...")
        
        request_data = {
            "model": target_model['name'],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a code editor assistant. You have access to file editing tools. "
                        "When asked to modify files, you MUST use the edit_file or write_file tools. "
                        "Do not just describe changes - actually make them using the tools."
                    )
                },
                {
                    "role": "user", 
                    "content": (
                        "Search for 'Row' or 'Column' in main.dart to find the UI layout, "
                        "then use edit_file to change the button layout from horizontal to vertical."
                    )
                }
            ],
            "tools": tools,
            "max_tokens": 3000,
            "temperature": 0.2
        }
        
        print("📤 Making request...")
        
        # Execute the conversation with proper handling
        messages = request_data["messages"]
        turn = 0
        max_turns = 6
        
        response = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            print(f"❌ Request failed: {response.text}")
            return
            
        data = response.json()
        
        while turn < max_turns:
            turn += 1
            print(f"\n🔄 Turn {turn}")
            
            if 'choices' not in data or len(data['choices']) == 0:
                print("❌ No response choices")
                break
                
            choice = data['choices'][0]
            message = choice.get('message', {})
            messages.append(message)
            
            # Check for tool calls
            if 'tool_calls' in message and message['tool_calls']:
                tool_calls = message['tool_calls']
                print(f"✅ Found {len(tool_calls)} tool call(s)")
                
                # Look for edit calls specifically
                edit_calls = [tc for tc in tool_calls if tc['function']['name'] in ['edit_file', 'write_file']]
                if edit_calls:
                    print("🎯 EDITING TOOL DETECTED!")
                
                # Execute each tool call
                tool_results = []
                for i, tool_call in enumerate(tool_calls):
                    func_name = tool_call['function']['name']
                    args = tool_call['function']['arguments']
                    
                    print(f"  🔧 {i+1}. {func_name}")
                    
                    # Show args for different tool types
                    try:
                        args_dict = json.loads(args)
                        if func_name == 'search_files':
                            print(f"     Query: {args_dict.get('query', 'N/A')}")
                        elif func_name == 'read_file':
                            print(f"     Path: {args_dict.get('path', 'N/A')}")
                        elif func_name in ['edit_file', 'write_file']:
                            print(f"     🎉 EDITING: {args_dict.get('path', 'N/A')}")
                            if 'start_line' in args_dict:
                                print(f"     Lines: {args_dict.get('start_line')}-{args_dict.get('end_line')}")
                    except:
                        pass
                    
                    # Execute the tool
                    tool_request = {
                        "tool_call_id": tool_call['id'],
                        "function_name": func_name,
                        "arguments": args
                    }
                    
                    try:
                        tool_response = await client.post(
                            f"{BASE_URL}/v1/tools/execute",
                            json=tool_request,
                            headers={"Content-Type": "application/json"}
                        )
                        
                        if tool_response.status_code == 200:
                            tool_result = tool_response.json()
                            success = tool_result.get('success', False)
                            
                            if success:
                                print(f"     ✅ Success")
                                result = tool_result.get('result', {})
                                if isinstance(result, dict):
                                    if 'lines_replaced' in result:
                                        print(f"     📝 EDITED {result['lines_replaced']} lines!")
                                    elif 'bytes_written' in result:
                                        print(f"     📝 WROTE {result['bytes_written']} bytes!")
                                    elif 'total_found' in result:
                                        print(f"     🔍 Found {result['total_found']} matches")
                                    elif 'lines_read' in result:
                                        print(f"     📖 Read {result['lines_read']} lines")
                            else:
                                print(f"     ❌ Failed: {tool_result.get('error', 'Unknown')}")
                            
                            # Format result for conversation
                            result_content = tool_result.get('result', tool_result.get('error', ''))
                            if isinstance(result_content, (dict, list)):
                                result_content = json.dumps(result_content)
                            
                            tool_results.append({
                                "tool_call_id": tool_call['id'],
                                "role": "tool",
                                "content": result_content
                            })
                        else:
                            print(f"     ❌ HTTP Error: {tool_response.text}")
                            tool_results.append({
                                "tool_call_id": tool_call['id'],
                                "role": "tool",
                                "content": f"Error: {tool_response.text}"
                            })
                            
                    except Exception as e:
                        print(f"     ❌ Exception: {e}")
                        tool_results.append({
                            "tool_call_id": tool_call['id'],
                            "role": "tool",
                            "content": f"Exception: {e}"
                        })
                
                # Add tool results to conversation
                messages.extend(tool_results)
                
                # Check if we successfully made edits
                successful_edits = [tr for tr in tool_results 
                                   if "lines_replaced" in tr["content"] or "bytes_written" in tr["content"]]
                
                if successful_edits:
                    print(f"\n🎉 SUCCESS! Made {len(successful_edits)} successful edit(s)!")
                    print("Getting final confirmation...")
                
                # Continue conversation
                continue_request = {
                    "model": target_model['name'],
                    "messages": messages,
                    "tools": tools,
                    "max_tokens": 2000,
                    "temperature": 0.2
                }
                
                try:
                    response = await client.post(
                        f"{BASE_URL}/v1/chat/completions",
                        json=continue_request,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        continue
                    else:
                        print(f"❌ Continue failed: {response.text}")
                        break
                        
                except Exception as e:
                    print(f"❌ Continue exception: {e}")
                    break
            
            else:
                # No tool calls - final response
                if 'content' in message and message['content']:
                    print("\n🎉 Final Response:")
                    print(message['content'])
                else:
                    print("❌ No content in final response")
                break
        
        if turn >= max_turns:
            print(f"⚠️ Reached max turns ({max_turns})")
        
        print("\n🏁 Simple test completed!")

if __name__ == "__main__":
    asyncio.run(test_simple_edit())

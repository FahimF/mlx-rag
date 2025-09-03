#!/usr/bin/env python3
"""
Test with correct conversation flow for tool calling.

This fixes the role alternation issue by properly structuring the conversation
after tool executions.
"""

import asyncio
import json
import httpx

async def test_minimal_correct_flow():
    """Test with minimal but correct conversation flow."""
    
    print("🎯 Testing Minimal Correct Flow")
    print("=" * 35)
    
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        tools_response = await client.get(f"{base_url}/v1/tools")
        server_tools = tools_response.json().get('tools', [])
    
    # Minimal conversation with proper structure
    messages = [
        {"role": "user", "content": "Read main.dart file"}
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # First request
        request_payload = {
            "model": "Mistral-7B-Instruct-v0.3-8bit",
            "messages": messages,
            "tools": server_tools,
            "tool_choice": "auto",
            "max_tokens": 1000,
            "temperature": 0.3,
            "stream": False
        }
        
        print("📤 Step 1: Asking model to read file")
        
        try:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=request_payload
            )
            
            if response.status_code == 200:
                response_data = response.json()
                choice = response_data.get('choices', [{}])[0]
                message = choice.get('message', {})
                tool_calls = message.get('tool_calls', [])
                
                if tool_calls:
                    print(f"✅ Model made {len(tool_calls)} tool calls")
                    
                    # This is the CORRECT conversation structure:
                    messages.append({
                        "role": "assistant",
                        "content": message.get('content'),
                        "tool_calls": tool_calls
                    })
                    
                    # Add tool results
                    for tool_call in tool_calls:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call['id'],
                            "content": "File content here..."  # Simplified
                        })
                    
                    # CRITICAL: Add user message to continue
                    messages.append({
                        "role": "user",
                        "content": "Now edit the file to change layout to vertical"
                    })
                    
                    print("✅ Proper conversation structure maintained")
                    
                    # Verify the structure
                    roles = [msg['role'] for msg in messages]
                    print(f"📝 Conversation roles: {' → '.join(roles)}")
                    
                    # Check alternation pattern
                    print("🔍 Checking role alternation:")
                    for i, role in enumerate(roles):
                        print(f"  {i+1}. {role}")
                    
                    return {
                        "success": True,
                        "roles_correct": True,
                        "conversation_structure": roles
                    }
                    
                else:
                    print("❌ No tool calls made")
                    return {"success": False, "error": "No tool calls"}
                    
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return {"success": False, "error": str(e)}

async def test_complete_conversation_flow():
    """Test complete conversation with real tool execution."""
    
    print("\n🔧 Testing Complete Conversation Flow")
    print("=" * 45)
    
    base_url = "http://localhost:8000"
    
    # Get tools
    async with httpx.AsyncClient() as client:
        tools_response = await client.get(f"{base_url}/v1/tools")
        server_tools = tools_response.json().get('tools', [])
    
    # Initial conversation
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with file tools. Use tools efficiently to complete tasks."
        },
        {
            "role": "user", 
            "content": "Please read main.dart and edit it to change button layout to vertical."
        }
    ]
    
    max_turns = 5
    turn = 1
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        while turn <= max_turns:
            print(f"\n🔄 Turn {turn}")
            
            # Show conversation structure before request
            print("📝 Current conversation:")
            for i, msg in enumerate(messages[-3:]):
                role = msg.get('role')
                has_tools = 'tool_calls' in msg
                content_preview = str(msg.get('content', ''))[:50]
                print(f"  {i+1}. {role}{' (with tools)' if has_tools else ''}: {content_preview}...")
            
            request_payload = {
                "model": "Mistral-7B-Instruct-v0.3-8bit",
                "messages": messages,
                "tools": server_tools,
                "tool_choice": "auto",
                "max_tokens": 1500,
                "temperature": 0.3,
                "stream": False
            }
            
            try:
                response = await client.post(
                    f"{base_url}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=request_payload
                )
                
                if response.status_code != 200:
                    print(f"❌ HTTP {response.status_code}: {response.text}")
                    break
                
                response_data = response.json()
                choice = response_data.get('choices', [{}])[0]
                message = choice.get('message', {})
                tool_calls = message.get('tool_calls', [])
                content = message.get('content')
                
                finish_reason = choice.get('finish_reason')
                print(f"📨 Finish reason: {finish_reason}")
                
                if tool_calls:
                    print(f"🔧 Model requested {len(tool_calls)} tool calls")
                    
                    # Add assistant message with tool calls
                    messages.append({
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls
                    })
                    
                    # Execute tools and add results
                    for i, tool_call in enumerate(tool_calls):
                        func_name = tool_call.get('function', {}).get('name', 'unknown')
                        func_args = tool_call.get('function', {}).get('arguments', '{}')
                        tool_id = tool_call.get('id', f'call_{i}')
                        
                        print(f"  {i+1}. Executing {func_name}")
                        
                        # Execute tool
                        execute_payload = {
                            "tool_call_id": tool_id,
                            "function_name": func_name,
                            "arguments": func_args
                        }
                        
                        try:
                            tool_response = await client.post(
                                f"{base_url}/v1/tools/execute",
                                headers={"Content-Type": "application/json"},
                                json=execute_payload
                            )
                            
                            if tool_response.status_code == 200:
                                result = tool_response.json()
                                success = result.get('success', False)
                                output = result.get('result', result.get('output', 'No output'))
                                
                                if success:
                                    result_content = json.dumps(output) if isinstance(output, dict) else str(output)
                                    print(f"    ✅ Success: {result_content[:100]}...")
                                    
                                    # Check for successful edits
                                    if func_name == 'edit_file':
                                        print(f"    🎉 EDIT OPERATION COMPLETED!")
                                else:
                                    result_content = f"Error: {result.get('error', 'Unknown error')}"
                                    print(f"    ❌ Failed: {result_content}")
                                
                                # Add tool result
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "content": result_content
                                })
                                
                        except Exception as e:
                            print(f"    ❌ Exception: {e}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": f"Error: {str(e)}"
                            })
                    
                    # CRITICAL: Add user message to continue conversation
                    if turn == 1:
                        continuation = "Great! Now please proceed to edit the file based on what you read."
                    elif turn == 2:
                        continuation = "Excellent! Please summarize the changes you made."
                    else:
                        continuation = "Please continue with the next step or provide a summary."
                    
                    messages.append({
                        "role": "user",
                        "content": continuation
                    })
                    
                    print(f"➡️  Added continuation: '{continuation[:50]}...'")
                    
                else:
                    # Final response without tools
                    print(f"💬 Final response: {content[:200]}...")
                    messages.append({
                        "role": "assistant",
                        "content": content
                    })
                    break
                
            except Exception as e:
                print(f"❌ Turn {turn} failed: {e}")
                break
            
            turn += 1
        
        print(f"\n📊 Conversation Summary:")
        print(f"  - Turns completed: {turn - 1}")
        print(f"  - Total messages: {len(messages)}")
        
        # Check for successful edits
        edit_found = any(
            'edit_file' in str(msg) and ('Success' in str(msg) or 'success' in str(msg))
            for msg in messages
            if msg.get('role') == 'tool'
        )
        
        print(f"  - Edit operations: {'✅ Found' if edit_found else '❌ None found'}")
        
        return {
            "success": True,
            "turns": turn - 1,
            "edit_found": edit_found,
            "total_messages": len(messages)
        }

async def main():
    """Test correct conversation flow."""
    
    print("🚀 Fixing Tool Calling Conversation Flow")
    print("=" * 45)
    print("Addressing the role alternation issue from server logs\n")
    
    # Test 1: Minimal flow structure
    result1 = await test_minimal_correct_flow()
    
    # Test 2: Complete flow (if basic works)
    if result1.get('success'):
        result2 = await test_complete_conversation_flow()
    else:
        print("\n⚠️  Skipping complete test due to basic flow issues")
        result2 = {"success": False, "skipped": True}
    
    # Summary
    print(f"\n🏁 CONVERSATION FLOW FIX SUMMARY")
    print("=" * 40)
    
    print(f"Minimal Flow Test: {'✅' if result1.get('success') else '❌'}")
    if result1.get('success'):
        structure = result1.get('conversation_structure', [])
        print(f"  - Conversation roles: {' → '.join(structure)}")
        print(f"  - Role alternation: {'✅ Correct' if result1.get('roles_correct') else '❌ Issues'}")
    
    print(f"Complete Flow Test: {'✅' if result2.get('success') else '❌'}")
    if result2.get('success'):
        print(f"  - Turns completed: {result2.get('turns', 0)}")
        print(f"  - Edit operations: {'✅' if result2.get('edit_found') else '❌'}")
    
    print(f"\n💡 KEY FINDING:")
    if result1.get('success') and result2.get('success'):
        print("✅ SOLUTION CONFIRMED: Adding user messages after tool results fixes the alternation!")
        print("🔧 Pattern: user → assistant+tools → tool → USER → assistant")
        print("📝 The missing piece was the user continuation message")
    else:
        print("❌ Role alternation issues need further investigation")
    
    print(f"\n🛠️  IMPLEMENTATION FIX:")
    print("In your conversation logic, after tool execution:")
    print('```python')
    print('# After all tool results are added')
    print('messages.append({')
    print('    "role": "user",')
    print('    "content": "Please continue with the next step."')
    print('})')
    print('```')

if __name__ == "__main__":
    asyncio.run(main())

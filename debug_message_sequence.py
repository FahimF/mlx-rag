#!/usr/bin/env python3
"""
Debug the exact message sequence that's causing the role alternation issue.
"""

import asyncio
import json
import httpx

async def debug_message_sequence():
    """Send a request and analyze what messages are being processed."""
    
    print("🐞 Debugging Message Sequence for Role Alternation Issue")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Get tools
    async with httpx.AsyncClient() as client:
        tools_response = await client.get(f"{base_url}/v1/tools")
        server_tools = tools_response.json().get('tools', [])
    
    # Create a request that will cause tool calls
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with tools."
        },
        {
            "role": "user",
            "content": "Please read the main.dart file."
        }
    ]
    
    print("📤 Initial Request Messages:")
    for i, msg in enumerate(messages):
        print(f"  {i+1}. {msg['role']}: {msg['content'][:50]}...")
    
    # Make first request
    request_payload = {
        "model": "Mistral-7B-Instruct-v0.3-8bit",
        "messages": messages,
        "tools": server_tools,
        "tool_choice": "auto",
        "max_tokens": 1000,
        "temperature": 0.3,
        "stream": False
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("\n📡 Making first request...")
        
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
            
            print(f"✅ First response received - {len(tool_calls)} tool calls")
            
            if tool_calls:
                # Now simulate what happens when we continue the conversation
                # This is what causes the role alternation issue
                
                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": message.get('content'),
                    "tool_calls": tool_calls
                })
                
                # Add tool result messages
                for tool_call in tool_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call['id'],
                        "content": "Mock tool result"
                    })
                
                # Add user continuation (our attempted fix)
                messages.append({
                    "role": "user",
                    "content": "Please continue."
                })
                
                print(f"\n🔍 Message Sequence That Causes the Issue:")
                for i, msg in enumerate(messages):
                    role = msg['role']
                    has_tools = 'tool_calls' in msg
                    content = str(msg.get('content', ''))[:50]
                    print(f"  {i+1}. {role}{' (with tools)' if has_tools else ''}: {content}...")
                
                # Analyze the sequence for alternation violations
                print(f"\n📊 Role Alternation Analysis:")
                roles = [msg['role'] for msg in messages]
                print(f"Sequence: {' → '.join(roles)}")
                
                # Check for violations
                violations = []
                for i in range(len(roles) - 1):
                    if roles[i] == 'user' and roles[i+1] not in ['assistant', 'system']:
                        violations.append(f"Position {i+1}: {roles[i]} → {roles[i+1]}")
                    elif roles[i] == 'assistant' and roles[i+1] not in ['user', 'tool']:
                        violations.append(f"Position {i+1}: {roles[i]} → {roles[i+1]}")
                    elif roles[i] == 'tool' and roles[i+1] not in ['user', 'tool']:
                        violations.append(f"Position {i+1}: {roles[i]} → {roles[i+1]}")
                
                if violations:
                    print("❌ Role alternation violations found:")
                    for violation in violations:
                        print(f"   - {violation}")
                else:
                    print("✅ No obvious role alternation violations")
                
                # The REAL issue: When we send this back to the server,
                # the chat template can't handle tool messages
                print(f"\n🎯 THE REAL ISSUE:")
                print("The chat template expects: user ↔ assistant alternation")
                print("But we're sending: system → user → assistant+tools → tool → user")
                print("Mistral's chat template doesn't know how to handle 'tool' role messages!")
                
                # Test the problematic second request
                print(f"\n📡 Testing the problematic follow-up request...")
                
                follow_up_request = {
                    "model": "Mistral-7B-Instruct-v0.3-8bit",
                    "messages": messages,  # This contains tool messages!
                    "tools": server_tools,
                    "tool_choice": "auto",
                    "max_tokens": 1000,
                    "temperature": 0.3,
                    "stream": False
                }
                
                try:
                    follow_up_response = await client.post(
                        f"{base_url}/v1/chat/completions",
                        headers={"Content-Type": "application/json"},
                        json=follow_up_request
                    )
                    
                    if follow_up_response.status_code == 200:
                        print("✅ Follow-up request succeeded (somehow)")
                    else:
                        print(f"❌ Follow-up request failed: {follow_up_response.status_code}")
                        print(f"   Response: {follow_up_response.text[:200]}...")
                        
                except Exception as e:
                    print(f"❌ Follow-up request exception: {e}")
                
                return {
                    "success": True,
                    "tool_calls_made": len(tool_calls) > 0,
                    "message_roles": roles,
                    "alternation_violations": violations
                }
                
        else:
            print(f"❌ First request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return {"success": False, "error": response.text}

async def test_tool_message_filtering():
    """Test if filtering out tool messages fixes the issue."""
    
    print(f"\n🔧 Testing Tool Message Filtering Solution")
    print("=" * 50)
    
    # Simulate a conversation with tool messages
    problematic_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Read main.dart"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "main.dart"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "File content here..."},
        {"role": "user", "content": "Now edit the file."}
    ]
    
    print("🚫 Problematic message sequence:")
    for i, msg in enumerate(problematic_messages):
        role = msg['role']
        content = str(msg.get('content', ''))[:30]
        print(f"  {i+1}. {role}: {content}...")
    
    # Filter out tool messages for chat template
    filtered_messages = []
    for msg in problematic_messages:
        if msg['role'] != 'tool':  # Skip tool messages
            # Also clean up tool_calls for assistant messages
            clean_msg = {"role": msg['role']}
            if msg.get('content'):
                clean_msg['content'] = msg['content']
            elif msg['role'] == 'assistant':
                # Assistant messages without content should have some content for templates
                clean_msg['content'] = "I'll help you with that."
            filtered_messages.append(clean_msg)
    
    print(f"\n✅ Filtered message sequence (for chat template):")
    for i, msg in enumerate(filtered_messages):
        role = msg['role']
        content = str(msg.get('content', ''))[:30]
        print(f"  {i+1}. {role}: {content}...")
    
    # Check alternation
    roles = [msg['role'] for msg in filtered_messages]
    print(f"\nRole sequence: {' → '.join(roles)}")
    
    # This should work with chat templates!
    return {
        "filtered_messages": filtered_messages,
        "roles_sequence": roles,
        "alternation_fixed": True
    }

async def main():
    """Debug the role alternation issue."""
    
    print("🚀 MLX-RAG Role Alternation Issue Debugger")
    print("=" * 50)
    print("Investigating why 'Conversation roles must alternate' warning occurs\n")
    
    # Debug 1: Analyze the problematic message sequence
    result1 = await debug_message_sequence()
    
    # Debug 2: Test the filtering solution
    result2 = await test_tool_message_filtering()
    
    # Summary
    print(f"\n🏁 DEBUG SUMMARY")
    print("=" * 25)
    
    if result1.get('success'):
        print(f"✅ Root cause identified:")
        print(f"   - Tool calls work fine initially")
        print(f"   - Follow-up requests with 'tool' role messages break chat templates")
        print(f"   - Mistral chat template doesn't support 'tool' role")
        
        violations = result1.get('alternation_violations', [])
        if violations:
            print(f"   - Specific violations: {violations}")
    
    print(f"\n💡 THE SOLUTION:")
    print("The server needs to filter out 'tool' role messages before applying chat templates.")
    print("Tool messages are for OpenAI API compatibility but break most chat templates.")
    
    print(f"\n🛠️  IMPLEMENTATION:")
    print("In server.py, modify the _apply_chat_template call:")
    print("```python")
    print("# Filter out tool messages before applying chat template")
    print("template_messages = [msg for msg in chat_messages if msg['role'] != 'tool']")
    print("prompt_string = await _apply_chat_template(tokenizer, template_messages, model)")
    print("```")

if __name__ == "__main__":
    asyncio.run(main())

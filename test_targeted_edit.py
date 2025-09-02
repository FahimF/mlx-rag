#!/usr/bin/env python3
"""
Targeted test that focuses on specific sections of the file for editing.
"""

import asyncio
import httpx
import json

BASE_URL = "http://localhost:8000"

async def test_targeted_edit():
    """Test with targeted approach for large file editing."""
    
    print("🧪 Testing targeted approach for file editing...")
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
        
        # Step 2: Targeted approach - focus on specific UI elements
        print("\n💬 Making targeted request focusing on UI layout...")
        
        # More targeted request that focuses on UI patterns
        targeted_request = {
            "model": target_model['name'],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a Flutter developer with file editing tools. "
                        "When working with large code files:\\n"
                        "1. First search for specific patterns to locate relevant sections\\n"
                        "2. Read specific sections of code rather than the entire file\\n"
                        "3. Make targeted edits to the relevant sections\\n"
                        "4. Focus on the task at hand - don't get distracted by the entire codebase\\n\\n"
                        "Available tools: search_files, read_file, edit_file"
                    )
                },
                {
                    "role": "user", 
                    "content": (
                        "I need to modify the Flutter UI in main.dart. "
                        "Please find the section that contains the action buttons layout "
                        "(likely in a Row or Column widget with ElevatedButton widgets) "
                        "and modify it to:\\n"
                        "1. Space out the buttons vertically instead of horizontally\\n" 
                        "2. Add Text widgets showing the picked books folder path and Calibre library path\\n\\n"
                        "Use search_files to find the relevant UI code sections first, "
                        "then read just those sections, then edit them."
                    )
                }
            ],
            "tools": tools,
            "max_tokens": 4000,
            "temperature": 0.1  # Very low temperature for focused behavior
        }
        
        print("📤 Making targeted request...")
        
        try:
            # Execute the conversation
            messages = targeted_request["messages"]
            turn_count = 0
            max_turns = 8
            
            # Initial request
            response = await client.post(
                f"{BASE_URL}/v1/chat/completions",
                json=targeted_request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                print(f"❌ Initial request failed: {response.text}")
                return
                
            data = response.json()
            
            while turn_count < max_turns:
                turn_count += 1
                print(f"\\n🔄 Turn {turn_count}")
                
                if 'choices' not in data or len(data['choices']) == 0:
                    print("❌ No response choices")
                    break
                    
                choice = data['choices'][0]
                message = choice.get('message', {})
                
                # Add assistant's response to conversation
                messages.append(message)
                
                # Check for tool calls
                if 'tool_calls' in message and message['tool_calls']:
                    print(f"✅ {len(message['tool_calls'])} tool call(s)\")\n                    \n                    has_edit_call = any(tc['function']['name'] == 'edit_file' for tc in message['tool_calls'])\n                    has_write_call = any(tc['function']['name'] == 'write_file' for tc in message['tool_calls'])\n                    \n                    if has_edit_call or has_write_call:\n                        print("🎯 Found editing tool call!\")\n                    \n                    # Execute each tool call\n                    tool_results = []\n                    for i, tool_call in enumerate(message['tool_calls']):\n                        func_name = tool_call['function']['name']\n                        func_args = tool_call['function']['arguments']\n                        \n                        print(f\"  🔧 {i+1}. {func_name}\")\n                        \n                        # Show different details based on tool type\n                        if func_name == 'search_files':\n                            args_dict = json.loads(func_args)\n                            print(f\"     Searching for: {args_dict.get('query', 'N/A')}\")\n                        elif func_name == 'read_file':\n                            args_dict = json.loads(func_args)\n                            print(f\"     Reading: {args_dict.get('path', 'N/A')}\")\n                            if 'start_line' in args_dict or 'end_line' in args_dict:\n                                print(f\"     Lines: {args_dict.get('start_line', 1)}-{args_dict.get('end_line', 'end')}\")\n                        elif func_name == 'edit_file':\n                            args_dict = json.loads(func_args)\n                            print(f\"     Editing: {args_dict.get('path', 'N/A')}\")\n                            print(f\"     Lines: {args_dict.get('start_line', '?')}-{args_dict.get('end_line', '?')}\")\n                            print(f\"     📝 ACTUAL EDIT HAPPENING! 🎉\")\n                        elif func_name == 'write_file':\n                            args_dict = json.loads(func_args)\n                            print(f\"     Writing: {args_dict.get('path', 'N/A')}\")\n                            print(f\"     📝 ACTUAL WRITE HAPPENING! 🎉\")\n                        \n                        tool_request = {\n                            \"tool_call_id\": tool_call['id'],\n                            \"function_name\": func_name,\n                            \"arguments\": func_args\n                        }\n                        \n                        try:\n                            tool_response = await client.post(\n                                f\"{BASE_URL}/v1/tools/execute\",\n                                json=tool_request,\n                                headers={\"Content-Type\": \"application/json\"}\n                            )\n                            \n                            if tool_response.status_code == 200:\n                                tool_result = tool_response.json()\n                                success = tool_result.get('success', False)\n                                print(f\"     ✅ Success: {success}\")\n                                \n                                if success:\n                                    result_content = tool_result.get('result', '')\n                                    if isinstance(result_content, dict):\n                                        if 'total_found' in result_content:  # Search results\n                                            print(f\"       Found {result_content['total_found']} matches\")\n                                            if result_content['total_found'] > 0:\n                                                for result in result_content.get('results', [])[:3]:  # Show first 3\n                                                    print(f\"         {result.get('file', '')}: line {result.get('line', '')}\")\n                                        elif 'lines_read' in result_content:  # File read\n                                            print(f\"       Read {result_content['lines_read']} lines\")\n                                        elif 'bytes_written' in result_content:  # File write\n                                            print(f\"       ✅ Wrote {result_content['bytes_written']} bytes\")\n                                        elif 'lines_replaced' in result_content:  # File edit\n                                            print(f\"       ✅ Replaced {result_content['lines_replaced']} lines\")\n                                            print(f\"       📁 File successfully modified!\")\n                                else:\n                                    print(f\"       ❌ Error: {tool_result.get('error', 'Unknown')}\")\n                                \n                                # Format result for conversation\n                                result_content = tool_result.get('result', tool_result.get('error', ''))\n                                if isinstance(result_content, dict) or isinstance(result_content, list):\n                                    result_content = json.dumps(result_content)\n                                \n                                tool_results.append({\n                                    \"tool_call_id\": tool_call['id'],\n                                    \"role\": \"tool\", \n                                    \"content\": result_content\n                                })\n                            else:\n                                print(f\"     ❌ Tool failed: {tool_response.text}\")\n                                tool_results.append({\n                                    \"tool_call_id\": tool_call['id'],\n                                    \"role\": \"tool\",\n                                    \"content\": f\"Error: {tool_response.text}\"\n                                })\n                                \n                        except Exception as e:\n                            print(f\"     ❌ Exception: {e}\")\n                            tool_results.append({\n                                \"tool_call_id\": tool_call['id'],\n                                \"role\": \"tool\",\n                                \"content\": f\"Exception: {e}\"\n                            })\n                    \n                    # Add tool results to conversation\n                    messages.extend(tool_results)\n                    \n                    # Check if we got an actual edit - if so, we might be done\n                    if has_edit_call or has_write_call:\n                        success_edits = [tr for tr in tool_results if \n                                       \"lines_replaced\" in tr[\"content\"] or \"bytes_written\" in tr[\"content\"]]\n                        if success_edits:\n                            print(f\"\\n🎉 SUCCESS! File was actually modified!\")\n                            print(\"\\n💬 Getting final response...\")\n                    \n                    # Continue conversation\n                    continue_request = {\n                        \"model\": target_model['name'],\n                        \"messages\": messages,\n                        \"tools\": tools,\n                        \"max_tokens\": 2000,\n                        \"temperature\": 0.1\n                    }\n                    \n                    try:\n                        response = await client.post(\n                            f\"{BASE_URL}/v1/chat/completions\",\n                            json=continue_request,\n                            headers={\"Content-Type\": \"application/json\"}\n                        )\n                        \n                        if response.status_code == 200:\n                            data = response.json()\n                            continue  # Continue the loop\n                        else:\n                            print(f\"❌ Continue request failed: {response.text}\")\n                            break\n                            \n                    except Exception as e:\n                        print(f\"❌ Continue request exception: {e}\")\n                        break\n                \n                else:\n                    # No more tool calls, show final response\n                    if 'content' in message and message['content']:\n                        print(\"\\n🎉 Final Response:\")\n                        print(message['content'])\n                    else:\n                        print(\"❌ No content in final response\")\n                    break\n            \n            if turn_count >= max_turns:\n                print(f\"⚠️ Reached maximum turns ({max_turns})\")\n                \n        except Exception as e:\n            print(f\"❌ Request exception: {e}\")\n            return\n        \n        print(\"\\n🏁 Targeted test completed!\")\n\nif __name__ == \"__main__\":\n    asyncio.run(test_targeted_edit())"

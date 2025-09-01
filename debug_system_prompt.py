#!/usr/bin/env python3
"""
Debug script to test system prompt generation for tool calling.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_system_prompt_generation():
    """Test the system prompt being generated for tools."""
    print("🔧 TESTING SYSTEM PROMPT GENERATION")
    print("=" * 50)
    
    # Get tools from server
    tools_response = requests.get(f"{BASE_URL}/v1/tools")
    if tools_response.status_code != 200:
        print("❌ Could not get tools")
        return
    
    tools_data = tools_response.json()
    tools = tools_data.get('tools', [])
    
    print(f"✅ Found {len(tools)} tools")
    
    # Test the system prompt generation locally
    try:
        from mlx_rag.tool_prompts import generate_tool_system_prompt, generate_contextual_prompt
        
        print("\n" + "=" * 50)
        print("BASIC SYSTEM PROMPT")
        print("=" * 50)
        
        basic_prompt = generate_tool_system_prompt(tools)
        print(basic_prompt)
        
        print("\n" + "=" * 50)
        print("CONTEXTUAL SYSTEM PROMPT")  
        print("=" * 50)
        
        user_query = "Modify main.dart file to space out the action buttons vertically"
        contextual_prompt = generate_contextual_prompt(tools, user_query)
        print(contextual_prompt)
        
        print("\n" + "=" * 50)
        print("CHECKING FOR JSON FORMAT INSTRUCTIONS")
        print("=" * 50)
        
        # Check if the prompt contains the JSON format instructions
        if '{"function": "tool_name", "arguments": {"parameter": "value"}}' in basic_prompt:
            print("✅ JSON format instructions found in basic prompt")
        else:
            print("❌ JSON format instructions NOT found in basic prompt")
            
        if '{"function"' in basic_prompt and '"arguments"' in basic_prompt:
            print("✅ JSON structure mentioned in basic prompt")
        else:
            print("❌ JSON structure not mentioned in basic prompt")
            
        # Look for the tool call format section
        if "Tool Call Format" in basic_prompt:
            print("✅ 'Tool Call Format' section found")
        else:
            print("❌ 'Tool Call Format' section not found")
        
        # Extract just the format instructions
        if "## Tool Call Format" in basic_prompt:
            format_start = basic_prompt.index("## Tool Call Format")
            format_section = basic_prompt[format_start:format_start+500]  # Get next 500 chars
            print(f"\n📋 FORMAT INSTRUCTIONS SECTION:\n{format_section}")
        
        print(f"\n📊 PROMPT STATISTICS:")
        print(f"   - Length: {len(basic_prompt)} characters")
        print(f"   - Lines: {basic_prompt.count('newline') + 1}")
        print(f"   - Contains 'JSON': {'JSON' in basic_prompt}")
        print(f"   - Contains 'function': {'function' in basic_prompt}")
        print(f"   - Contains 'arguments': {'arguments' in basic_prompt}")
        
        return basic_prompt
        
    except ImportError as e:
        print(f"❌ Could not import tool_prompts: {e}")
        return None
    except Exception as e:
        print(f"❌ Error generating system prompt: {e}")
        return None

def test_actual_chat_request():
    """Test what system prompt is actually sent in a chat request."""
    print("\n" + "=" * 60)
    print("TESTING ACTUAL CHAT REQUEST SYSTEM PROMPT")
    print("=" * 60)
    
    # Make a request identical to our debug script to see what's sent
    tools_response = requests.get(f"{BASE_URL}/v1/tools")
    tools = tools_response.json().get('tools', []) if tools_response.status_code == 200 else []
    
    request_data = {
        "model": "DeepSeek-R1-0528-Qwen3-8B-MLX-4bit",
        "messages": [
            {
                "role": "user", 
                "content": "Please use the read_file tool to read main.dart"
            }
        ],
        "max_tokens": 100,  # Short response to see if tools are called
        "temperature": 0.1,
        "stream": False,
        "tools": tools,
        "tool_choice": "auto"
    }
    
    print(f"🔧 Sending request with {len(tools)} tools")
    print(f"🔧 User message: {request_data['messages'][0]['content']}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            choice = data.get('choices', [{}])[0]
            message = choice.get('message', {})
            
            print(f"\n📨 RESPONSE:")
            print(f"   - Content: {message.get('content', 'None')[:200]}...")
            print(f"   - Tool calls: {len(message.get('tool_calls', []))}")
            print(f"   - Finish reason: {choice.get('finish_reason', 'None')}")
            
            if message.get('tool_calls'):
                print("✅ Model made tool calls!")
                for i, call in enumerate(message['tool_calls']):
                    print(f"   {i+1}. {call.get('function', {}).get('name', 'Unknown')}")
            else:
                print("❌ Model did not make tool calls")
                print(f"   Full content: {message.get('content', '')}")
                
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception during request: {e}")

if __name__ == "__main__":
    system_prompt = test_system_prompt_generation()
    test_actual_chat_request()
    
    print(f"\n🎯 DEBUGGING SUMMARY:")
    print(f"   - Tools available: Yes")
    print(f"   - System prompt generated: {'Yes' if system_prompt else 'No'}")
    print(f"   - Format instructions included: {'Yes' if system_prompt and 'JSON' in system_prompt else 'No'}")
    print(f"   - Models following instructions: Test above to see")

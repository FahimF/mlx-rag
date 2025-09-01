#!/usr/bin/env python3
"""
Test different system prompts to try to get GLM-4.5-Air-4bit to use function calling.
"""

import requests
import json

BASE_URL = "http://localhost:8000"
MODEL_NAME = "DeepSeek-R1-0528-Qwen3-8B-MLX-4bit"
TEST_PROMPT = "Find the main.dart file and read its contents."

def test_function_calling_prompts():
    """Test different system prompts to encourage function calling."""
    
    # Get tools first
    tools_response = requests.get(f"{BASE_URL}/v1/tools")
    tools_list = tools_response.json().get('tools', []) if tools_response.status_code == 200 else []
    
    if not tools_list:
        print("No tools available")
        return
    
    # Test prompts in order of directness
    test_prompts = [
        # 1. Very direct OpenAI format instruction
        {
            "name": "OpenAI Format Instructions",
            "content": """You are a helpful assistant that can use functions. When you need to perform actions, you should call the appropriate functions using the OpenAI function calling format.

You have access to these functions:
- list_directory: List files and directories
- read_file: Read file contents  
- search_files: Search for text patterns
- write_file: Write content to files
- edit_file: Edit file sections

IMPORTANT: To call a function, you MUST use the proper OpenAI function calling format. Do NOT write Python code blocks. Instead, make actual function calls through the tool_calls mechanism.

When a user asks you to work with files, use the functions immediately."""
        },
        
        # 2. Example-based instruction
        {
            "name": "Example-Based Instructions", 
            "content": """You are an assistant with access to file system tools. When users ask about files, you should use the available functions.

Available functions:
- list_directory() - to explore directories
- read_file(path) - to read file contents
- search_files(query) - to find files/content

Example: If user asks "read main.dart", you should call read_file with path="main.dart".

Always use functions instead of guessing or assuming."""
        },
        
        # 3. Minimal instruction
        {
            "name": "Minimal Instructions",
            "content": "You have file system tools available. Use them when working with files."
        }
    ]
    
    for i, prompt_config in enumerate(test_prompts):
        print(f"\n{'='*60}")
        print(f"TEST {i+1}: {prompt_config['name']}")
        print(f"{'='*60}")
        
        messages = [
            {"role": "system", "content": prompt_config["content"]},
            {"role": "user", "content": TEST_PROMPT}
        ]
        
        request_body = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.3,
            "stream": False,
            "tools": tools_list,
            "tool_choice": "auto"
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=request_body,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                choice = data.get('choices', [{}])[0]
                message = choice.get('message', {})
                
                if message.get('tool_calls'):
                    print(f"✅ SUCCESS! Model requested {len(message['tool_calls'])} tool calls:")
                    for j, tool_call in enumerate(message['tool_calls']):
                        print(f"  {j+1}. {tool_call.get('function', {}).get('name', 'Unknown')}")
                        print(f"     Args: {tool_call.get('function', {}).get('arguments', 'None')}")
                    return True  # Success!
                else:
                    print("❌ No tool calls - Response:")
                    print(f"   {message.get('content', 'No content')[:200]}...")
                    
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    return False

if __name__ == "__main__":
    print("🧪 TESTING DIFFERENT FUNCTION CALLING PROMPTS")
    print(f"Model: {MODEL_NAME}")
    print(f"Test Query: {TEST_PROMPT}")
    
    success = test_function_calling_prompts()
    
    if not success:
        print(f"\n💡 CONCLUSION: GLM-4.5-Air-4bit may not properly support OpenAI function calling format.")
        print("Consider trying a different model that's known to support function calling.")
        print("Popular options: GPT models, Claude models, or other models specifically trained for function calling.")

#!/usr/bin/env python3
"""
Debug script specifically for testing Qwen3-Coder-30B model tool calling.
This will test different prompt strategies and response parsing.
"""

import asyncio
import httpx
import json
import re
from datetime import datetime

BASE_URL = "http://localhost:8000"
WORKING_MODEL = "Qwen2.5-Coder-7B-Instruct-MLX-4bit"  # Known working model
PROBLEM_MODEL = "Qwen3-Coder-30B-A3B-Instruct-4bit"   # Model that doesn't work
TEST_PROMPT = "Read the main.dart file"

async def test_model_comparison():
    """Compare responses between working and non-working models."""
    print("🔍 COMPARING MODEL TOOL CALLING BEHAVIOR")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Get tools
        tools_response = await client.get(f"{BASE_URL}/v1/tools")
        tools = tools_response.json().get('tools', []) if tools_response.status_code == 200 else []
        
        if not tools:
            print("❌ No tools available")
            return
            
        models_to_test = [WORKING_MODEL, PROBLEM_MODEL]
        
        for model_name in models_to_test:
            print(f"\n{'='*30} {model_name} {'='*30}")
            
            # Test different prompt styles
            prompt_styles = [
                {
                    "name": "Direct Tool Instruction",
                    "system": f"""You have access to file tools. When asked to work with files, you MUST use the appropriate tools.

Available tools:
{chr(10).join([f"- {tool['function']['name']}: {tool['function']['description']}" for tool in tools])}

IMPORTANT: Always use tools when asked to work with files. Respond with proper tool calls.""",
                    "user": TEST_PROMPT
                },
                {
                    "name": "OpenAI Format Example",
                    "system": f"""You are an assistant with file access tools. Use the OpenAI function calling format.

Tools available: {', '.join([tool['function']['name'] for tool in tools])}

When you need to use a tool, make a proper function call. Do not write code - use the tools.""",
                    "user": TEST_PROMPT
                },
                {
                    "name": "JSON Format Instruction",
                    "system": f"""You can use these tools by making function calls:

{chr(10).join([f"- {tool['function']['name']}" for tool in tools])}

To use a tool, respond with:
{{"function": "tool_name", "arguments": {{"param": "value"}}}}

Use tools when working with files.""",
                    "user": TEST_PROMPT
                },
                {
                    "name": "Minimal Instruction",
                    "system": "You have file tools available. Use them when working with files.",
                    "user": TEST_PROMPT
                }
            ]
            
            for prompt_style in prompt_styles:
                print(f"\n  📋 Testing: {prompt_style['name']}")
                
                messages = [
                    {"role": "system", "content": prompt_style["system"]},
                    {"role": "user", "content": prompt_style["user"]}
                ]
                
                request_body = {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": 1024,
                    "temperature": 0.3,
                    "stream": False,
                    "tools": tools,
                    "tool_choice": "auto"
                }
                
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
                        content = message.get('content', '')
                        
                        if tool_calls:
                            print(f"    ✅ SUCCESS: {len(tool_calls)} tool calls")
                            for call in tool_calls:
                                func_name = call.get('function', {}).get('name', 'Unknown')
                                print(f"      - {func_name}")
                        else:
                            print(f"    ❌ NO TOOLS: Text response instead")
                            print(f"      Response: {content[:100]}...")
                            
                            # Try to parse text response for tool intentions
                            if _contains_tool_intention(content):
                                print(f"      📝 Contains tool intention but not properly formatted")
                    else:
                        print(f"    ❌ HTTP Error: {response.status_code}")
                        
                except Exception as e:
                    print(f"    ❌ Exception: {e}")

def _contains_tool_intention(text):
    """Check if text contains intention to use tools."""
    intentions = [
        "read", "file", "examine", "check", "look at", "tool", "function",
        "main.dart", "explore", "analyze"
    ]
    text_lower = text.lower()
    return any(intention in text_lower for intention in intentions)

async def test_qwen3_specific_formats():
    """Test Qwen3-specific prompt formats that might work better."""
    print(f"\n🧪 TESTING QWEN3-SPECIFIC FORMATS")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Get tools
        tools_response = await client.get(f"{BASE_URL}/v1/tools")
        tools = tools_response.json().get('tools', []) if tools_response.status_code == 200 else []
        
        # Qwen3-specific prompt styles
        qwen_prompts = [
            {
                "name": "Qwen3 Function Format",
                "system": """You are a coding assistant with access to file system functions.

Functions available:
- read_file(path): Read file contents
- list_directory(path): List directory contents  
- search_files(query, path): Search for text in files

When you need to work with files, call the appropriate function using this format:
<function_call>
<name>function_name</name>
<arguments>{"param": "value"}</arguments>
</function_call>""",
                "user": "Read the main.dart file to see its contents"
            },
            {
                "name": "Qwen3 Tool Format",
                "system": """<|im_start|>system
You are a helpful assistant with file tools. Use tools when needed.

Available tools:
- read_file: Read file contents
- list_directory: List files
- search_files: Search content

To use a tool, output: <tool_call>tool_name:{"arguments"}</tool_call>
<|im_end|>""",
                "user": "Read main.dart"
            },
            {
                "name": "Qwen3 Step-by-Step",
                "system": """You help users with file operations. You have these tools:
- read_file
- list_directory  
- search_files

When asked to work with files:
1. Think about what tool to use
2. Call the tool with proper JSON arguments
3. Analyze the result

Use standard OpenAI function calling format.""",
                "user": "I need to read the main.dart file. Please use the appropriate tool."
            }
        ]
        
        for prompt_config in qwen_prompts:
            print(f"\n  📋 Testing: {prompt_config['name']}")
            
            messages = [
                {"role": "system", "content": prompt_config["system"]},
                {"role": "user", "content": prompt_config["user"]}
            ]
            
            request_body = {
                "model": PROBLEM_MODEL,
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.1,  # Lower temperature for more deterministic behavior
                "stream": False,
                "tools": tools,
                "tool_choice": "auto"
            }
            
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
                    content = message.get('content', '')
                    
                    if tool_calls:
                        print(f"    ✅ SUCCESS: {len(tool_calls)} tool calls")
                        for call in tool_calls:
                            func_name = call.get('function', {}).get('name', 'Unknown')
                            args = call.get('function', {}).get('arguments', '{}')
                            print(f"      - {func_name}({args})")
                        return True  # Found a working format!
                    else:
                        print(f"    ❌ NO TOOLS: {content[:200]}...")
                        
                        # Check if response contains alternative tool formats we can parse
                        alt_formats = _extract_alternative_tool_formats(content)
                        if alt_formats:
                            print(f"    🔍 Found alternative format: {alt_formats}")
                        
                else:
                    print(f"    ❌ HTTP Error: {response.status_code}")
                    
            except Exception as e:
                print(f"    ❌ Exception: {e}")
    
    return False

def _extract_alternative_tool_formats(text):
    """Look for alternative tool calling formats in the response."""
    patterns = [
        r'<function_call>\s*<name>(.*?)</name>\s*<arguments>(.*?)</arguments>\s*</function_call>',
        r'<tool_call>(.*?):(.*?)</tool_call>',
        r'```(\w+)\s*(.*?)```',
        r'function:\s*(\w+)\s*arguments:\s*({.*?})',
        r'call:\s*(\w+)\((.*?)\)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            return {"pattern": pattern, "matches": matches}
    
    return None

async def test_manual_tool_calling():
    """Test if we can manually trigger tool calling by parsing the response."""
    print(f"\n🛠️  TESTING MANUAL TOOL CALL PARSING")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Get tools
        tools_response = await client.get(f"{BASE_URL}/v1/tools")
        tools = tools_response.json().get('tools', []) if tools_response.status_code == 200 else []
        
        # Simple request without forcing tools
        messages = [
            {"role": "user", "content": "Read the main.dart file and tell me what's in it"}
        ]
        
        request_body = {
            "model": PROBLEM_MODEL,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.3,
            "stream": False
            # No tools in request - let model respond naturally
        }
        
        try:
            response = await client.post(
                f"{BASE_URL}/v1/chat/completions",
                json=request_body
            )
            
            if response.status_code == 200:
                data = response.json()
                choice = data.get('choices', [{}])[0]
                message = choice.get('message', {})
                content = message.get('content', '')
                
                print(f"Raw response: {content}")
                
                # Try to parse tool intentions from the response
                from mlx_rag.server import _parse_tool_calls_from_text
                detected_calls = _parse_tool_calls_from_text(content)
                
                if detected_calls:
                    print(f"\n✅ DETECTED {len(detected_calls)} tool calls from response:")
                    for call in detected_calls:
                        func_name = call.get('function', {}).get('name', 'Unknown')
                        args = call.get('function', {}).get('arguments', '{}')
                        print(f"  - {func_name}({args})")
                    
                    # Now execute the detected tool calls
                    print(f"\n🔧 Executing detected tools...")
                    for call in detected_calls:
                        tool_response = await client.post(
                            f"{BASE_URL}/v1/tools/execute",
                            json={
                                "tool_call_id": call["id"],
                                "function_name": call["function"]["name"],
                                "arguments": call["function"]["arguments"]
                            }
                        )
                        
                        if tool_response.status_code == 200:
                            tool_result = tool_response.json()
                            print(f"  ✅ {call['function']['name']}: Success")
                            print(f"     Result: {str(tool_result.get('result', ''))[:100]}...")
                        else:
                            print(f"  ❌ {call['function']['name']}: Failed")
                    
                    return True
                else:
                    print(f"❌ No tool calls detected in response")
                    return False
            
        except Exception as e:
            print(f"❌ Exception: {e}")
            return False

async def main():
    """Main test function."""
    print(f"🔍 QWEN3-CODER-30B TOOL CALLING DEBUG")
    print(f"Time: {datetime.now()}")
    print("=" * 60)
    
    print(f"Working Model: {WORKING_MODEL}")
    print(f"Problem Model: {PROBLEM_MODEL}")
    print(f"Test Query: {TEST_PROMPT}")
    
    # Test 1: Compare models with different prompts
    await test_model_comparison()
    
    # Test 2: Try Qwen3-specific formats
    qwen_success = await test_qwen3_specific_formats()
    
    # Test 3: Manual parsing approach
    manual_success = await test_manual_tool_calling()
    
    print(f"\n📊 RESULTS:")
    print(f"  - Qwen3-specific formats: {'✅ Success' if qwen_success else '❌ Failed'}")
    print(f"  - Manual tool parsing: {'✅ Success' if manual_success else '❌ Failed'}")
    
    if manual_success:
        print(f"\n💡 SOLUTION:")
        print(f"  The Qwen3-Coder-30B model doesn't follow OpenAI tool calling format,")
        print(f"  but we can parse its natural language responses to extract tool intentions.")
        print(f"  The existing _parse_tool_calls_from_text function should handle this.")
    elif qwen_success:
        print(f"\n💡 SOLUTION:")
        print(f"  Found a working prompt format for Qwen3-Coder-30B model.")
    else:
        print(f"\n🚨 CONCLUSION:")
        print(f"  Qwen3-Coder-30B may not be compatible with tool calling.")
        print(f"  Consider using {WORKING_MODEL} for tool-enabled tasks.")

if __name__ == "__main__":
    asyncio.run(main())

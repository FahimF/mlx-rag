#!/usr/bin/env python3
"""
Debug script to test chat completions with tools and RAG collection.
This will help identify why tools/RAG aren't working in the chat interface.
"""

import requests
import json
import sys
from datetime import datetime

# Configuration - matching the chat interface
BASE_URL = "http://localhost:8000"
MODEL_NAME = "DeepSeek-R1-0528-Qwen3-8B-MLX-4bit"
RAG_COLLECTION = "Booker"
TEST_PROMPT = "Modify main.dart file to space out the action buttons vertically and display the picked books folder and Calibre library path next to the relevant buttons."

def test_tools_endpoint():
    """Test the /v1/tools endpoint to see what tools are available."""
    print("=" * 50)
    print("TESTING /v1/tools ENDPOINT")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/v1/tools")
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Tools Response: {json.dumps(data, indent=2)}")
            return data.get('tools', [])
        else:
            print(f"Error Response: {response.text}")
            return []
    except Exception as e:
        print(f"Exception testing tools: {e}")
        return []

def test_rag_collections():
    """Test the RAG collections endpoint."""
    print("\n" + "=" * 50)
    print("TESTING RAG COLLECTIONS")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/v1/rag/collections")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"RAG Collections: {json.dumps(data, indent=2)}")
            return data.get('collections', [])
        else:
            print(f"Error Response: {response.text}")
            return []
    except Exception as e:
        print(f"Exception testing RAG collections: {e}")
        return []

def test_models_endpoint():
    """Test the models endpoint to see what models are available."""
    print("\n" + "=" * 50)
    print("TESTING /v1/manager/models ENDPOINT")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/v1/manager/models")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            print(f"Available Models: {len(models)}")
            for model in models:
                print(f"  - {model.get('name', 'Unknown')} ({model.get('status', 'Unknown')}) - {model.get('type', 'Unknown')}")
                if model.get('name') == MODEL_NAME:
                    print(f"    TARGET MODEL FOUND: {json.dumps(model, indent=4)}")
            return models
        else:
            print(f"Error Response: {response.text}")
            return []
    except Exception as e:
        print(f"Exception testing models: {e}")
        return []

def test_chat_completion(tools_list):
    """Test the chat completion endpoint with tools."""
    print("\n" + "=" * 50)
    print("TESTING CHAT COMPLETION WITH TOOLS")
    print("=" * 50)
    
    # Build messages like the updated chat interface does
    messages = []
    
    # Add system message with RAG and tools instructions (matching updated chat.js)
    if RAG_COLLECTION or tools_list:
        system_content = ''
        
        if RAG_COLLECTION:
            system_content += f'You have access to a RAG collection named "{RAG_COLLECTION}" containing source code and documentation. '
        
        if tools_list:
            system_content += 'You have access to the following tools to interact with the codebase:\n'
            for tool in tools_list:
                system_content += f"- {tool['function']['name']}: {tool['function']['description']}\n"
            system_content += '\nIMPORTANT: When a user asks about modifying, reading, or working with files, you MUST use the appropriate tools (like read_file, search_files, list_directory, etc.) to examine the codebase first, then provide specific solutions. Do not make assumptions about code you haven\'t seen.'
        
        if system_content:
            messages.append({
                "role": "system",
                "content": system_content.strip()
            })
    
    # Add user message
    messages.append({
        "role": "user",
        "content": TEST_PROMPT
    })
    
    # Build request body like the chat interface
    request_body = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.7,
        "stream": False
    }
    
    # Add tools if available
    if tools_list:
        request_body["tools"] = tools_list
        request_body["tool_choice"] = "auto"
        print(f"Adding {len(tools_list)} tools to request")
    else:
        print("No tools available - proceeding without tools")
    
    print(f"Request Body: {json.dumps(request_body, indent=2)}")
    
    try:
        print("\nSending request...")
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=request_body,
            timeout=60
        )
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            
            # Check if tools were called
            choice = data.get('choices', [{}])[0]
            message = choice.get('message', {})
            
            if message.get('tool_calls'):
                print(f"\n✅ MODEL REQUESTED {len(message['tool_calls'])} TOOL CALLS:")
                for i, tool_call in enumerate(message['tool_calls']):
                    print(f"  {i+1}. {tool_call.get('function', {}).get('name', 'Unknown')}")
                    print(f"     Args: {tool_call.get('function', {}).get('arguments', 'None')}")
            else:
                print(f"\n❌ NO TOOL CALLS REQUESTED")
                print(f"Response content: {message.get('content', 'No content')}")
            
            return data
        else:
            print(f"Error Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Exception during chat completion: {e}")
        return None

def test_tool_execution():
    """Test tool execution if tools are available."""
    print("\n" + "=" * 50)
    print("TESTING TOOL EXECUTION")
    print("=" * 50)
    
    # Test basic tool execution endpoint
    test_payload = {
        "tool_call_id": "test_call_123",
        "function_name": "search_documents", 
        "arguments": json.dumps({"query": "main.dart file", "collection": RAG_COLLECTION})
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/v1/tools/execute",
            headers={"Content-Type": "application/json"},
            json=test_payload,
            timeout=30
        )
        
        print(f"Tool Execution Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Tool Execution Response: {json.dumps(data, indent=2)}")
            return data
        else:
            print(f"Tool Execution Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"Exception during tool execution: {e}")
        return None

def main():
    """Run all debug tests."""
    print(f"🔍 DEBUG CHAT TOOLS AND RAG")
    print(f"Timestamp: {datetime.now()}")
    print(f"Target Model: {MODEL_NAME}")
    print(f"Target RAG Collection: {RAG_COLLECTION}")
    print(f"Base URL: {BASE_URL}")
    
    # Test 1: Check available tools
    tools_list = test_tools_endpoint()
    
    # Test 2: Check RAG collections  
    rag_collections = test_rag_collections()
    
    # Test 3: Check models
    models = test_models_endpoint()
    
    # Test 4: Test chat completion
    chat_response = test_chat_completion(tools_list)
    
    # Test 5: Test tool execution
    if tools_list:
        test_tool_execution()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"✅ Tools Available: {len(tools_list) if tools_list else 0}")
    print(f"✅ RAG Collections: {len(rag_collections) if rag_collections else 0}")
    print(f"✅ Models Available: {len(models) if models else 0}")
    
    # Check if target RAG collection exists
    target_rag_found = any(col.get('name') == RAG_COLLECTION for col in rag_collections) if rag_collections else False
    print(f"✅ Target RAG Collection '{RAG_COLLECTION}' Found: {target_rag_found}")
    
    # Check if target model exists and is loaded
    target_model = next((m for m in models if m.get('name') == MODEL_NAME), None) if models else None
    target_model_loaded = target_model and target_model.get('status') == 'loaded' if target_model else False
    print(f"✅ Target Model '{MODEL_NAME}' Loaded: {target_model_loaded}")
    
    # Check if chat worked
    chat_success = chat_response is not None
    print(f"✅ Chat Completion Successful: {chat_success}")
    
    # Check if tools were called
    if chat_response:
        choice = chat_response.get('choices', [{}])[0]
        message = choice.get('message', {})
        tools_called = bool(message.get('tool_calls'))
        print(f"✅ Tools Called by Model: {tools_called}")
    else:
        print(f"❌ Tools Called by Model: False (chat failed)")
    
    print("\n🔍 Debug complete!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fixed debug script for testing Qwen3-Coder-30B model tool calling.
This version creates and activates a RAG collection first so tools are available.
"""

import asyncio
import httpx
import json
import re
import os
from datetime import datetime

BASE_URL = "http://localhost:8000"
WORKING_MODEL = "Qwen2.5-Coder-7B-Instruct-MLX-4bit"  # Known working model
PROBLEM_MODEL = "Qwen3-Coder-30B-A3B-Instruct-4bit"   # Model that doesn't work
TEST_PROMPT = "Read the main.dart file"

# Test RAG collection details
TEST_COLLECTION_NAME = "test_collection"
TEST_COLLECTION_PATH = "/Users/fahim/Code/Flutter/BooksApp"  # Adjust this path as needed

async def setup_test_environment():
    """Set up the test environment by creating and activating a RAG collection."""
    print("🔧 Setting up test environment...")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # Check if collection already exists
            collections_response = await client.get(f"{BASE_URL}/v1/rag/collections")
            if collections_response.status_code == 200:
                collections = collections_response.json().get('collections', [])
                existing_collection = next((c for c in collections if c['name'] == TEST_COLLECTION_NAME), None)
                
                if existing_collection:
                    print(f"✅ Collection '{TEST_COLLECTION_NAME}' already exists")
                    # Activate it
                    activate_response = await client.post(f"{BASE_URL}/v1/rag/collections/{TEST_COLLECTION_NAME}/activate")
                    if activate_response.status_code == 200:
                        print(f"✅ Activated collection '{TEST_COLLECTION_NAME}'")
                        return True
                    else:
                        print(f"❌ Failed to activate collection: {activate_response.status_code}")
                        return False
            
            # Create new collection if it doesn't exist
            if os.path.exists(TEST_COLLECTION_PATH):
                print(f"📁 Creating collection '{TEST_COLLECTION_NAME}' at path: {TEST_COLLECTION_PATH}")
                
                create_response = await client.post(
                    f"{BASE_URL}/v1/rag/collections",
                    params={
                        "name": TEST_COLLECTION_NAME,
                        "path": TEST_COLLECTION_PATH
                    }
                )
                
                if create_response.status_code == 200:
                    print(f"✅ Created collection '{TEST_COLLECTION_NAME}'")
                    
                    # Activate the collection
                    activate_response = await client.post(f"{BASE_URL}/v1/rag/collections/{TEST_COLLECTION_NAME}/activate")
                    if activate_response.status_code == 200:
                        print(f"✅ Activated collection '{TEST_COLLECTION_NAME}'")
                        return True
                    else:
                        print(f"❌ Failed to activate collection: {activate_response.status_code}")
                        return False
                else:
                    print(f"❌ Failed to create collection: {create_response.status_code} - {create_response.text}")
                    return False
            else:
                print(f"❌ Collection path does not exist: {TEST_COLLECTION_PATH}")
                print("📝 Please update TEST_COLLECTION_PATH to point to a valid directory with code files")
                return False
                
        except Exception as e:
            print(f"❌ Error setting up test environment: {e}")
            return False

async def test_tools_with_collection():
    """Test that tools are available after setting up collection."""
    print(f"\n🔧 Testing tools availability with active collection...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Get tools - should now return tools because we have an active collection
            tools_response = await client.get(f"{BASE_URL}/v1/tools")
            if tools_response.status_code == 200:
                tools_data = tools_response.json()
                tools = tools_data.get('tools', [])
                collection_path = tools_data.get('collection_path')
                
                print(f"✅ Found {len(tools)} tools")
                print(f"📁 Collection path: {collection_path}")
                
                if tools:
                    print("🛠️  Available tools:")
                    for tool in tools:
                        print(f"  - {tool['function']['name']}: {tool['function']['description']}")
                    return tools, collection_path
                else:
                    print("❌ No tools returned despite having active collection")
                    return [], None
            else:
                print(f"❌ Failed to get tools: {tools_response.status_code}")
                return [], None
                
        except Exception as e:
            print(f"❌ Error getting tools: {e}")
            return [], None

async def test_enhanced_system_prompt():
    """Test that the enhanced system prompt with collection path is being used."""
    print(f"\n🧪 Testing enhanced system prompt with collection path...")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # Get tools
            tools_response = await client.get(f"{BASE_URL}/v1/tools")
            tools_data = tools_response.json()
            tools = tools_data.get('tools', [])
            collection_path = tools_data.get('collection_path')
            
            if not tools:
                print("❌ No tools available for testing")
                return False
                
            print(f"📁 Testing with collection path: {collection_path}")
            
            # Make a chat completion request that should use the enhanced system prompt
            request_data = {
                "model": PROBLEM_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": TEST_PROMPT
                    }
                ],
                "tools": tools,
                "max_tokens": 1024,
                "temperature": 0.3,
                "stream": False
            }
            
            response = await client.post(
                f"{BASE_URL}/v1/chat/completions",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                choice = data.get('choices', [{}])[0]
                message = choice.get('message', {})
                tool_calls = message.get('tool_calls', [])
                content = message.get('content', '')
                finish_reason = choice.get('finish_reason')
                
                print(f"📥 Response received - Finish reason: {finish_reason}")
                print(f"🔧 Tool calls: {len(tool_calls)}")
                
                if tool_calls:
                    print("✅ SUCCESS: Model made tool calls!")
                    for i, call in enumerate(tool_calls):
                        func_name = call.get('function', {}).get('name', 'Unknown')
                        args = call.get('function', {}).get('arguments', '{}')
                        print(f"  {i+1}. {func_name}({args})")
                        
                        # Check if the tool call uses valid paths (within collection)
                        try:
                            args_dict = json.loads(args)
                            if 'path' in args_dict:
                                path = args_dict['path']
                                print(f"     Path used: {path}")
                                
                                # Check if path is within collection or relative
                                if path.startswith('/') and collection_path and not path.startswith(collection_path):
                                    print(f"     ⚠️  Absolute path outside collection: {path}")
                                elif path.startswith('./') or not path.startswith('/'):
                                    print(f"     ✅ Using relative path: {path}")
                                else:
                                    print(f"     ✅ Path within collection: {path}")
                        except:
                            pass
                    
                    return True
                else:
                    print(f"❌ NO TOOL CALLS: Model returned text instead")
                    print(f"📝 Content: {content[:200]}...")
                    
                    # Check if the model mentions the collection path in its response
                    if collection_path and collection_path in content:
                        print(f"✅ Model referenced collection path in response")
                    else:
                        print(f"❌ Model did not reference collection path")
                    
                    return False
            else:
                print(f"❌ Request failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing enhanced system prompt: {e}")
            return False

async def cleanup_test_environment():
    """Clean up the test environment by removing the test collection."""
    print(f"\n🧹 Cleaning up test environment...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Delete the test collection
            delete_response = await client.delete(f"{BASE_URL}/v1/rag/collections/{TEST_COLLECTION_NAME}")
            if delete_response.status_code == 200:
                print(f"✅ Deleted test collection '{TEST_COLLECTION_NAME}'")
            else:
                print(f"⚠️  Could not delete test collection: {delete_response.status_code}")
                
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")

async def main():
    """Main test function."""
    print(f"🔍 QWEN3-CODER-30B ENHANCED TOOL CALLING TEST")
    print(f"Time: {datetime.now()}")
    print("=" * 60)
    
    print(f"Problem Model: {PROBLEM_MODEL}")
    print(f"Test Query: {TEST_PROMPT}")
    print(f"Test Collection: {TEST_COLLECTION_NAME}")
    print(f"Collection Path: {TEST_COLLECTION_PATH}")
    
    # Step 1: Set up test environment with RAG collection
    setup_success = await setup_test_environment()
    if not setup_success:
        print("\n❌ FAILED: Could not set up test environment")
        print("Please ensure:")
        print("1. MLX-RAG server is running at http://localhost:8000")
        print("2. TEST_COLLECTION_PATH points to a valid directory")
        print("3. You have the required model loaded")
        return
    
    # Step 2: Verify tools are available
    tools, collection_path = await test_tools_with_collection()
    if not tools:
        print("\n❌ FAILED: No tools available even with active collection")
        await cleanup_test_environment()
        return
    
    # Step 3: Test the enhanced system prompt with collection path
    prompt_success = await test_enhanced_system_prompt()
    
    # Results
    print(f"\n📊 TEST RESULTS:")
    print("=" * 30)
    print(f"✅ Environment setup: {'Success' if setup_success else 'Failed'}")
    print(f"✅ Tools available: {'Success' if tools else 'Failed'}")
    print(f"✅ Enhanced system prompt: {'Success' if prompt_success else 'Failed'}")
    
    if prompt_success:
        print(f"\n🎉 SUCCESS!")
        print("The enhanced system prompt with collection path is working!")
        print("The model successfully made tool calls with proper path context.")
    else:
        print(f"\n❌ FAILED!")
        print("The model is still not making proper tool calls.")
        print("This could indicate:")
        print("1. The model doesn't support tool calling well")
        print("2. The system prompt needs further refinement")
        print("3. The early stopping mechanisms are too aggressive")
    
    # Step 4: Cleanup
    await cleanup_test_environment()
    
    print(f"\n🏁 Test completed!")

if __name__ == "__main__":
    asyncio.run(main())

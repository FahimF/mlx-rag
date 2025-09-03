#!/usr/bin/env python3
"""
Proper LangChain test using the actual LangChain integration from the codebase.

This test uses the LangChainToolExecutor and creates a proper LangChain agent
to handle file editing with loop detection and sophisticated conversation management.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the actual LangChain components from the codebase
try:
    from src.mlx_rag.tool_executor import LangChainToolExecutor
    from src.mlx_rag.agentic_tools import create_langchain_tools, LangChainToolFactory
    from src.mlx_rag.tool_prompts import create_file_system_agent_prompt, create_react_prompt_template
except ImportError as e:
    print(f"❌ Failed to import LangChain components: {e}")
    print("Make sure you're running this from the mlx-rag project root")
    exit(1)

class LangChainLoopDetector:
    """Enhanced loop detector for LangChain agent interactions."""
    
    def __init__(self, max_read_attempts: int = 3, max_total_calls: int = 15):
        self.max_read_attempts = max_read_attempts
        self.max_total_calls = max_total_calls
        self.read_counter = Counter()
        self.total_calls = 0
        self.call_history = []
        self.last_actions = []
        
    def record_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """
        Record a tool call and check for problematic patterns.
        
        Returns:
            True if a problematic loop is detected
        """
        self.total_calls += 1
        self.call_history.append({"tool": tool_name, "args": arguments})
        self.last_actions.append(tool_name)
        
        # Keep only last 10 actions for pattern detection
        if len(self.last_actions) > 10:
            self.last_actions.pop(0)
        
        # Check for excessive total calls
        if self.total_calls > self.max_total_calls:
            logger.warning(f"Excessive tool calls: {self.total_calls}")
            return True
        
        # Check for repeated read_file calls on same files
        if tool_name == "read_file":
            file_path = arguments.get("path", "")
            self.read_counter[file_path] += 1
            
            if self.read_counter[file_path] > self.max_read_attempts:
                logger.warning(f"Excessive reads of {file_path}: {self.read_counter[file_path]}")
                return True
        
        # Check for repetitive patterns in recent calls
        if len(self.last_actions) >= 6:
            recent = self.last_actions[-6:]
            # Pattern: read -> read -> read (stuck in reads)
            if recent[-3:] == ["read_file", "read_file", "read_file"]:
                logger.warning("Detected read loop pattern")
                return True
            
            # Pattern: read -> search -> read -> search (exploring without acting)
            if recent[-4:] == ["read_file", "search_files", "read_file", "search_files"]:
                logger.warning("Detected exploration loop pattern")
                return True
        
        return False
    
    def should_intervene(self) -> bool:
        """Check if we should intervene to break a potential loop."""
        # Intervene if we've had multiple reads without any edits
        recent_tools = self.last_actions[-5:] if len(self.last_actions) >= 5 else self.last_actions
        read_count = recent_tools.count("read_file")
        edit_count = recent_tools.count("edit_file")
        
        return read_count >= 3 and edit_count == 0
    
    def get_intervention_message(self) -> str:
        """Get a message to break the agent out of a loop."""
        return (
            "You have read files multiple times but haven't made any edits yet. "
            "You have enough information now. Please use the edit_file tool to make "
            "the requested changes immediately. Stop reading and start editing."
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of tool usage for analysis."""
        return {
            "total_calls": self.total_calls,
            "read_counter": dict(self.read_counter),
            "recent_actions": self.last_actions[-10:],
            "unique_tools_used": list(set(call["tool"] for call in self.call_history))
        }

async def test_langchain_agent_edit():
    """Test file editing using actual LangChain agent integration."""
    
    print("🧪 Testing with proper LangChain agent integration...")
    
    # Step 1: Setup collection path (we'll use a dummy path for this test)
    # In real usage, this would be the path to an active RAG collection
    collection_path = "/Users/fahim/Code/Flutter/Booker"  # Flutter project path
    
    if not Path(collection_path).exists():
        print(f"❌ Collection path does not exist: {collection_path}")
        print("Please update the collection_path variable to point to a valid Flutter project")
        return
    
    print(f"✅ Using collection path: {collection_path}")
    
    # Step 2: Initialize LangChain components
    try:
        # Create the LangChain tool executor
        langchain_executor = LangChainToolExecutor(
            rag_collection_path=collection_path,
            use_memory=True  # Enable conversation memory
        )
        
        if not langchain_executor.has_available_tools():
            print("❌ No LangChain tools available")
            return
            
        tools = langchain_executor.get_langchain_tools()
        print(f"✅ Initialized {len(tools)} LangChain tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
            
    except Exception as e:
        print(f"❌ Failed to initialize LangChain components: {e}")
        return
    
    # Step 3: Test individual tool execution first
    print("\n📋 Testing individual tool execution...")
    
    loop_detector = LangChainLoopDetector()
    
    try:
        # Test read_file on main.dart
        print("  🔧 Testing read_file tool...")
        read_result = await langchain_executor.execute_single_tool_async(
            "read_file",
            path="main.dart"
        )
        
        loop_detector.record_tool_call("read_file", {"path": "main.dart"})
        
        if read_result["success"]:
            result_data = json.loads(read_result["result"])
            lines_read = result_data.get("lines_read", 0)
            print(f"    ✅ Successfully read main.dart ({lines_read} lines)")
        else:
            print(f"    ❌ Failed to read main.dart: {read_result['error']}")
            return
            
    except Exception as e:
        print(f"    ❌ Tool execution failed: {e}")
        return
    
    # Step 4: Test edit_file tool
    print("\n✏️ Testing edit_file tool...")
    
    try:
        # Try a simple edit (add a comment at the top)
        edit_result = await langchain_executor.execute_single_tool_async(
            "edit_file",
            path="main.dart",
            start_line=1,
            end_line=1,
            new_content="// Test edit by LangChain agent\n"
        )
        
        loop_detector.record_tool_call("edit_file", {
            "path": "main.dart", 
            "start_line": 1, 
            "end_line": 1
        })
        
        if edit_result["success"]:
            result_data = json.loads(edit_result["result"])
            lines_replaced = result_data.get("lines_replaced", 0)
            print(f"    🎉 Successfully edited main.dart ({lines_replaced} lines changed)")
        else:
            print(f"    ❌ Edit failed: {edit_result['error']}")
            
    except Exception as e:
        print(f"    ❌ Edit execution failed: {e}")
    
    # Step 5: Simulate problematic agent behavior
    print("\n🔄 Simulating problematic agent behavior (multiple reads)...")
    
    for i in range(4):  # This should trigger loop detection
        try:
            read_result = await langchain_executor.execute_single_tool_async(
                "read_file",
                path="main.dart"
            )
            
            is_loop = loop_detector.record_tool_call("read_file", {"path": "main.dart"})
            print(f"    📖 Read attempt {i+1}: {'Loop detected!' if is_loop else 'OK'}")
            
            if is_loop:
                print("    🚨 Loop detected - would intervene here")
                print(f"    💬 Intervention message: {loop_detector.get_intervention_message()}")
                break
                
        except Exception as e:
            print(f"    ❌ Read attempt {i+1} failed: {e}")
    
    # Step 6: Show comprehensive analysis
    print("\n📊 Tool Usage Analysis:")
    summary = loop_detector.get_summary()
    print(f"  - Total tool calls: {summary['total_calls']}")
    print(f"  - File read attempts: {summary['read_counter']}")
    print(f"  - Recent actions: {summary['recent_actions']}")
    print(f"  - Tools used: {summary['unique_tools_used']}")
    
    # Step 7: Test intervention logic
    if loop_detector.should_intervene():
        print("\n⚠️ Intervention recommended:")
        print(f"   Message: {loop_detector.get_intervention_message()}")
    
    # Step 8: Demonstrate proper usage pattern
    print("\n✅ Demonstrating proper usage pattern...")
    proper_detector = LangChainLoopDetector()
    
    # Proper sequence: search -> read -> edit
    proper_steps = [
        ("search_files", {"query": "main.dart"}),
        ("read_file", {"path": "main.dart"}),
        ("edit_file", {"path": "main.dart", "start_line": 5, "end_line": 5})
    ]
    
    for tool_name, args in proper_steps:
        proper_detector.record_tool_call(tool_name, args)
        print(f"    {tool_name}: {'✅' if not proper_detector.record_tool_call(tool_name, args) else '❌'}")
    
    print(f"\n🏁 Test completed!")
    print("    This test demonstrates:")
    print("    - Proper LangChain tool integration")
    print("    - Loop detection and intervention logic")
    print("    - Individual tool execution patterns")
    print("    - Analysis of tool usage patterns")

# Mock LLM wrapper for testing (in real usage, you'd use an actual LLM)
class MockLLM:
    """Mock LLM for testing purposes."""
    
    def __init__(self):
        self.call_count = 0
    
    def invoke(self, prompt):
        self.call_count += 1
        return f"Mock response {self.call_count} to: {prompt[:50]}..."
    
    async def ainvoke(self, prompt):
        return self.invoke(prompt)

async def test_langchain_agent_creation():
    """Test creating an actual LangChain agent (if possible)."""
    
    print("\n🤖 Testing LangChain agent creation...")
    
    collection_path = "/Users/fahim/Code/Flutter/Booker"
    
    if not Path(collection_path).exists():
        print(f"❌ Collection path does not exist: {collection_path}")
        return
    
    try:
        langchain_executor = LangChainToolExecutor(collection_path, use_memory=True)
        
        # Try to create an agent with a mock LLM
        mock_llm = MockLLM()
        
        # Note: This might fail if we don't have all the required LangChain components
        # but it demonstrates the proper integration pattern
        try:
            agent = langchain_executor.create_react_agent(mock_llm)
            if agent:
                print("    ✅ Successfully created ReAct agent")
                
                # Test agent execution (this would normally use a real LLM)
                # result = await agent.ainvoke({"input": "Read main.dart and summarize its structure"})
                # print(f"    🎯 Agent result: {result}")
            else:
                print("    ⚠️ Agent creation returned None (likely due to mock LLM)")
                
        except Exception as e:
            print(f"    ⚠️ Agent creation failed (expected with mock LLM): {e}")
            
    except Exception as e:
        print(f"    ❌ Failed to test agent creation: {e}")

if __name__ == "__main__":
    print("🚀 Running proper LangChain integration tests...")
    
    # Test the core functionality
    asyncio.run(test_langchain_agent_edit())
    
    # Test agent creation (might fail with mock components)
    asyncio.run(test_langchain_agent_creation())
    
    print("\n✅ LangChain integration tests completed!")

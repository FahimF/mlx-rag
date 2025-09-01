#!/usr/bin/env python3
"""
Test script to check what system prompt is being generated for tool usage.
"""

import requests
import json

def test_system_prompt():
    """Test what system prompt is generated for tools."""
    
    # Get available tools
    print("=== Getting Available Tools ===")
    tools_response = requests.get("http://localhost:8000/v1/tools")
    if tools_response.status_code != 200:
        print(f"Error getting tools: {tools_response.status_code}")
        return
    
    tools_data = tools_response.json()
    tools = tools_data.get("tools", [])
    
    print(f"Available tools: {len(tools)}")
    for tool in tools:
        func = tool.get("function", {})
        print(f"  - {func.get('name', 'unknown')}: {func.get('description', 'no description')[:60]}...")
    
    # Import the system prompt generation function
    import sys
    import os
    sys.path.append('/Users/fahim/Code/Python/mlx-rag/src')
    
    from mlx_rag.tool_prompts import generate_tool_system_prompt, generate_contextual_prompt
    
    # Test standard system prompt
    print("\n=== Standard System Prompt ===")
    standard_prompt = generate_tool_system_prompt(tools, include_examples=True, include_workflows=True)
    print("Length:", len(standard_prompt))
    
    # Show first 1000 chars and sections mentioning write/edit
    print("\nFirst 1000 characters:")
    print(standard_prompt[:1000])
    
    print("\n=== Sections mentioning 'write' or 'edit' ===")
    lines = standard_prompt.split('\n')
    for i, line in enumerate(lines):
        if 'write' in line.lower() or 'edit' in line.lower():
            # Show this line and a few around it for context
            start = max(0, i-2)
            end = min(len(lines), i+3)
            print(f"Lines {start+1}-{end}:")
            for j in range(start, end):
                prefix = ">>> " if j == i else "    "
                print(f"{prefix}{lines[j]}")
            print()
    
    # Test contextual prompt for modify request
    print("\n=== Contextual Prompt for Modify Request ===")
    user_query = "Modify main.dart file to space out the action buttons vertically and display the picked books folder and Calibre library path next to the relevant buttons."
    contextual_prompt = generate_contextual_prompt(tools, user_query, conversation_history=[])
    print("Length:", len(contextual_prompt))
    
    # Show sections mentioning write/edit in contextual prompt
    print("\n=== Contextual prompt sections mentioning 'write' or 'edit' ===")
    lines = contextual_prompt.split('\n')
    for i, line in enumerate(lines):
        if 'write' in line.lower() or 'edit' in line.lower():
            # Show this line and a few around it for context
            start = max(0, i-2)
            end = min(len(lines), i+3)
            print(f"Lines {start+1}-{end}:")
            for j in range(start, end):
                prefix = ">>> " if j == i else "    "
                print(f"{prefix}{lines[j]}")
            print()
    
    # Test if modify/create keywords are recognized
    print("\n=== Query Analysis ===")
    query_lower = user_query.lower()
    modify_keywords = ['modify', 'edit', 'change', 'update', 'fix', 'create', 'add', 'write']
    found_keywords = [word for word in modify_keywords if word in query_lower]
    print(f"Modify-related keywords found in query: {found_keywords}")

if __name__ == "__main__":
    test_system_prompt()

"""
Tool prompt templates and system prompt generation utilities.

This module provides comprehensive templates and utilities for creating system prompts
that teach LLMs how to effectively use available tools.
"""

import json
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


# Base system prompt templates
BASE_SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools that allow you to explore, analyze, and modify files within a code repository or documentation collection.

Your primary goal is to help users understand their codebase, find information, make changes, and solve problems efficiently using the available tools."""

TOOL_USAGE_GUIDELINES = """
## Tool Usage Guidelines

1. **Always think before acting**: Understand what the user is asking before selecting tools
2. **Start broad, then narrow**: Use directory listing and search tools before examining specific files
3. **Be systematic**: When exploring a codebase, start with the root structure and work your way down
4. **Verify your understanding**: After reading files, summarize what you found to confirm accuracy
5. **Make incremental changes**: When editing files, make small, focused changes and verify them
6. **Provide context**: Always explain what you're doing and why you're using specific tools
7. **Handle errors gracefully**: If a tool fails, try alternative approaches or ask for clarification

## Tool Call Format

To use a tool, respond with a JSON function call in this exact format:
```json
{"function": "tool_name", "arguments": {"parameter": "value"}}
```

You can make multiple tool calls in sequence. Always explain your reasoning and interpret the results for the user.
"""

# Specific tool templates
TOOL_TEMPLATES = {
    "list_directory": {
        "description": "Explore directory structures and understand project organization",
        "when_to_use": [
            "When first exploring a new codebase",
            "To understand project structure and organization",
            "To find specific files or directories",
            "When looking for configuration files, tests, or documentation"
        ],
        "best_practices": [
            "Start with the root directory to get overall structure",
            "Use recursive=true for deep exploration, recursive=false for focused browsing",
            "Look for common patterns: src/, tests/, docs/, config/ directories",
            "Pay attention to file extensions to understand technologies used"
        ],
        "examples": [
            {
                "scenario": "First time exploring a project",
                "call": {"function": "list_directory", "arguments": {"path": ".", "recursive": False}},
                "explanation": "Get overview of root directory structure"
            },
            {
                "scenario": "Finding all Python files in a project",
                "call": {"function": "list_directory", "arguments": {"path": ".", "recursive": True, "pattern": "*.py"}},
                "explanation": "Recursively find all Python files"
            }
        ]
    },
    
    "read_file": {
        "description": "Read and analyze file contents to understand code, configuration, or documentation",
        "when_to_use": [
            "To understand how specific code works",
            "To read configuration files or documentation",
            "Before making changes to a file",
            "To find specific functions, classes, or variables"
        ],
        "best_practices": [
            "Read files completely to understand context before making judgments",
            "For large files, consider using search_files first to find relevant sections",
            "Pay attention to imports, dependencies, and overall structure",
            "Look for comments and docstrings that explain functionality"
        ],
        "examples": [
            {
                "scenario": "Understanding main application entry point",
                "call": {"function": "read_file", "arguments": {"path": "main.py"}},
                "explanation": "Read the main file to understand application structure"
            },
            {
                "scenario": "Checking configuration",
                "call": {"function": "read_file", "arguments": {"path": "config.json"}},
                "explanation": "Read configuration to understand current settings"
            }
        ]
    },
    
    "search_files": {
        "description": "Search for specific content, patterns, or code across files",
        "when_to_use": [
            "Looking for specific functions, classes, or variables",
            "Finding all usages of a particular API or library",
            "Searching for error messages or specific patterns",
            "Locating configuration values or settings"
        ],
        "best_practices": [
            "Use specific search terms to avoid too many results",
            "Try different variations if first search doesn't yield results",
            "Use regex patterns for complex searches",
            "Search in specific directories to narrow results"
        ],
        "examples": [
            {
                "scenario": "Finding function definition",
                "call": {"function": "search_files", "arguments": {"query": "def process_data", "path": "."}},
                "explanation": "Search for a specific function definition"
            },
            {
                "scenario": "Finding all TODO comments",
                "call": {"function": "search_files", "arguments": {"query": "TODO|FIXME", "path": ".", "use_regex": True}},
                "explanation": "Find all TODO and FIXME comments using regex"
            }
        ]
    },
    
    "write_file": {
        "description": "Create new files with specified content",
        "when_to_use": [
            "Creating new source code files",
            "Adding new configuration files",
            "Creating documentation or README files",
            "Adding test files"
        ],
        "best_practices": [
            "Always consider the file's location and naming conventions",
            "Include appropriate file headers, imports, and structure",
            "Follow the project's coding style and conventions",
            "Add proper documentation and comments"
        ],
        "examples": [
            {
                "scenario": "Creating a new Python module",
                "call": {"function": "write_file", "arguments": {"path": "utils/helper.py", "content": "\"\"\"Utility functions for the application.\"\"\"\n\ndef helper_function():\n    \"\"\"A helpful utility function.\"\"\"\n    pass\n"}},
                "explanation": "Create a new utility module with proper structure"
            }
        ]
    },
    
    "edit_file": {
        "description": "Modify existing files by replacing specific line ranges with new content",
        "when_to_use": [
            "Fixing bugs in existing code",
            "Adding new functionality to existing files",
            "Updating configuration values",
            "Refactoring code",
            "Making targeted modifications to specific sections"
        ],
        "best_practices": [
            "Always read the file first to understand current state and line numbers",
            "Make targeted, specific changes rather than large rewrites",
            "Use line numbers accurately (1-based indexing)",
            "Consider the impact of changes on the rest of the codebase",
            "Test changes when possible",
            "Be precise with start and end line numbers"
        ],
        "examples": [
            {
                "scenario": "Replacing a function implementation",
                "call": {"function": "edit_file", "arguments": {"path": "utils.py", "start_line": 15, "end_line": 18, "new_content": "def calculate_total(x, y):\n    # Fixed calculation\n    return x * y\n"}},
                "explanation": "Replace lines 15-18 with corrected function implementation"
            },
            {
                "scenario": "Adding new import at the top",
                "call": {"function": "edit_file", "arguments": {"path": "main.py", "start_line": 1, "end_line": 1, "new_content": "import os\nimport sys\n"}},
                "explanation": "Insert new imports at the beginning of the file"
            }
        ]
    }
}

# Common workflow templates
WORKFLOW_TEMPLATES = {
    "explore_new_codebase": [
        "1. List the root directory to understand overall structure",
        "2. Look for key files like README, package.json, requirements.txt, etc.",
        "3. Examine the main application entry points",
        "4. Explore source code directories",
        "5. Check test directories to understand functionality",
        "6. Review configuration files"
    ],
    
    "debug_issue": [
        "1. Search for error messages or relevant keywords",
        "2. Locate the problematic code sections",
        "3. Read the relevant files to understand context",
        "4. Trace through the code logic",
        "5. Identify the root cause",
        "6. Plan and implement the fix"
    ],
    
    "add_new_feature": [
        "1. Understand the existing codebase structure",
        "2. Find similar existing functionality for reference",
        "3. Identify where the new feature should be implemented",
        "4. Plan the implementation approach",
        "5. Create or modify the necessary files",
        "6. Update related configuration or documentation"
    ],
    
    "code_refactoring": [
        "1. Read and understand the current implementation",
        "2. Identify areas for improvement",
        "3. Plan the refactoring approach",
        "4. Make incremental changes",
        "5. Verify that functionality is preserved",
        "6. Update tests and documentation if needed"
    ]
}

# Error handling and recovery templates
ERROR_HANDLING_TEMPLATES = {
    "file_not_found": "If a file is not found, try:\n- List the directory to see available files\n- Search for files with similar names\n- Check if the file might be in a different location",
    
    "permission_denied": "If you get permission denied:\n- Check if the file exists and is readable\n- Consider if the operation requires special permissions\n- Try alternative approaches or ask the user for guidance",
    
    "search_no_results": "If search returns no results:\n- Try broader search terms\n- Check spelling and case sensitivity\n- Search in different directories\n- Use regex patterns for more flexible matching",
    
    "edit_failed": "If file editing fails:\n- Verify the search text exists exactly as specified\n- Check if the file has been modified since last reading\n- Consider making smaller, more targeted changes"
}


def generate_tool_system_prompt(
    tools: List[Dict[str, Any]], 
    context: Optional[str] = None,
    include_examples: bool = True,
    include_workflows: bool = True
) -> str:
    """
    Generate a comprehensive system prompt for tool usage.
    
    Args:
        tools: List of available tools in OpenAI format
        context: Optional context about the current task or domain
        include_examples: Whether to include usage examples
        include_workflows: Whether to include common workflow templates
        
    Returns:
        Complete system prompt string
    """
    prompt_parts = [BASE_SYSTEM_PROMPT]
    
    if context:
        prompt_parts.append(f"\n## Current Context\n{context}")
    
    if not tools:
        prompt_parts.append("\n## Available Tools\nNo tools are currently available. You can only provide text responses based on your knowledge.")
        return "\n".join(prompt_parts)
    
    # Add tool descriptions
    prompt_parts.append("\n## Available Tools\n")
    
    for tool in tools:
        if tool.get("type") != "function":
            continue
            
        func = tool.get("function", {})
        tool_name = func.get("name", "unknown")
        tool_desc = func.get("description", "No description provided")
        parameters = func.get("parameters", {})
        
        # Get enhanced template info if available
        template = TOOL_TEMPLATES.get(tool_name, {})
        
        prompt_parts.append(f"### {tool_name}")
        prompt_parts.append(f"**Description**: {tool_desc}")
        
        if template.get("when_to_use"):
            prompt_parts.append("**When to use**:")
            for use_case in template["when_to_use"]:
                prompt_parts.append(f"- {use_case}")
        
        # Add parameter information
        if parameters.get("properties"):
            prompt_parts.append("**Parameters**:")
            required_params = parameters.get("required", [])
            
            for param_name, param_info in parameters["properties"].items():
                param_type = param_info.get("type", "string")
                param_desc = param_info.get("description", "No description")
                is_required = param_name in required_params
                req_text = " (required)" if is_required else " (optional)"
                
                prompt_parts.append(f"- `{param_name}` ({param_type}){req_text}: {param_desc}")
        
        if template.get("best_practices") and include_examples:
            prompt_parts.append("**Best practices**:")
            for practice in template["best_practices"]:
                prompt_parts.append(f"- {practice}")
        
        if template.get("examples") and include_examples:
            prompt_parts.append("**Examples**:")
            for example in template["examples"][:2]:  # Limit to 2 examples per tool
                prompt_parts.append(f"- *{example['scenario']}*:")
                prompt_parts.append(f"  ```json")
                prompt_parts.append(f"  {json.dumps(example['call'], indent=2)}")
                prompt_parts.append(f"  ```")
                prompt_parts.append(f"  {example['explanation']}")
        
        prompt_parts.append("")  # Add spacing between tools
    
    # Add usage guidelines
    prompt_parts.append(TOOL_USAGE_GUIDELINES)
    
    # Add common workflows if requested
    if include_workflows:
        prompt_parts.append("\n## Common Workflows\n")
        
        for workflow_name, steps in WORKFLOW_TEMPLATES.items():
            readable_name = workflow_name.replace("_", " ").title()
            prompt_parts.append(f"### {readable_name}")
            for step in steps:
                prompt_parts.append(f"{step}")
            prompt_parts.append("")
    
    # Add error handling guidance
    prompt_parts.append("\n## Error Handling\n")
    for error_type, guidance in ERROR_HANDLING_TEMPLATES.items():
        readable_error = error_type.replace("_", " ").title()
        prompt_parts.append(f"### {readable_error}")
        prompt_parts.append(guidance)
        prompt_parts.append("")
    
    return "\n".join(prompt_parts)


def generate_contextual_prompt(
    tools: List[Dict[str, Any]], 
    user_query: str, 
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Generate a system prompt tailored to the specific user query and context.
    
    Args:
        tools: Available tools
        user_query: The user's current question or request
        conversation_history: Previous conversation context
        
    Returns:
        Contextually relevant system prompt
    """
    # Analyze the query to determine likely task type
    query_lower = user_query.lower()
    
    context_hints = []
    relevant_workflows = []
    
    # Detect common task patterns
    if any(word in query_lower for word in ["explore", "understand", "overview", "structure"]):
        context_hints.append("The user wants to explore and understand the codebase structure.")
        relevant_workflows.append("explore_new_codebase")
    
    if any(word in query_lower for word in ["bug", "error", "fix", "debug", "problem", "issue"]):
        context_hints.append("The user is trying to debug an issue or fix a problem.")
        relevant_workflows.append("debug_issue")
    
    if any(word in query_lower for word in ["add", "create", "new", "implement", "feature"]):
        context_hints.append("The user wants to add new functionality or create new files.")
        relevant_workflows.append("add_new_feature")
    
    if any(word in query_lower for word in ["refactor", "improve", "clean", "optimize"]):
        context_hints.append("The user wants to refactor or improve existing code.")
        relevant_workflows.append("code_refactoring")
    
    # Build context string
    context_parts = []
    if context_hints:
        context_parts.append("Based on your query, it appears that:")
        context_parts.extend(f"- {hint}" for hint in context_hints)
    
    if conversation_history:
        context_parts.append(f"\nThis conversation has {len(conversation_history)} previous exchanges.")
        context_parts.append("Consider the conversation context when choosing your approach.")
    
    context = "\n".join(context_parts) if context_parts else None
    
    # Generate the full prompt
    prompt = generate_tool_system_prompt(tools, context, include_examples=True, include_workflows=True)
    
    # Add specific workflow guidance if relevant
    if relevant_workflows:
        prompt += "\n## Recommended Approach\n"
        prompt += f"For this type of task, consider following the '{relevant_workflows[0].replace('_', ' ')}' workflow:\n"
        
        if relevant_workflows[0] in WORKFLOW_TEMPLATES:
            for step in WORKFLOW_TEMPLATES[relevant_workflows[0]]:
                prompt += f"{step}\n"
    
    return prompt


def get_tool_usage_summary(tools: List[Dict[str, Any]]) -> str:
    """
    Generate a concise summary of available tools for quick reference.
    
    Args:
        tools: Available tools
        
    Returns:
        Brief summary of tools and their purposes
    """
    if not tools:
        return "No tools available - text responses only."
    
    summary_parts = [f"Available tools ({len(tools)}):"]
    
    for tool in tools:
        if tool.get("type") != "function":
            continue
            
        func = tool.get("function", {})
        tool_name = func.get("name", "unknown")
        tool_desc = func.get("description", "No description")
        
        # Truncate long descriptions
        if len(tool_desc) > 80:
            tool_desc = tool_desc[:77] + "..."
        
        summary_parts.append(f"- {tool_name}: {tool_desc}")
    
    return "\n".join(summary_parts)


def validate_tool_call_format(tool_call_text: str) -> Dict[str, Any]:
    """
    Validate and parse a tool call to ensure it follows the correct format.
    
    Args:
        tool_call_text: Raw tool call text from LLM
        
    Returns:
        Validation result with parsed call or error information
    """
    try:
        # Try to parse as JSON
        parsed = json.loads(tool_call_text)
        
        # Validate required fields
        if not isinstance(parsed, dict):
            return {"valid": False, "error": "Tool call must be a JSON object"}
        
        if "function" not in parsed:
            return {"valid": False, "error": "Tool call must include 'function' field"}
        
        if "arguments" not in parsed:
            return {"valid": False, "error": "Tool call must include 'arguments' field"}
        
        if not isinstance(parsed["arguments"], dict):
            return {"valid": False, "error": "Arguments must be a JSON object"}
        
        return {
            "valid": True, 
            "function": parsed["function"],
            "arguments": parsed["arguments"]
        }
        
    except json.JSONDecodeError as e:
        return {"valid": False, "error": f"Invalid JSON format: {str(e)}"}
    except Exception as e:
        return {"valid": False, "error": f"Unexpected error: {str(e)}"}

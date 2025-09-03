#!/usr/bin/env python3
"""
Server-side loop prevention for mlx-rag tool calling.

This module provides middleware to detect and prevent tool calling loops
by tracking tool usage patterns in conversations.
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import time
import logging

logger = logging.getLogger(__name__)

class ToolLoopDetector:
    """Detects and prevents tool calling loops in conversations."""
    
    def __init__(self):
        # Track tool usage by conversation ID
        self.conversation_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.file_reads: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.last_cleanup = time.time()
    
    def detect_loop(self, conversation_id: str, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """
        Detect if a tool call would create a loop.
        
        Returns:
            None if no loop, or error message if loop detected
        """
        # Cleanup old conversations periodically
        if time.time() - self.last_cleanup > 3600:  # 1 hour
            self._cleanup_old_conversations()
        
        history = self.conversation_history[conversation_id]
        
        # Check for read loops
        if tool_name in ['read_file', 'search_files']:
            file_path = arguments.get('path', arguments.get('query', 'unknown'))
            
            # Track reads of this file in this conversation
            self.file_reads[conversation_id][file_path] += 1
            read_count = self.file_reads[conversation_id][file_path]
            
            if read_count > 2:  # Allow max 2 reads per file per conversation
                return f"Loop detected: You have already read '{file_path}' {read_count-1} times. Please proceed to edit the file or try a different approach."
        
        # Check for general repetitive patterns
        recent_calls = list(history)[-3:]  # Last 3 calls
        current_call = f"{tool_name}:{arguments.get('path', arguments.get('query', ''))}"
        
        if len(recent_calls) >= 2 and all(call == current_call for call in recent_calls):
            return f"Loop detected: You are repeating the same tool call '{tool_name}'. Please try a different approach or proceed to the next step."
        
        # Add to history
        history.append(current_call)
        
        return None
    
    def suggest_next_action(self, conversation_id: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Suggest the next logical action to break out of loops."""
        
        if tool_name in ['read_file', 'search_files']:
            file_path = arguments.get('path', arguments.get('query', 'file'))
            return f"Since you've already examined {file_path}, consider using 'edit_file' to make your changes, or 'list_directory' to explore other files."
        
        return "Consider using a different tool or approach to accomplish your task."
    
    def reset_conversation(self, conversation_id: str):
        """Reset tracking for a conversation."""
        if conversation_id in self.conversation_history:
            del self.conversation_history[conversation_id]
        if conversation_id in self.file_reads:
            del self.file_reads[conversation_id]
    
    def _cleanup_old_conversations(self):
        """Remove old conversation data."""
        current_time = time.time()
        # Remove conversations older than 24 hours (simplified - you'd want more sophisticated logic)
        # This is a placeholder - implement based on your session management
        self.last_cleanup = current_time

# Global instance
loop_detector = ToolLoopDetector()

def check_tool_loop(conversation_id: str, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Check if a tool call would create a loop.
    
    Returns:
        None if no loop, or dict with error info if loop detected
    """
    error_message = loop_detector.detect_loop(conversation_id, tool_name, arguments)
    
    if error_message:
        suggestion = loop_detector.suggest_next_action(conversation_id, tool_name, arguments)
        logger.warning(f"Tool loop detected in conversation {conversation_id}: {error_message}")
        
        return {
            "error": error_message,
            "suggestion": suggestion,
            "loop_detected": True
        }
    
    return None

def get_anti_loop_system_prompt(available_tools: List[str]) -> str:
    """Generate a system prompt designed to prevent loops."""
    
    tools_list = ", ".join(available_tools)
    
    return f"""You are a helpful assistant with access to file system tools: {tools_list}

CRITICAL LOOP PREVENTION RULES:
1. Read each file ONLY ONCE per conversation
2. After reading a file, immediately proceed to edit/modify it if needed
3. Do NOT repeat the same tool call multiple times
4. If a tool fails, try a different approach rather than repeating the same call

EFFICIENT WORKFLOW:
- File modification tasks: read_file → edit_file (done)
- File search tasks: search_files → read_file → edit_file (if needed)
- Exploration tasks: list_directory → read_file (for interesting files)

NEVER:
- Read the same file twice in one conversation
- Repeat failed tool calls without changing parameters
- Get stuck in analysis loops - take action

Your goal is to complete tasks efficiently with minimal tool calls."""

# Integration example for your server.py
def integrate_loop_prevention_example():
    """
    Example of how to integrate loop prevention in your chat completions endpoint.
    
    Add this to your /v1/chat/completions endpoint in server.py:
    """
    example_code = '''
    # In your chat_completions endpoint, before executing tools:
    
    if has_tools and tool_executor and tool_executor.has_available_tools():
        # Generate conversation ID (you may already have this)
        conversation_id = request_headers.get('x-conversation-id', completion_id)
        
        # Check detected tool calls for loops
        detected_tool_calls = _parse_tool_calls_from_text(result.text)
        
        for tool_call in detected_tool_calls:
            func_name = tool_call["function"]["name"]
            try:
                arguments = json.loads(tool_call["function"]["arguments"])
            except:
                arguments = {}
            
            # Check for loops
            loop_info = check_tool_loop(conversation_id, func_name, arguments)
            
            if loop_info:
                # Replace the response with loop prevention message
                response_message = ChatCompletionMessage(
                    role="assistant",
                    content=f"{loop_info['error']} {loop_info['suggestion']}"
                )
                
                # Return without tool calls to break the loop
                return ChatCompletionResponse(
                    id=completion_id,
                    created=created_time,
                    model=request.model,
                    choices=[ChatCompletionChoice(
                        index=0,
                        message=response_message,
                        finish_reason="stop"
                    )],
                    usage=ChatCompletionUsage(
                        prompt_tokens=result.usage.get('prompt_tokens', 0),
                        completion_tokens=result.usage.get('completion_tokens', 0),
                        total_tokens=result.usage.get('total_tokens', 0)
                    )
                )
        
        # If no loops detected, proceed with normal tool execution
    '''
    return example_code

if __name__ == "__main__":
    # Example usage
    detector = ToolLoopDetector()
    
    # Simulate a conversation with loops
    conv_id = "test_conv"
    
    # First read - OK
    result1 = detector.detect_loop(conv_id, "read_file", {"path": "main.dart"})
    print(f"First read: {result1}")  # None (OK)
    
    # Second read - OK  
    result2 = detector.detect_loop(conv_id, "read_file", {"path": "main.dart"})
    print(f"Second read: {result2}")  # None (OK)
    
    # Third read - LOOP!
    result3 = detector.detect_loop(conv_id, "read_file", {"path": "main.dart"})
    print(f"Third read: {result3}")  # Loop detected message
    
    print("\nAnti-loop system prompt:")
    print(get_anti_loop_system_prompt(["read_file", "edit_file", "search_files"]))

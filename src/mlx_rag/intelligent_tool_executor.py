"""
Intelligent Tool Auto-Execution System for MLX-RAG

This module implements smart tool execution that analyzes user queries and 
automatically executes relevant tools before generating responses, making
tool usage more reliable for local models that don't have native function calling.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from mlx_rag.tool_executor import get_tool_executor

logger = logging.getLogger(__name__)


class IntelligentToolExecutor:
    """Intelligent system that auto-executes tools based on user query analysis."""
    
    def __init__(self, tool_executor=None):
        self.tool_executor = tool_executor
        
        # Query patterns that suggest specific tool usage
        self.tool_patterns = {
            'list_directory': [
                r'\b(?:list|show|see|explore|browse|what.*(?:files|directories|folders))\b.*\b(?:directory|folder|files|structure)\b',
                r'\b(?:contents|files|directories)\b.*\b(?:in|of|inside)\b.*\b(?:directory|folder)\b',
                r'\bls\b|\bdir\b|\btree\b.*structure',
                r'\b(?:what.*in|contents.*of|files.*in)\b.*\b(?:directory|folder|path)\b',
                r'\b(?:explore|browse|navigate)\b.*\b(?:project|codebase|repository)\b'
            ],
            
            'read_file': [
                r'\b(?:read|show|display|open|view|examine|check|see)\b.*\b(?:file|content|code)\b',
                r'\b(?:what.*in|contents.*of|show.*me)\b.*\b(?:file|\.py|\.js|\.md|\.txt|\.json|\.yaml|\.yml)\b',
                r'\b(?:cat|type|head|tail)\b.*(?:file|\.py|\.js|\.md|\.txt)',
                r'\b(?:code|implementation|function|class)\b.*\bin\b.*(?:file|\.py|\.js)',
                r'\b(?:how.*(?:works|implemented)|what.*does)\b.*(?:file|function|class)'
            ],
            
            'search_files': [
                r'\b(?:find|search|locate|grep|look.*for)\b.*\b(?:in|across|through)\b.*\b(?:files|code|codebase)\b',
                r'\b(?:where.*(?:defined|located|used)|find.*(?:function|class|variable|method))\b',
                r'\b(?:search|grep|find)\b.*\b(?:pattern|regex|text|string)\b',
                r'\b(?:all.*(?:occurrences|instances|uses)|everywhere.*(?:used|mentioned))\b',
                r'\b(?:find.*all|search.*for|locate.*all)\b.*\b(?:references|usages|calls)\b'
            ],
            
            'write_file': [
                r'\b(?:create|write|make|add|generate)\b.*\b(?:new|file|script|module|component)\b',
                r'\b(?:add.*file|create.*file|new.*file|write.*to)\b',
                r'\b(?:scaffold|generate|bootstrap)\b.*\b(?:project|module|component)\b',
                r'\b(?:make|create).*(?:\.py|\.js|\.md|\.txt|\.json|\.yaml|\.yml)\b'
            ],
            
            'edit_file': [
                r'\b(?:edit|modify|change|update|fix|alter|patch)\b.*\b(?:file|code|function|class|method)\b',
                r'\b(?:replace|substitute|swap|change)\b.*\bin\b.*(?:file|code)',
                r'\b(?:fix|repair|correct|update)\b.*\b(?:bug|error|issue|problem)\b',
                r'\b(?:refactor|improve|optimize|clean.*up)\b.*(?:code|function|class)',
                r'\b(?:add.*to|insert.*in|append.*to)\b.*(?:file|function|class)'
            ]
        }
        
        # File patterns for common requests
        self.file_patterns = {
            'main': [r'\bmain\.py\b', r'\bapp\.py\b', r'\bindex\.js\b', r'\bmain\.js\b', r'\b__init__\.py\b'],
            'config': [r'\bconfig\b', r'\bsettings\b', r'\b\.env\b', r'\b\.json\b', r'\b\.yaml\b', r'\b\.yml\b'],
            'readme': [r'\breadme\b', r'\b\.md\b'],
            'tests': [r'\btest\b', r'\bspec\b'],
        }
    
    async def analyze_and_execute_tools(
        self, 
        user_query: str, 
        conversation_history: List[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Analyze user query and automatically execute relevant tools.
        
        Args:
            user_query: The user's request
            conversation_history: Previous conversation context
            
        Returns:
            Tuple of (tool_results, context_summary)
        """
        if not self.tool_executor or not self.tool_executor.has_available_tools():
            return [], ""
        
        query_lower = user_query.lower()
        tool_results = []
        context_parts = []
        
        # Detect what tools to execute based on query patterns
        tools_to_execute = self._detect_required_tools(user_query)
        
        logger.info(f"Detected tools to execute for query '{user_query[:50]}...': {tools_to_execute}")
        
        for tool_name, suggested_args in tools_to_execute:
            try:
                logger.info(f"Auto-executing tool: {tool_name} with args: {suggested_args}")
                
                # Execute the tool
                result = await self.tool_executor.execute_tool_call(
                    tool_call_id=f"auto_{tool_name}_{hash(user_query) % 10000}",
                    function_name=tool_name,
                    arguments=suggested_args
                )
                
                if result.success:
                    tool_results.append({
                        'tool': tool_name,
                        'arguments': suggested_args,
                        'result': result.result,
                        'execution_time_ms': result.execution_time_ms
                    })
                    
                    # Create context summary for this tool result
                    context_summary = self._summarize_tool_result(tool_name, result.result)
                    if context_summary:
                        context_parts.append(f"**{tool_name.replace('_', ' ').title()} Result:**\\n{context_summary}")
                else:
                    logger.warning(f"Tool execution failed: {tool_name} - {result.error}")
                    
            except Exception as e:
                logger.error(f"Error auto-executing tool {tool_name}: {e}")
        
        # Combine context parts into a summary
        context_summary = "\\n\\n".join(context_parts) if context_parts else ""
        
        logger.info(f"Auto-executed {len(tool_results)} tools, generated {len(context_summary)} chars of context")
        
        return tool_results, context_summary
    
    def _detect_required_tools(self, user_query: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Detect which tools should be executed based on the user query.
        
        Returns:
            List of (tool_name, suggested_arguments) tuples
        """
        query_lower = user_query.lower()
        tools_to_execute = []
        
        # Check for explicit file references
        explicit_files = self._extract_file_references(user_query)
        explicit_dirs = self._extract_directory_references(user_query)
        
        # Rule-based tool detection
        for tool_name, patterns in self.tool_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    args = self._suggest_tool_arguments(tool_name, user_query, explicit_files, explicit_dirs)
                    if args is not None:
                        tools_to_execute.append((tool_name, args))
                    break  # Only match once per tool
        
        # Special logic for common scenarios
        tools_to_execute.extend(self._detect_special_scenarios(user_query, explicit_files, explicit_dirs))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tools = []
        for tool, args in tools_to_execute:
            key = (tool, str(sorted(args.items())))
            if key not in seen:
                seen.add(key)
                unique_tools.append((tool, args))
        
        return unique_tools
    
    def _extract_file_references(self, query: str) -> List[str]:
        """Extract explicit file references from the query."""
        # Look for file patterns with extensions
        file_patterns = [
            r'\b([\w\-\/\.]+\.[a-zA-Z0-9]+)\b',  # file.ext
            r'\b([\w\-\/]+\/[\w\-\.]+)\b',     # path/file
            r'"([^"]+\.[a-zA-Z0-9]+)"',                # "file.ext"
            r"'([^']+\.[a-zA-Z0-9]+)'",                # 'file.ext'
            r'`([^`]+\.[a-zA-Z0-9]+)`'                 # `file.ext`
        ]
        
        files = []
        for pattern in file_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            files.extend(matches)
        
        # Clean up paths and remove duplicates
        cleaned_files = []
        for file in files:
            if file and len(file) > 2:  # Avoid very short matches
                cleaned_files.append(file.strip())
        
        return list(set(cleaned_files))
    
    def _extract_directory_references(self, query: str) -> List[str]:
        """Extract directory references from the query."""
        dir_patterns = [
            r'\b(src|lib|app|components|utils|helpers|tests|docs|config)\b',
            r'\b([\w\-]+/?)\s+(?:directory|folder|dir)\b',
            r'\bin\s+(?:the\s+)?([\w\-\/]+)\s+(?:directory|folder)\b',
            r'"([^"]+/)"',  # "path/"
            r"'([^']+/)'",  # 'path/'
        ]
        
        directories = []
        for pattern in dir_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            directories.extend(matches)
        
        return list(set(dir.strip().rstrip('/') for dir in directories if dir.strip()))
    
    def _suggest_tool_arguments(
        self, 
        tool_name: str, 
        query: str, 
        explicit_files: List[str], 
        explicit_dirs: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Suggest arguments for a tool based on the query context."""
        
        if tool_name == 'list_directory':
            # Default to root, unless specific directory mentioned
            path = "."
            if explicit_dirs:
                path = explicit_dirs[0]
            
            # Check if recursive exploration is suggested
            recursive = any(word in query.lower() for word in ['all', 'recursive', 'everywhere', 'entire'])
            
            # Check for file pattern hints
            pattern = "*"
            if '.py' in query.lower():
                pattern = "*.py"
            elif '.js' in query.lower():
                pattern = "*.js"
            elif '.md' in query.lower():
                pattern = "*.md"
            elif '.json' in query.lower():
                pattern = "*.json"
            
            args = {"path": path}
            if pattern != "*":
                args["pattern"] = pattern
            if recursive:
                args["include_hidden"] = False
                
            return args
        
        elif tool_name == 'read_file':
            if explicit_files:
                return {"path": explicit_files[0]}
            
            # Try to infer file based on context
            query_lower = query.lower()
            if 'main' in query_lower:
                return {"path": "main.py"}
            elif 'config' in query_lower:
                return {"path": "config.py"}
            elif 'readme' in query_lower:
                return {"path": "README.md"}
                
            return None  # Can't determine file to read
        
        elif tool_name == 'search_files':
            # Extract search terms
            search_patterns = [
                r'search\s+for\s+"([^"]+)"',
                r"search\s+for\s+'([^']+)'",
                r'find\s+"([^"]+)"',
                r"find\s+'([^']+)'",
                r'\bgrep\s+"([^"]+)"',
                r"\bgrep\s+'([^']+)'",
                r'\b(?:function|class|method|variable)\s+([\w_]+)\b',
                r'\b([\w_]+)\s+(?:function|class|method|variable)\b'
            ]
            
            search_query = None
            for pattern in search_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    search_query = match.group(1)
                    break
            
            if not search_query:
                # Fall back to extracting quoted strings or key terms
                quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', query)
                if quoted:
                    search_query = next(q for q in quoted[0] if q)
                else:
                    # Extract potential search terms (avoid common words)
                    common_words = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b', query)
                    words = [w for w in words if w.lower() not in common_words]
                    if words:
                        search_query = words[0]  # Use first significant word
            
            if search_query:
                args = {"query": search_query}
                if explicit_dirs:
                    args["path"] = explicit_dirs[0]
                return args
                
            return None
        
        elif tool_name == 'write_file':
            if explicit_files:
                return {"path": explicit_files[0], "content": "# TODO: Add content here"}
            return None
        
        elif tool_name == 'edit_file':
            # For edit operations, we should first read the file to provide context
            # instead of trying to auto-execute edits with placeholder content
            return None
        
        return {}
    
    def _detect_special_scenarios(
        self, 
        query: str, 
        explicit_files: List[str], 
        explicit_dirs: List[str]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Detect special scenarios that require multiple tools."""
        tools = []
        query_lower = query.lower()
        
        # Scenario: "What's in this project?" or "Explore the codebase"
        if any(phrase in query_lower for phrase in [
            "what's in this project", 'explore the codebase', 'understand the code',
            'project structure', 'codebase overview', 'what does this do'
        ]):
            tools.append(('list_directory', {"path": ".", "include_hidden": False}))
            # Also try to read main files
            for main_file in ['main.py', 'app.py', '__init__.py', 'index.js', 'README.md']:
                tools.append(('read_file', {"path": main_file}))
        
        # Scenario: Looking for specific functionality
        elif any(phrase in query_lower for phrase in [
            'how does', 'what does', 'explain how', 'show me how'
        ]) and not explicit_files:
            # First search for relevant files
            key_terms = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b', query)
            if key_terms:
                tools.append(('search_files', {"query": key_terms[0]}))
        
        # Scenario: File modification requests with explicit files
        elif any(phrase in query_lower for phrase in [
            'modify', 'edit', 'change', 'update', 'fix'
        ]) and explicit_files:
            # For edit requests, first read the file to provide context
            tools.append(('read_file', {"path": explicit_files[0]}))
        
        # Scenario: File modification requests with no specific file
        elif any(phrase in query_lower for phrase in [
            'modify', 'edit', 'change', 'update', 'fix'
        ]) and not explicit_files:
            # First list directory to see what's available
            tools.append(('list_directory', {"path": ".", "include_hidden": False}))
        
        return tools
    
    def _summarize_tool_result(self, tool_name: str, result: Any) -> str:
        """Create a concise summary of tool execution results."""
        
        if tool_name == 'list_directory' and isinstance(result, dict):
            items = result.get('items', [])
            total = result.get('total', 0)
            path = result.get('path', '.')
            
            if total == 0:
                return f"Directory '{path}' is empty."
            
            # Categorize items
            dirs = [item for item in items if item.get('type') == 'directory']
            files = [item for item in items if item.get('type') == 'file']
            
            summary = f"Directory '{path}' contains {total} items:"
            if dirs:
                summary += f"\\n- {len(dirs)} directories: {', '.join(d['name'] for d in dirs[:5])}"
                if len(dirs) > 5:
                    summary += f" (and {len(dirs) - 5} more)"
            
            if files:
                summary += f"\\n- {len(files)} files: {', '.join(f['name'] for f in files[:5])}"
                if len(files) > 5:
                    summary += f" (and {len(files) - 5} more)"
            
            return summary
        
        elif tool_name == 'read_file' and isinstance(result, dict):
            path = result.get('path', 'unknown')
            total_lines = result.get('total_lines', 0)
            lines_read = result.get('lines_read', 0)
            
            summary = f"Read file '{path}' ({lines_read}/{total_lines} lines)"
            
            # Add brief content preview
            content = result.get('content', '')
            if content:
                lines = content.split('\\n')[:3]  # First 3 lines
                preview = '\\n'.join(lines)
                if len(preview) > 200:
                    preview = preview[:200] + "..."
                summary += f"\\nContent preview:\\n```\\n{preview}\\n```"
            
            return summary
        
        elif tool_name == 'search_files' and isinstance(result, dict):
            query = result.get('query', 'unknown')
            total_found = result.get('total_found', 0)
            search_results = result.get('results', [])
            
            if total_found == 0:
                return f"No results found for search query '{query}'"
            
            summary = f"Found {total_found} matches for '{query}'"
            if search_results:
                # Show first few results
                files_found = list(set(r['file'] for r in search_results[:10]))
                summary += f"\\nFiles with matches: {', '.join(files_found)}"
                
                # Show a couple of actual matches
                for i, match in enumerate(search_results[:3]):
                    summary += f"\\n- {match['file']}:{match['line']}: {match['content'][:100]}"
                    if len(match['content']) > 100:
                        summary += "..."
            
            return summary
        
        elif tool_name in ['write_file', 'edit_file'] and isinstance(result, dict):
            path = result.get('path', 'unknown')
            operation = 'Created' if tool_name == 'write_file' else 'Modified'
            return f"{operation} file '{path}' successfully"
        
        # Fallback for unknown result formats
        return f"Tool '{tool_name}' executed successfully"
    
    def create_enhanced_system_prompt(
        self, 
        tools: List[Dict[str, Any]], 
        user_query: str,
        tool_results: List[Dict[str, Any]] = None
    ) -> str:
        """
        Create an enhanced system prompt that includes tool results and better instructions.
        """
        prompt_parts = []
        
        # Base instruction
        prompt_parts.append(
            "You are an intelligent AI assistant with access to file system tools. "
            "You can explore, read, search, and modify files within the codebase. "
            "When appropriate, use the available tools to provide accurate, helpful responses."
        )
        
        # Add tool results context if available
        if tool_results:
            prompt_parts.append("\\n## Available Context (from tool execution)")
            for tool_result in tool_results:
                tool_name = tool_result.get('tool', 'unknown')
                result_summary = self._summarize_tool_result(tool_name, tool_result.get('result', {}))
                prompt_parts.append(f"\\n### {tool_name.replace('_', ' ').title()}")
                prompt_parts.append(result_summary)
        
        # Enhanced tool usage instructions for local models
        if tools:
            prompt_parts.append("\\n## Available Tools")
            prompt_parts.append(
                "You have access to the following tools. When you need to use a tool, "
                "respond with EXACTLY this JSON format:\\n"
                "```json\\n"
                '{"function": "tool_name", "arguments": {"param": "value"}}\\n'
                "```\\n"
                "Always explain what you're doing and why you're using each tool."
            )
            
            # List available tools concisely
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool.get("function", {})
                    name = func.get("name", "unknown")
                    desc = func.get("description", "No description")
                    prompt_parts.append(f"\\n**{name}**: {desc}")
        
        # Add task-specific guidance
        query_lower = user_query.lower()
        if any(word in query_lower for word in ['explore', 'understand', 'overview', 'structure']):
            prompt_parts.append(
                "\\n## Exploration Strategy\\n"
                "1. Start with directory listing to understand structure\\n"
                "2. Read key files like README, main entry points\\n"
                "3. Search for specific functionality if needed\\n"
                "4. Provide a comprehensive overview of findings"
            )
        elif any(word in query_lower for word in ['modify', 'edit', 'change', 'fix', 'update']):
            prompt_parts.append(
                "\\n## Modification Strategy\\n"
                "1. First read the target file to understand current state\\n"
                "2. Make precise, targeted changes\\n"
                "3. Explain what you changed and why\\n"
                "4. Consider impact on related code"
            )
        
        return "\\n".join(prompt_parts)


def get_intelligent_tool_executor(rag_collection_path: Optional[str] = None) -> IntelligentToolExecutor:
    """Get an intelligent tool executor for the given RAG collection."""
    if rag_collection_path:
        tool_executor = get_tool_executor(rag_collection_path)
        return IntelligentToolExecutor(tool_executor)
    return IntelligentToolExecutor()

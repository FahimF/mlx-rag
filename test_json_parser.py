#!/usr/bin/env python3

import json
import re

def _parse_arguments_robust(arguments_str: str) -> dict:
    """Parse JSON arguments with robust handling of unescaped quotes and control characters."""
    
    # Strategy 1: Try direct JSON parsing first
    try:
        return json.loads(arguments_str)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Fix literal newlines and control characters
    try:
        # Replace literal newlines with escaped newlines
        fixed_str = arguments_str.replace('\n', '\\n')
        fixed_str = fixed_str.replace('\r', '\\r')
        fixed_str = fixed_str.replace('\t', '\\t')
        
        return json.loads(fixed_str)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Fix literal newlines AND unescaped single quotes
    try:
        # Start with the string from Strategy 2
        fixed_str = arguments_str.replace('\n', '\\n')
        fixed_str = fixed_str.replace('\r', '\\r')
        fixed_str = fixed_str.replace('\t', '\\t')
        
        # Find string values (content between double quotes) and escape single quotes within them
        def escape_quotes_in_strings(match):
            content = match.group(1)
            # Escape single quotes that aren't already escaped
            escaped = re.sub(r"(?<!\\\\)'", r"\\'", content)
            return f'"{escaped}"'
        
        # Apply escaping to content within double quotes
        # This regex handles escaped quotes within the string content
        fixed_str = re.sub(r'"([^"\\\\]*(?:\\\\.[^"\\\\]*)*)"', escape_quotes_in_strings, fixed_str)
        
        return json.loads(fixed_str)
    except (json.JSONDecodeError, re.error):
        pass
    
    # Strategy 4: More aggressive fixing - handle various quote issues
    try:
        # Start with control character fixes
        fixed = arguments_str.replace('\n', '\\n')
        fixed = fixed.replace('\r', '\\r')
        fixed = fixed.replace('\t', '\\t')
        
        # First, temporarily replace already-escaped single quotes to protect them
        temp_placeholder = "__TEMP_ESCAPED_QUOTE__"
        fixed = fixed.replace("\\'", temp_placeholder)
        
        # Now replace unescaped single quotes with escaped ones
        fixed = fixed.replace("'", "\\'")            
        
        # Restore the originally escaped quotes
        fixed = fixed.replace(temp_placeholder, "\\'")            
        
        return json.loads(fixed)
    except (json.JSONDecodeError, re.error):
        pass
    
    # If all strategies fail, raise a descriptive error
    error_details = []
    if '\n' in arguments_str:
        error_details.append('contains literal newlines (use \\n instead)')
    if "'" in arguments_str:
        error_details.append('contains unescaped single quotes')
    
    error_msg = f"Could not parse JSON arguments with any strategy: {arguments_str[:100]}{'...' if len(arguments_str) > 100 else ''}"
    if error_details:
        error_msg += f" Issues detected: {', '.join(error_details)}"
        
    raise json.JSONDecodeError(error_msg, arguments_str, 0)

def test_parser():
    # Test with the exact problematic JSON from the error
    test_json = '{"path": "/Users/fahim/Code/Flutter/Booker/lib/main.dart", "start_line": 1, "end_line": 100, "new_content": "import \'package:flutter/material.dart\';\n\nvoid main() {\n  runApp(MyApp());\n}\n\nclass MyApp extends StatelessWidget {\n  @override\n  Widget build(BuildContext context) {\n    return MaterialApp(\n      home: Scaffold(\n        appBar: AppBar(\n          title: Text(\'Booker\'),\n        ),\n        body: Center(\n          child: Column(\n            mainAxisAlignment: MainAxisAlignment.center,\n            children: <Widget>[\n              Text(\'Picked Books Folder\'),\n              SizedBox(height: 20),\n              Text(\'Calibre Library Path\'),\n              SizedBox(height: 20),\n              ElevatedButton(\n                onPressed: () {},\n                child: Text(\'Pick Books Folder\'),\n              ),\n              SizedBox(height: 20),\n              ElevatedButton(\n                onPressed: () {},\n                child: Text(\'Open Calibre Library\'),\n              ),\n            ],\n          ),\n        ),\n      ),\n    );\n  }\n}"}'

    print('Testing robust JSON parsing...')
    print('JSON length:', len(test_json))
    print('Contains literal newlines:', '\n' in test_json)
    print('Contains single quotes:', "'" in test_json)
    
    # Show the problematic part
    print('\nFirst 200 chars:')
    print(repr(test_json[:200]))

    try:
        result = _parse_arguments_robust(test_json)
        print('\n✅ SUCCESS! Parsed arguments:')
        print('  - path:', result.get('path', 'N/A'))
        print('  - start_line:', result.get('start_line', 'N/A'))
        print('  - end_line:', result.get('end_line', 'N/A'))
        print('  - new_content length:', len(result.get('new_content', '')) if result.get('new_content') else 'N/A')
        print('  - new_content starts with:', repr(result.get('new_content', '')[:50]) if result.get('new_content') else 'N/A')
        
        # Test that the parsed content is correct
        new_content = result.get('new_content', '')
        if "import 'package:flutter/material.dart';" in new_content:
            print('  ✅ Flutter import statement correctly preserved')
        if 'Text(\'Booker\')' in new_content:
            print('  ✅ Single quotes in Dart code correctly preserved')
            
    except Exception as e:
        print('\n❌ FAILED:', str(e))
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_parser()

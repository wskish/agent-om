from pydantic import Field
from ai_toolchat import BaseToolParam, ToolMessage
from loguru import logger
import asyncio
import os
import re
import fnmatch
from typing import Optional, List

class GrepToolParam(BaseToolParam):
    pattern: str = Field(..., description="The regular expression pattern to search for in file contents")
    include: Optional[str] = Field(None, description="File pattern to include in the search (e.g. \"*.js\", \"*.{ts,tsx}\")")
    path: Optional[str] = Field(None, description="The directory to search in. Defaults to the current working directory.")

async def greptool(param: GrepToolParam):
    """
    This tool searches file contents using regular expressions.
    Use this tool when you need to find files containing specific text patterns.
    
    For example:
    - Find files containing 'error': {'pattern': 'error'}
    - Find function definitions in JavaScript files: {'pattern': 'function\\s+\\w+', 'include': '*.js'}
    - Find API calls in a specific folder: {'pattern': 'api\\.call', 'path': '/path/to/src', 'include': '*.{js,ts}'}
    """
    
    # Set the search path
    search_path = param.path if param.path else os.getcwd()
    
    yield ToolMessage(f"Searching for pattern: '{param.pattern}' in {search_path}")
    if param.include:
        yield ToolMessage(f"Limiting search to files matching: {param.include}")
    
    # Prepare the regex pattern
    try:
        regex = re.compile(param.pattern)
    except re.error as e:
        err = f"Invalid regular expression: {str(e)}"
        logger.error(err)
        raise ValueError(err)
    
    # Function to process a file
    async def process_file(file_path):
        matches = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                for i, line in enumerate(f, 1):
                    if regex.search(line):
                        matches.append((file_path, i, line.strip()))
        except Exception as e:
            logger.debug(f"Error processing file {file_path}: {str(e)}")
        return matches
    
    # Function to check if a file should be included based on pattern
    def should_include_file(file_path):
        if not param.include:
            return True
        
        filename = os.path.basename(file_path)
        # Handle patterns like "*.{js,ts}"
        if '{' in param.include and '}' in param.include:
            parts = param.include.split('{')
            prefix = parts[0]
            suffix_part = parts[1].split('}')
            suffixes = suffix_part[0].split(',')
            rest = suffix_part[1] if len(suffix_part) > 1 else ''
            
            for suffix in suffixes:
                pattern = prefix + suffix + rest
                if fnmatch.fnmatch(filename, pattern):
                    return True
            return False
        else:
            return fnmatch.fnmatch(filename, param.include)
    
    # Find all files recursively
    all_files = []
    for root, _, files in os.walk(search_path):
        for file in files:
            file_path = os.path.join(root, file)
            if should_include_file(file_path):
                all_files.append(file_path)
    
    # Sort files by modification time (newest first)
    all_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Process files and collect results
    tasks = [process_file(file) for file in all_files]
    results = await asyncio.gather(*tasks)
    
    # Flatten results
    all_matches = [match for file_matches in results for match in file_matches]
    
    # Format and return results
    if all_matches:
        yield ToolMessage(f"Found {len(all_matches)} matches in {len([r for r in results if r])} files")
        
        # Format the output
        output_lines = []
        current_file = None
        
        for file_path, line_num, line_content in all_matches:
            if file_path != current_file:
                output_lines.append(f"\n{file_path}:")
                current_file = file_path
            
            output_lines.append(f"  Line {line_num}: {line_content}")
        
        yield "\n".join(output_lines)
    else:
        yield "No matches found."
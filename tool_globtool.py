from pydantic import Field
from ai_toolchat import BaseToolParam, ToolMessage
from loguru import logger
import asyncio
import os
import glob
from typing import Optional, List

class GlobToolParam(BaseToolParam):
    pattern: str = Field(..., description="The glob pattern to match files against")
    path: Optional[str] = Field(None, description="The directory to search in. Defaults to the current working directory.")

async def globtool(param: GlobToolParam):
    """
    This tool performs fast file pattern matching using glob syntax.
    Use this tool when you need to find files by name patterns.
    
    For example:
    - Find all Python files: {'pattern': '**/*.py'}
    - Find all JSON files in a specific directory: {'pattern': 'config/**/*.json', 'path': '/path/to/project'}
    - Find all text files in current directory: {'pattern': '*.txt'}
    """
    
    # Set the search path
    search_path = param.path if param.path else os.getcwd()
    
    # Construct the full glob pattern
    full_pattern = os.path.join(search_path, param.pattern)
    
    yield ToolMessage(f"Searching for files matching pattern: {param.pattern} in {search_path}")
    
    try:
        # Use recursive glob for ** patterns
        if '**' in param.pattern:
            matches = glob.glob(full_pattern, recursive=True)
        else:
            matches = glob.glob(full_pattern)
        
        # Sort matches by modification time (newest first)
        matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Format the output
        if matches:
            yield ToolMessage(f"Found {len(matches)} matching files")
            result = "\n".join(matches)
            yield result
        else:
            yield "No files found matching the pattern."
            
    except Exception as e:
        err = f"Error performing glob search: {str(e)}"
        logger.error(err)
        raise ValueError(err)
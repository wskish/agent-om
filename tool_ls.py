from pydantic import Field
from ai_toolchat import BaseToolParam, ToolMessage
from loguru import logger
import asyncio
import os
import fnmatch
from typing import Optional, List

class LSParam(BaseToolParam):
    path: str = Field(..., description="The absolute path to the directory to list (must be absolute, not relative)")
    ignore: Optional[List[str]] = Field(None, description="List of glob patterns to ignore")

async def ls(param: LSParam):
    """
    This tool lists files and directories in a given path.
    Use this tool when you need to see the contents of a directory.
    
    For example:
    - List files in the current project directory: {'path': '/path/to/project'}
    - List files excluding patterns: {'path': '/path/to/project', 'ignore': ['*.log', 'node_modules/**']}
    """
    
    # Verify that the path is absolute
    if not os.path.isabs(param.path):
        raise ValueError(f"Path must be absolute, not relative: {param.path}")
    
    yield ToolMessage(f"Listing contents of directory: {param.path}")
    
    try:
        # Get all files and directories in the specified path
        if os.path.isdir(param.path):
            entries = os.listdir(param.path)
            
            # Filter out entries based on ignore patterns
            if param.ignore:
                filtered_entries = []
                for entry in entries:
                    full_path = os.path.join(param.path, entry)
                    should_ignore = False
                    
                    for pattern in param.ignore:
                        if fnmatch.fnmatch(entry, pattern) or fnmatch.fnmatch(full_path, pattern):
                            should_ignore = True
                            break
                    
                    if not should_ignore:
                        filtered_entries.append(entry)
                
                entries = filtered_entries
            
            # Sort entries (directories first, then files, alphabetically)
            dirs = []
            files = []
            
            for entry in entries:
                full_path = os.path.join(param.path, entry)
                if os.path.isdir(full_path):
                    dirs.append(entry + "/")
                else:
                    files.append(entry)
            
            dirs.sort()
            files.sort()
            sorted_entries = dirs + files
            
            # Format the output
            if sorted_entries:
                result = []
                for i, entry in enumerate(sorted_entries):
                    full_path = os.path.join(param.path, entry.rstrip('/'))
                    
                    # Get file size for files
                    if not entry.endswith('/'):
                        try:
                            size = os.path.getsize(full_path)
                            size_str = f"{size} bytes"
                        except:
                            size_str = "unknown size"
                        
                        result.append(f"{entry} ({size_str})")
                    else:
                        result.append(entry)
                
                yield f"Total: {len(sorted_entries)} entries\n" + "\n".join(result)
            else:
                yield "Directory is empty."
        else:
            raise ValueError(f"Path is not a directory: {param.path}")
            
    except FileNotFoundError:
        raise ValueError(f"Directory not found: {param.path}")
    except PermissionError:
        raise ValueError(f"Permission denied to access directory: {param.path}")
    except Exception as e:
        err = f"Error listing directory: {str(e)}"
        logger.error(err)
        raise ValueError(err)
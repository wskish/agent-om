from pydantic import Field
from ai_toolchat import BaseToolParam, ToolMessage
from loguru import logger
import os

class ReplaceParam(BaseToolParam):
    file_path: str = Field(..., description="The absolute path to the file to write (must be absolute, not relative)")
    content: str = Field(..., description="The content to write to the file")

async def replace(param: ReplaceParam):
    """
    This tool writes content to a file, completely replacing its existing content.
    Use this tool when you need to create a new file or completely overwrite an existing file.
    
    For example:
    - Create a new file: {'file_path': '/path/to/new_file.txt', 'content': 'This is a new file.'}
    - Replace an existing file: {'file_path': '/path/to/existing.py', 'content': 'print("New content")'}
    """
    
    # Verify that the path is absolute
    if not os.path.isabs(param.file_path):
        raise ValueError(f"File path must be absolute, not relative: {param.file_path}")
    
    if os.path.exists(param.file_path):
        yield ToolMessage(f"Replacing content of existing file: {param.file_path}")
    else:
        yield ToolMessage(f"Creating new file: {param.file_path}")
        
        # Verify parent directory exists
        parent_dir = os.path.dirname(param.file_path)
        if not os.path.exists(parent_dir):
            raise ValueError(f"Parent directory does not exist: {parent_dir}")
    
    try:
        # Write content to the file
        with open(param.file_path, 'w', encoding='utf-8') as f:
            f.write(param.content)
        
        file_size = len(param.content)
        yield ToolMessage(f"Successfully wrote {file_size} characters to file: {param.file_path}")
        
    except Exception as e:
        err = f"Error writing to file: {str(e)}"
        logger.error(err)
        raise ValueError(err)
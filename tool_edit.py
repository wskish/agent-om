from pydantic import Field
from ai_toolchat import BaseToolParam, ToolMessage
from loguru import logger
import os

class EditParam(BaseToolParam):
    file_path: str = Field(..., description="The absolute path to the file to modify")
    old_string: str = Field(..., description="The text to replace")
    new_string: str = Field(..., description="The text to replace it with")

async def edit(param: EditParam):
    """
    This tool edits a file by replacing one string with another.
    Use this tool for targeted edits to specific parts of files.
    
    For example:
    - Fix a typo: {'file_path': '/path/to/file.txt', 'old_string': 'typo', 'new_string': 'fixed'}
    - Update code: {'file_path': '/path/to/file.py', 'old_string': 'def old_func():\n    return 0', 'new_string': 'def new_func():\n    return 1'}
    - Create a new file: {'file_path': '/path/to/new_file.txt', 'old_string': '', 'new_string': 'New file content'}
    
    IMPORTANT: When editing existing files, make sure your old_string is unique and includes sufficient context (at least 3-5 lines before and after).
    """
    
    # Verify that the path is absolute
    if not os.path.isabs(param.file_path):
        raise ValueError(f"File path must be absolute, not relative: {param.file_path}")
    
    # Check if we're creating a new file or editing an existing one
    creating_new_file = False
    if not os.path.exists(param.file_path) and param.old_string == "":
        creating_new_file = True
        yield ToolMessage(f"Creating new file: {param.file_path}")
    else:
        yield ToolMessage(f"Editing file: {param.file_path}")
    
    try:
        if creating_new_file:
            # Create new file
            # First verify parent directory exists
            parent_dir = os.path.dirname(param.file_path)
            if not os.path.exists(parent_dir):
                raise ValueError(f"Parent directory does not exist: {parent_dir}")
            
            # Write the new content
            with open(param.file_path, 'w', encoding='utf-8') as f:
                f.write(param.new_string)
                
            yield ToolMessage(f"Successfully created file: {param.file_path}")
        else:
            # Read the existing file
            try:
                with open(param.file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
            except Exception as e:
                raise ValueError(f"Error reading file: {str(e)}")
            
            # If old_string is empty but file exists, append new content
            if param.old_string == "":
                yield ToolMessage(f"Appending content to file: {param.file_path}")
                new_content = content + param.new_string
            else:
                # Check if old_string exists in the file
                occurrences = content.count(param.old_string)
                
                if occurrences == 0:
                    raise ValueError(f"Could not find the exact string to replace in {param.file_path}")
                elif occurrences > 1:
                    raise ValueError(f"Found multiple ({occurrences}) instances of the string to replace in {param.file_path}. Please provide more context to make the match unique.")
                
                # Replace the old string with the new string
                new_content = content.replace(param.old_string, param.new_string, 1)
            
            # Write the modified content back to the file
            with open(param.file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            yield ToolMessage(f"Successfully updated file: {param.file_path}")
        
    except Exception as e:
        err = f"Error editing file: {str(e)}"
        logger.error(err)
        raise ValueError(err)
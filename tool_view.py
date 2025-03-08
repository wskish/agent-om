from pydantic import Field
from ai_toolchat import BaseToolParam, ToolMessage
from loguru import logger
import os
import imghdr
import base64
from typing import Optional

class ViewParam(BaseToolParam):
    file_path: str = Field(..., description="The absolute path to the file to read")
    offset: Optional[int] = Field(None, description="The line number to start reading from. Only provide if the file is too large to read at once")
    limit: Optional[int] = Field(None, description="The number of lines to read. Only provide if the file is too large to read at once.")

async def view(param: ViewParam):
    """
    This tool reads a file from the local filesystem.
    Use this tool when you need to read file contents.
    
    For example:
    - Read an entire file: {'file_path': '/path/to/file.txt'}
    - Read a specific part of a large file: {'file_path': '/path/to/log.txt', 'offset': 100, 'limit': 50}
    
    Images will be displayed. For Jupyter notebooks (.ipynb files), use ReadNotebook instead.
    """
    
    # Verify that the path is absolute
    if not os.path.isabs(param.file_path):
        raise ValueError(f"File path must be absolute, not relative: {param.file_path}")
    
    yield ToolMessage(f"Reading file: {param.file_path}")
    
    try:
        # Check if the file exists
        if not os.path.exists(param.file_path):
            raise ValueError(f"File not found: {param.file_path}")
        
        # Check if it's an image file
        if imghdr.what(param.file_path):
            yield ToolMessage(f"File is an image: {param.file_path}")
            
            # Read the image as binary and encode as base64
            with open(param.file_path, 'rb') as f:
                image_data = f.read()
            
            # Determine image type
            image_type = imghdr.what(None, h=image_data)
            
            # Create an HTML img tag with the embedded image
            base64_str = base64.b64encode(image_data).decode('utf-8')
            img_html = f'<img src="data:image/{image_type};base64,{base64_str}" alt="Image" />'
            
            yield img_html
            return  # End here for images
        
        # Check if it's a notebook (should use ReadNotebook instead)
        if param.file_path.endswith('.ipynb'):
            yield ToolMessage("This is a Jupyter notebook file. Please use ReadNotebook tool instead.")
            return
        
        # Read the file as text
        with open(param.file_path, 'r', encoding='utf-8', errors='replace') as f:
            if param.offset is not None:
                # Skip to the offset if provided
                for _ in range(param.offset):
                    next(f, None)
            
            limit = param.limit or 2000  # Default to 2000 lines max
            
            lines = []
            for i, line in enumerate(f):
                if i >= limit:
                    break
                    
                # Truncate long lines
                if len(line) > 2000:
                    line = line[:2000] + " [... truncated ...]"
                    
                lines.append(line)
        
        content = ''.join(lines)
        
        # Check if we reached the limit
        if param.limit is not None or len(lines) == 2000:
            yield ToolMessage(f"Showing lines {param.offset or 0} to {(param.offset or 0) + len(lines)} (max 2000 lines)")
        
        # Return the file content
        yield content
            
    except Exception as e:
        err = f"Error reading file: {str(e)}"
        logger.error(err)
        raise ValueError(err)
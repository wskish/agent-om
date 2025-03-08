from pydantic import Field
from ai_toolchat import BaseToolParam, ToolMessage
from loguru import logger
import asyncio
from typing import Optional

class BashParam(BaseToolParam):
    command: str = Field(..., description="The command to execute")
    timeout: Optional[int] = Field(None, description="Optional timeout in milliseconds (max 600000)")

async def bash(param: BashParam):
    """
    This tool executes a given bash command in a persistent shell session.
    Use this tool when you need to run shell commands, git operations, or other command-line tasks.
    The command will run in the current working directory with access to the environment.
    
    For example:
    - To list files: {'command': 'ls -la'}
    - To check git status: {'command': 'git status'}
    - With timeout: {'command': 'sleep 10', 'timeout': 5000} (will timeout after 5 seconds)
    """
    
    yield ToolMessage(f"Executing: {param.command}")
    
    # Set timeout if provided, otherwise use default (30 minutes)
    timeout_seconds = None
    if param.timeout:
        # Convert milliseconds to seconds, with a maximum of 10 minutes
        timeout_seconds = min(param.timeout / 1000, 600)
    
    try:
        # Create the subprocess
        process = await asyncio.create_subprocess_shell(
            param.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Wait for the subprocess to finish with optional timeout
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_seconds
        )
        
        stdout_text = stdout.decode() if stdout else ""
        stderr_text = stderr.decode() if stderr else ""
        
        # Check return code
        if process.returncode != 0:
            cmdstr = param.command
            err = f"Error executing command '{cmdstr}':\n{stdout_text}\n{stderr_text}"
            logger.warning(err)
            raise ValueError(err)
        
        # If the output is too large, truncate it
        if len(stdout_text) > 30000:
            yield ToolMessage(f"Output exceeded 30000 characters. Truncating...")
            stdout_text = stdout_text[:30000] + "\n[Output truncated...]"
        
        # Return the command output
        yield stdout_text
        
    except asyncio.TimeoutError:
        raise ValueError(f"Command timed out after {timeout_seconds} seconds: {param.command}")
    except Exception as e:
        raise ValueError(f"Error executing command '{param.command}': {str(e)}")
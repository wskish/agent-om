from pydantic import Field
from ai_toolchat import BaseToolParam, ToolMessage
import asyncio

class ExecParam(BaseToolParam):
    command: str = Field(..., description="The command to execute.")

async def exec(param: ExecParam):
    """Run a command line tool on behalf of the user."""

    yield ToolMessage(f"Executing  '{param.command}'")
    
    process = await asyncio.create_subprocess_shell(
        param.command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=128*1024*4
    )
    
    # Wait for the subprocess to finish and capture the output
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        # If the process did not exit successfully, log the error
        raise ValueError(f"Error executing command: {stderr.decode()}")

    yield stdout.decode()


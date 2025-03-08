# an example tool that can be used as a template to create new tools
from pydantic import Field # for defining the parameters
import asyncio  # tools must be async to work in out framework
from ai_toolchat import BaseToolParam, ToolMessage
from loguru import logger
from typing import Optional

# Define a pydantic model based on BaseToolParam to define the parameters for the tool
# All tool functions should take a single parameter that is a subclass of BaseToolParam
# The remaining parameters can be anything you want to pass to the tool
class ExampleParam(BaseToolParam):
    input: str = Field(..., description="The input to the tool.")
    debug: Optional[str] = Field(..., description="anything for debug")
    
# define the tool function
async def example(param: ExampleParam):
    """
    [Document here the purpose of the tool function and when it should be used.}
    This is an example tool that can be used as a template to create new tools.
    Call it whenever the user requests to test the tool framework. 
    It will return the input string as well as some messages to the user
    """

    # ToolMessage is a helper class return a message to the user
    # It is helpful to provide the user with context about what the tool is doing
    yield ToolMessage(f"Example tool invoked with input of length {len(param.input)}")

    cmd = ['echo', str(param.input)]

    process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,  stderr=asyncio.subprocess.PIPE)                    

    # Wait for the subprocess to finish and capture the output
    stdout, stderr = await process.communicate()

    stdout = stdout.decode() if stdout else ""
    stderr = stderr.decode() if stderr else ""

    # check return code
    if process.returncode != 0:
        # raise a ValueError if the tool encounters an error.  
        # provide helpful content in the error message.  
        # this will be returned to the LLM model that called the tool
        cmdstr = " ".join(cmd)
        err = f"Error executing command '{cmdstr}':\n{stdout}\n{stderr}"
        logger.warning(err)
        raise ValueError(err)

    
    # yield the output of the tool.  This will be returned to the LLM model for subsequent processing
    yield stdout
    

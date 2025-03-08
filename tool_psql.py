from pydantic import Field
from ai_toolchat import BaseToolParam, ToolMessage
from loguru import logger
import os
import asyncio

class PsqlParam(BaseToolParam):
    psql_args: list[str] = Field(description="The command line args to psql.")

    
async def psql(param: PsqlParam):
    """
    This tool invokes the 'psql' command command line with the specified args on the command line. 
    Use this tool whenever the user needs to access their PostgreSQL database. 
    For example, use {'psql_args': ['-c', '\l']} to list all available databases or 
    {'psql_args': ['-d', dbname, '-c', '\dt+']} to describe the dbname tables.
    The PGHOST, PGUSER, and PGPASSWORD environment variables are already set so just supply the commmand args to send to psql. 
    Ignore system databases such as azure_sys, azure_maintenance, template0, template1, and postgres unless the user asks about them specifically.
    """
        
    os.environ["PGPASSWORD"] = os.environ["KAIC_POSTGRES_PASS"]
    os.environ["PGHOST"]     = os.environ["KAIC_POSTGRES_HOST"]
    os.environ["PGUSER"]     = os.environ["KAIC_POSTGRES_USER"]
    

    if param.psql_args[0] == "psql":  # pop the initial psql if it was specified
        param.psql_args = param.psql_args[1:]
        
    cmd =  ['psql'] + param.psql_args

    yield ToolMessage(f"Executing:  {' '.join(cmd)}")
    
    # execute the psql command and return the output
    process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

    # Wait for the subprocess to finish and capture the output
    stdout, stderr = await process.communicate()

    stdout = stdout.decode() if stdout else ""
    stderr = stderr.decode() if stderr else ""

    # Check the return code to determine if the command was successful
    if process.returncode != 0:
        # Raise a ValueError if the tool encounters an error
        raise ValueError(f"Error executing psql command: {stderr}")
    
    # check return code
    if process.returncode != 0:
        # raise a ValueError if the tool encounters an error.  
        # provide helpful content in the error message.  
        # this will be returned to the LLM model that called the tool
        err = f"Error executing command {cmd}:\n{stdout}\n{stderr}"
        logger.warning(err)
        raise ValueError(err)

    linecount = len(stdout.split("\n"))
    yield ToolMessage(f"Reading {linecount} lines of psql output.")
    
    # Yield the output of the tool, which will be returned to the LLM model for subsequent processing
    yield stdout



 

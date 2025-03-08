from openai import AsyncOpenAI, RateLimitError, BadRequestError, APIError
from openai.types import FunctionDefinition
from openai.types.chat import ChatCompletionToolParam, ChatCompletionMessageParam, ChatCompletionToolMessageParam, ChatCompletionMessageToolCall, ChatCompletionAssistantMessageParam
from typing import Callable, Optional,  Union
from types import AsyncGeneratorType
from loguru import logger 
from pydantic import BaseModel


client = AsyncOpenAI(max_retries=4)


class BaseToolParam(BaseModel):
    """
    openai requires json schema to have {"additionalProperties": false},
    which requires this model_config extra="forbid".
    So anything class that ends up as a parameter to a tool function needs to inherit from this class
    or otherwise have model_config = dict(extra="forbid") in the class definition
    """
    model_config = dict(extra="forbid")

class ToolMessage(str):
    """
    A tool function can yield a ToolMessage to send a message to the user
    """
    pass

class ThinkingMessage(str):
    """
    Used to mark thinking content from Claude models
    """
    pass

# a tool function can raise ValueError with a helpful message to the model allow the model to retry the tool call
ToolFunctionType = Callable[
    [Optional[BaseToolParam]],  # A tool function can take either no argument or one argument of type BaseToolParam
    AsyncGeneratorType[
        # Tool functions can yield the following:
        #   - ToolFunctionType objects: added to the toolspec for the next completion round
        #   - ToolMessage objects: sent as messages to the user (but not the model)
        #   - str: The main payload of the tool function, sent as a message to the model (but not the user)
        Union[str, ToolMessage, 'ToolFunctionType'],
        None,  # The generator does not expect any value to be sent to it
        None, # The generator does not return any value
    ]
]



class CompletionUsage(BaseModel):
    """Mimics OpenAI's usage structure"""
    prompt_tokens: int
    completion_tokens: int


class CompletionLog(BaseModel):    
    """
    called for every llm chat completion to log inputs, outputs and token usage
    """
    model           : str
    messages        : list[dict]
    tools           : list[dict]    
    temperature     : float 
    chat_completion : Optional[str] = None
    tool_completion : Optional[list[dict]] = None
    usage           : CompletionUsage   #  prompt_tokens, completion_tokens
    retry           : Optional[int] = None
    error           : Optional[str] = None
    
# The type of the completion logger function
# optional function call on for each llm completion to log inputs, outputs and token usage
CompletionLoggerFunctionType = Callable[[CompletionLog], None]


import inspect

def toolfunc_to_toolspec(toolfunc : ToolFunctionType) -> ChatCompletionToolParam:
    """
    Convert a list of Tool objects to a list of ChatCompletionToolParam objects
    """
    if not toolfunc.__doc__: raise ValueError("Tool function requires a descriptive docstring")
    if not len(toolfunc.__doc__) > 10: raise ValueError("Tool function docstring is too short")    
    arg_types = [param.annotation for param in inspect.signature(toolfunc).parameters.values()]
    if len(arg_types) > 1: raise ValueError("Tool function must have no more than one argument")
    arg_type = arg_types[0] if arg_types else None
    if arg_type and not issubclass(arg_type, BaseToolParam): raise ValueError("Tool function argument must be a subclass of BaseToolParam")
    toolfunc.__setattr__("__param_class__", arg_type)   # save the argument type for later use as it is relatively expensive to inspect  
    param_json_schema = arg_type.model_json_schema() if arg_type else None
    return ChatCompletionToolParam( type     = 'function',
                                    function = FunctionDefinition(  name        = toolfunc.__name__, 
                                                                    description = toolfunc.__doc__,
                                                                    parameters  = param_json_schema,
                                                                    strict      = True))

    
def check_duplicate_tools(toolspecs : list[ChatCompletionToolParam]):
    fnames = [t['function'].name for t in toolspecs]
    if len(fnames) != len(set(fnames)): 
        logger.warning(f"Duplicate tool names in toolspecs: {fnames}")


    
async def toolchat(messages : list[ChatCompletionMessageParam],    # note: openai defines these input messages as typed dicts, not pydantic models
                   tools    : list[ToolFunctionType], 
                   model    : str,
                   log_func : Optional[CompletionLoggerFunctionType] = None):
    """
    A streaming chat completion function that supports tool calls
    Emits a stream of chat completion messages to the user while internally handling tool calls.
    Note that the tool calls and tool responses are not exposed to the user.
    This means the subsequent user message context does not include the tool calls or tool responses, they
    are only visible within this loop (and in the logs if a log_func is provided). 
    # see example streaming tool call processing
    # https://github.com/Azure-Samples/azureai-assistant-tool/blob/7e4ec6fedfd165cd42273bc927329dab5aa4a22c/sdk/azure-ai-assistant/azure/ai/assistant/management/chat_assistant_client.py#L312   
    """                   
    # convert our ToolFunctionsbjects to oai ChatCompletionToolParam objects
    toolspec = [toolfunc_to_toolspec(tool) for tool in tools]
    
    # we support maximum of 3 retries on error but note that we loop through here multiple times if there are tool calls to process
    # so that after each tool call the model has an opportunity to process the tool outputs and send response messages to the user.
    retries = 0
    loops = 0
    while True:                        
        loops += 1
        if loops > 20:  raise ValueError("Too many loops")
        if retries > 5: raise ValueError("Too many retries")
        check_duplicate_tools(toolspec)  # log warning if there are duplicate tool names in the toolspec
        
        # these are the outputs we accumulate via streaming
        tool_calls = []
        chat_response_content = ""
        usage = None
        
        try:            
            stream = await client.chat.completions.create(model=model, 
                                                          messages=messages, 
                                                          tools=toolspec, 
                                                          stream=True, 
                                                          stream_options={"include_usage": True})                                                        
            async for chunk in stream:       
                delta = chunk.choices[0].delta if chunk.choices else None
                if chunk.usage:             
                    usage = CompletionUsage(prompt_tokens=chunk.usage.prompt_tokens, 
                                            completion_tokens=chunk.usage.completion_tokens)
                if delta and delta.content:
                    chat_response_content += delta.content
                    yield delta.content
                if delta and delta.tool_calls:
                    for tcchunk in delta.tool_calls:
                        while len(tool_calls) <= tcchunk.index:
                            tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        tc = tool_calls[tcchunk.index]
                        tc["id"] += tcchunk.id or ""
                        tc["function"]["name"] += tcchunk.function.name or ""
                        tc["function"]["arguments"] += tcchunk.function.arguments or ""

        except RateLimitError as e:   
            logger.warning(f"OpenAI RateLimitError: {e}")
            logger.info("Retrying...")
            retries += 1
            continue            
        except APIError as e: 
            if "invalid_request_error" in str(e):
                raise e
            logger.error(f"OpenAI APIError: {e}")
            logger.info("Retrying...")
            retries += 1
            continue
        except BadRequestError as e: # too many token
            logger.error(f"OpenAI BadRequestError: {e}") 
            raise      
            """
            logger.warning("removing oldest 20% of the messages that are not system messages to make room")
            # remove oldest 20% percentage of the messages that are not system messages to make room 
            count = sum([m for m in messages if m.role != "system"]) // 5
            count = max(1, count)  # remove at least one message to make progress
            # remove count messages that are not system messages
            for i in range(count):                                
                # find find the first non-system message so that we can remove it
                index = next(i for i, m in enumerate(messages) if m.role != "system")
                # stop before we remove the last user message as that would be pointless
                if messages[index].role == "user" and sum([m for m in messages if m.role == "user"]) == 1:
                    raise ValueError("Messages are too long")
                messages.pop(index)                
            continue
            """
        except Exception as e:
            logger.exception(e)
            raise

        # log the results of the completion request 
        if log_func: 
            log_func(CompletionLog(model=model, 
                                   messages=messages,
                                   tools=toolspec, 
                                   temperature=0, 
                                   chat_completion=chat_response_content, 
                                   tool_completion=tool_calls,
                                   usage=usage,
                                   retry=retries))
            
        assistant_chat_response_message = None
        assistant_tool_response_message = None
        
        # done processing streaming events
        if chat_response_content:
            assistant_chat_response_message = ChatCompletionAssistantMessageParam(role="assistant", content=chat_response_content)
            
        if tool_calls:                
            # convert the tool call dicts to ChatCompletionMessageToolCall objects instead of the dicts we used to extract from the stream
            tool_calls = [ChatCompletionMessageToolCall(**tc) for tc in tool_calls]            
            assistant_tool_response_message = ChatCompletionAssistantMessageParam(role="assistant", tool_calls=tool_calls)                        

        if chat_response_content:
            messages.append(assistant_chat_response_message)

        if tool_calls:
            # add the ChatCompletionMessageToolCall to the message stack manually as they were streamed
            # this is needed to provide model context for subsequent completions            
             messages.append(assistant_tool_response_message)        
        
        if not tool_calls:
            break  # return when there are no more tool calls to process

        ## process the tool calls, emiting user messages but not the results of the tool call.
        ## the results of the tool call are sent to the model as ChatCompletionToolMessageParam objects in our loop for subsequent completions

        # reset the toolspec to the original list (e.g. remove any tools that were added by previous tool calls)
        toolspec = [toolfunc_to_toolspec(tool) for tool in tools]
        
        yield '\n'   # XXX
        for tc in tool_calls:
            # find the function the model tool call is referring to
            toolfunc = next(t for t in tools if t.__name__ == tc.function.name)
            param = toolfunc.__param_class__.model_validate_json(tc.function.arguments)
            stream = toolfunc(param)
            tool_result = ""
            try:
                async for chunk in stream:
                    match chunk:        
                        case ToolMessage():                  
                            # If the tool function yields a ToolMessage, it is a message to the user, not the model
                            yield chunk                                                 
                        case str():   # send the tool result to the model
                            tool_result += chunk                                                                  
                        case ToolFunctionType():  # add tool to toolspec for next completion round
                            toolspec.append(toolfunc_to_toolspec(chunk))
                            logger.info(f"toolfunc Added tool {chunk.__name__} to toolspec")
                        case _:
                            logger.error(f"Unexpected chunk type: {type(chunk)}")
                            raise AssertionError(f"Unexpected chunk type: {type(chunk)}")
            except AssertionError as e:
                raise
            except ValueError as e:
                # this is a somewhat expected error in the models usage of the tool
                tool_result = f"Error executing tool '{toolfunc.__name__}': {str(e)}.  Please try again."                            
                logger.warning(tool_result)
            except Exception as e:            
                # this is a bad, unexpected error in the tool    
                logger.exception(e) 
                tool_result = f"Error executing tool '{toolfunc.__name__}': {e}"                
            messages.append(ChatCompletionToolMessageParam(tool_call_id=tc.id, content=tool_result, role='tool'))   # let the model know what happened                
        yield '\n'   # XXX
        # continue into while loop for another round of completions
    # end of main while loop
            
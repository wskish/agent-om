import os
from anthropic import AsyncAnthropic, NotFoundError, BadRequestError
from anthropic.types import ToolUseBlock, ToolResultBlockParam, TextBlock
from typing import Optional
from loguru import logger
import inspect
import json
from ai_toolchat import BaseToolParam, ToolMessage, ToolFunctionType, CompletionUsage, CompletionLog, CompletionLoggerFunctionType

# Initialize client with API key from environment variable
client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))




def toolfunc_to_toolspec(toolfunc: ToolFunctionType) -> dict:
    """Convert a tool function to Claude's tool specification format"""
    if not toolfunc.__doc__:
        raise ValueError("Tool function requires a descriptive docstring")
    if not len(toolfunc.__doc__) > 10:
        raise ValueError("Tool function docstring is too short")
    
    arg_types = [param.annotation for param in inspect.signature(toolfunc).parameters.values()]
    if len(arg_types) > 1:
        raise ValueError("Tool function must have no more than one argument")
    
    arg_type = arg_types[0] if arg_types else None
    if arg_type and not issubclass(arg_type, BaseToolParam):
        raise ValueError(f"Tool function argument must be a subclass of BaseToolParam (found {arg_type})")
    
    toolfunc.__setattr__("__param_class__", arg_type)
    param_json_schema = arg_type.model_json_schema() if arg_type else {}
    
    return {
        "name": toolfunc.__name__,
        "description": toolfunc.__doc__,
        "input_schema": param_json_schema
    }


def check_duplicate_tools(toolspecs: list[dict]):
    fnames = [t['name'] for t in toolspecs]
    if len(fnames) != len(set(fnames)):
        logger.warning(f"Duplicate tool names in toolspecs: {fnames}")



async def toolchat( messages: list[dict],
                    tools: list[ToolFunctionType],
                    model: str,
                    temperature: float = 0,
                    log_func: Optional[CompletionLoggerFunctionType] = None):
    """
    A streaming chat completion function that supports tool calls for Claude
    Emits a stream of chat completion messages while handling tool calls internally
    Note that the tool calls and tool responses are not exposed to the user.
    This means the subsequent user message context does not include the tool calls or tool responses, they
    are only visible within this loop (and in the logs if a log_func is provided).     
    """

    # convert our ToolFunctions to claude tool specs
    toolspec = [toolfunc_to_toolspec(tool) for tool in tools]
    retries = 0
    loops = 0

    system_message = None
    if messages[0]['role'] == 'system':
        system_message = messages.pop(0)['content']
        
    while True:
        loops += 1
        if loops > 20:   raise ValueError("Too many loops")
        if retries > 5:  raise ValueError("Too many retries")            
        check_duplicate_tools(toolspec)
                    
        content_blocks = []
        content_blocks_json = []    
        usage = CompletionUsage(prompt_tokens=0, completion_tokens=0)
        try:
            stream = await client.messages.create(  system=system_message,
                                                    model=model,
                                                    messages=messages,
                                                    tools=toolspec,
                                                    temperature=temperature,
                                                    max_tokens=8192,
                                                    stream=True)            
          
            async for event in stream:
                #print(event)   # see below for example message stream
                if event.type == "message_start":
                    message = event.message 
                    # Message(id='msg_019GZi8kD5Gq8oa5GRosRHrf', content=[], model='claude-3-5-sonnet-20241022', role='assistant', stop_reason=None, stop_sequence=None, type='message', usage=Usage(input_tokens=480, output_tokens=1)
                    #logger.info(message)
                if hasattr(event, 'message'):
                    if hasattr(event.message, 'usage'):   
                        usage.prompt_tokens += event.message.usage.input_tokens
                if hasattr(event, 'usage'):
                    if hasattr(event.usage, 'output_tokens'):
                        usage.completion_tokens += event.usage.output_tokens            
                if event.type == "content_block_start":
                    content_blocks.append(event.content_block)  
                    content_blocks_json.append("")
                if event.type == "content_block_stop":
                    if content_blocks_json[event.index]:
                        # XXX nervous about this loads failing and not having a good feedback mechanism from here
                        content_blocks[event.index].input = json.loads(content_blocks_json[event.index])
                if event.type == "content_block_delta":
                    if hasattr(event.delta, 'text'):
                        content_blocks[event.index].text += event.delta.text
                        yield event.delta.text
                    elif hasattr(event.delta, 'partial_json'):
                        content_blocks_json[event.index] += event.delta.partial_json
        except (BadRequestError, NotFoundError, TypeError) as e:
            logger.error(f"Error during completion: {e}")
            for m in messages:
                logger.error(str(m))            
            raise e
        except Exception as e:
            logger.exception(f"Error during completion: {e}")
            retries += 1
            continue

        # separate out the tool use blocks and text blocks
        tooluseblocks = [b      for b in content_blocks if isinstance(b, ToolUseBlock)]
        textblocks    = [b.text for b in content_blocks if isinstance(b, TextBlock)]      
        
        # completion complete
        # log the results of the completion if a log_func is provided
        if log_func:
            log_func(CompletionLog( model=model,
                                    messages=messages,
                                    tools=toolspec,
                                    temperature=temperature,
                                    chat_completion=" ".join(textblocks),
                                    tool_completion=[t.model_dump() for t in tooluseblocks],
                                    usage=usage,
                                    retry=retries))            
        
        if not tooluseblocks:
            break  # return when there are no more tool calls to process

        # capture the assistmant messages back onto the message list for subsequent completions
        messages.append({"role": "assistant", "content": [c for c in content_blocks]})
        
        # reset the toolspec to the original list (e.g. remove any tools that were added by previous tool calls)
        toolspec = [toolfunc_to_toolspec(tool) for tool in tools]      
        
        yield '\n'   # XXX
        toolcontents = []        
        for tb in tooluseblocks:
            is_error = False                            
            # find the function the model tool call is referring to
            toolfunc = next(t for t in tools if t.__name__ == tb.name)
            tool_result = ""          
            try:
                param = toolfunc.__param_class__.model_validate(tb.input)
                stream = toolfunc(param)                      
                async for chunk in stream:
                    match chunk:        
                        case ToolMessage():                  
                            # If the tool function yields a ToolMessage, it is a message to the user, not the model
                            yield chunk                                                 
                        case str():   # send the tool result to the model
                            tool_result += chunk                                                                  
                        case ToolFunctionType():  # add tool to toolspec for next completion round
                            toolspec.append(toolfunc_to_toolspec(chunk))
                            #logger.info(f"toolfunc Added tool {chunk.__name__} to toolspec")
                        case _:
                            logger.error(f"Unexpected chunk type: {type(chunk)}")
                            raise AssertionError(f"Unexpected chunk type: {type(chunk)}")
            except AssertionError as e:
                raise
            except ValueError as e:
                # this is a somewhat expected error in the models usage of the tool
                tool_result = f"Error executing tool '{toolfunc.__name__}': {str(e)}.  Please try again."                            
                logger.warning(tool_result)
                is_error = True
            except Exception as e:            
                # this is a bad, unexpected error in the tool    
                logger.exception(e) 
                tool_result = f"Error executing tool '{toolfunc.__name__}': {e}"        
                is_error = True
            tresult = ToolResultBlockParam( type        = "tool_result",
                                            tool_use_id = tb.id, 
                                            content     = tool_result,
                                            is_error    = is_error)   
            toolcontents.append(tresult)     
        messages.append({'role':'user', 'content':toolcontents})
        yield '\n'   # XXX
        # continue into while loop for another round of completions
    # end of main while loop        





"""
Example Message Stream:

RawMessageStartEvent(message=Message(id='msg_019GZi8kD5Gq8oa5GRosRHrf', content=[], model='claude-3-5-sonnet-20241022', role='assistant', stop_reason=None, stop_sequence=None, type='message', usage=Usage(input_tokens=480, output_tokens=1)), type='message_start')
RawContentBlockStartEvent(content_block=TextBlock(text='', type='text'), index=0, type='content_block_start')
RawContentBlockDeltaEvent(delta=TextDelta(text='I', type='text_delta'), index=0, type='content_block_delta')
RawContentBlockDeltaEvent(delta=TextDelta(text="'ll", type='text_delta'), index=0, type='content_block_delta')
RawContentBlockDeltaEvent(delta=TextDelta(text=' help', type='text_delta'), index=0, type='content_block_delta')
RawContentBlockDeltaEvent(delta=TextDelta(text=' you check', type='text_delta'), index=0, type='content_block_delta')
RawContentBlockDeltaEvent(delta=TextDelta(text=' both', type='text_delta'), index=0, type='content_block_delta')
RawContentBlockDeltaEvent(delta=TextDelta(text=' the current time and weather', type='text_delta'), index=0, type='content_block_delta')
RawContentBlockDeltaEvent(delta=TextDelta(text=' in New York City.', type='text_delta'), index=0, type='content_block_delta')
RawContentBlockStopEvent(index=0, type='content_block_stop')
RawContentBlockStartEvent(content_block=ToolUseBlock(id='toolu_017boALWwSXtv2dH5StroN2m', input={}, name='get_time', type='tool_use'), index=1, type='content_block_start')
RawContentBlockDeltaEvent(delta=InputJSONDelta(partial_json='', type='input_json_delta'), index=1, type='content_block_delta')
RawContentBlockDeltaEvent(delta=InputJSONDelta(partial_json='{"', type='input_json_delta'), index=1, type='content_block_delta')
RawContentBlockDeltaEvent(delta=InputJSONDelta(partial_json='lo', type='input_json_delta'), index=1, type='content_block_delta')
RawContentBlockDeltaEvent(delta=InputJSONDelta(partial_json='cation', type='input_json_delta'), index=1, type='content_block_delta')
RawContentBlockDeltaEvent(delta=InputJSONDelta(partial_json='": "N', type='input_json_delta'), index=1, type='content_block_delta')
RawContentBlockDeltaEvent(delta=InputJSONDelta(partial_json='ew York, ', type='input_json_delta'), index=1, type='content_block_delta')
RawContentBlockDeltaEvent(delta=InputJSONDelta(partial_json='NY"}', type='input_json_delta'), index=1, type='content_block_delta')
RawContentBlockStopEvent(index=1, type='content_block_stop')
RawContentBlockStartEvent(content_block=ToolUseBlock(id='toolu_01HbYtC8zMU1dFS1XoLEqD6s', input={}, name='get_weather', type='tool_use'), index=2, type='content_block_start')
RawContentBlockDeltaEvent(delta=InputJSONDelta(partial_json='', type='input_json_delta'), index=2, type='content_block_delta')
RawContentBlockDeltaEvent(delta=InputJSONDelta(partial_json='{"location', type='input_json_delta'), index=2, type='content_block_delta')
RawContentBlockDeltaEvent(delta=InputJSONDelta(partial_json='": "', type='input_json_delta'), index=2, type='content_block_delta')
RawContentBlockDeltaEvent(delta=InputJSONDelta(partial_json='New Yo', type='input_json_delta'), index=2, type='content_block_delta')
RawContentBlockDeltaEvent(delta=InputJSONDelta(partial_json='rk, NY"}', type='input_json_delta'), index=2, type='content_block_delta')
RawContentBlockStopEvent(index=2, type='content_block_stop')
RawMessageDeltaEvent(delta=Delta(stop_reason='tool_use', stop_sequence=None), type='message_delta', usage=MessageDeltaUsage(output_tokens=112))
RawMessageStopEvent(type='message_stop')
"""    
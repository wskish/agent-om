#!/usr/bin/env python

import asyncio

from pydantic import BaseModel, Field
from typing import Literal
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit import print_formatted_text
import ai_pricing
from ai_toolchat_claude import ToolFunctionType, ToolMessage, toolchat, CompletionLog


MsgRoleType = Literal["system", "user", "assistant"]

class ChatCompletionMessage(BaseModel):
    """
    OpenAI Chat Completion Message    
    """
    role    : MsgRoleType = Field(..., description='The OpenAI role system/user/assistant')   # https://platform.openai.com/docs/guides/text-generation
    content : str         = Field(..., description='The content of the message')

class UserMessage(ChatCompletionMessage):
    def __init__(self, content: str):
        super().__init__(role="user", content=content)

class SystemMessage(ChatCompletionMessage):
    def __init__(self, content: str):
        super().__init__(role="system", content=content)

class AssistantMessage(ChatCompletionMessage):
    def __init__(self, content: str):
        super().__init__(role="assistant", content=content)


from datetime import datetime

def main(toolfuncs : list[ToolFunctionType]):
    import sys
    from datetime import datetime
    
    messages = [SystemMessage("We are assisting the user in a variety of tasks. Use available tools as appropriate. "
                              "Output in markdown format. Use tables for tabular data. "
                              "When using tools, explain to the user what tool you are using and a lay person description of the args."
                               f"The current date is {datetime.utcnow().strftime('%Y-%m-%d')}.")]
        
    session_cost = 0
    #model = "gpt-4o-mini"
    #model = "gpt-4o-2024-08-06"
    #model = 'claude-3-5-sonnet-20241022'
    model = 'claude-3-7-sonnet-20250219'
    
    print_formatted_text(FormattedText([ ("fg:violet", model)]))

    session = PromptSession(history=FileHistory('.repl_history'))
    
    while True:
        try:
            text = session.prompt('>>> ')
            if not text:
                continue
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        else:
            try:
                user_message = UserMessage(text)    
                messages.append(user_message)
            except Exception as e:
                print(f"Error: {e}")

        usage = []
        def clog(log: CompletionLog):
            usage.append(log.usage)
            open(f"completion.log", "a").write(str(log.model_dump_json(indent=2)) + "\n")        
            
        async def run_toolchat():
            current_assistant_message = ""
            async for txt in toolchat(messages=[m.model_dump() for m in messages], tools=toolfuncs, model=model, temperature=.1, log_func=clog):    
                # Define the regex pattern to match "**{msg}**\n"
                if isinstance(txt, ToolMessage):    
                    txt = f"\033[35m→  {txt}\033[0m\n"                     
                sys.stdout.write(txt)
                sys.stdout.flush()
                current_assistant_message += txt
            messages.append(AssistantMessage(current_assistant_message))   
                     
        asyncio.run(run_toolchat())
       
        calls = len(usage)
        prompt_tokens = sum(u.prompt_tokens for u in usage)
        completion_tokens = sum(u.completion_tokens for u in usage)
        cost = ai_pricing.cost(model, prompt_tokens, completion_tokens)
        session_cost += cost
        txt = f'\n\n{calls} calls: prompt: {prompt_tokens}, completion: {completion_tokens}, cost: ${cost:.4f}, session: ${session_cost:.4f}\n'
        print_formatted_text(FormattedText([ ("fg:violet", txt)]))

    print("GoodBye!")


# Check if an event loop is already running
if __name__ == "__main__":
    from tool_exec import exec
    from tool_psql import psql
    from tool_pdf_to_text import pdf_to_text
    
    toolfuncs = [exec, psql, pdf_to_text]
    for tool in toolfuncs:
         print_formatted_text(FormattedText([ ("fg:violet", "Available tool: " + tool.__name__)]))
    main(toolfuncs)

#!/usr/bin/env python

import asyncio
from pydantic import BaseModel, Field
from typing import Literal
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit import print_formatted_text
import ai_pricing
from ai_toolchat import toolchat as openai_toolchat
from ai_toolchat_claude import toolchat as claude_toolchat

from ai_toolchat import ToolFunctionType, ToolMessage, ThinkingMessage, CompletionLog

MsgRoleType = Literal["system", "user", "assistant"]

class ChatCompletionMessage(BaseModel):
    role    : MsgRoleType = Field(..., description='The role: system/user/assistant')
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

def choose_toolchat_impl(model_name: str):
    """
    Return the correct toolchat function based on model_name.
    For example, if the user sets a model that starts with 'claude',
    use the claude_toolchat. Otherwise, assume OpenAI.
    """
    # You can customize detection logic as you like.
    # For instance, we check if "claude" is in the name:
    if "claude" in model_name.lower():
        return claude_toolchat
    else:
        return openai_toolchat

def main(toolfuncs : list[ToolFunctionType]):
    import sys
    from datetime import datetime
    
    # System message as string
    system_message = ("We are assisting the user in a variety of tasks. Use available tools as appropriate. "
                     "Output in markdown format. Use tables for tabular data. "
                     "When using tools, explain to the user what tool you are using and a lay person description of the args. "
                     f"The current date is {datetime.utcnow().strftime('%Y-%m-%d')}.")
                     
    # Initialize empty messages list (no system message)
    messages = []
        
    session_cost = 0

    # Default model
    model = "claude-3-7-sonnet-20250219"
    # Default thinking budget (only used with Claude models)
    thinking_budget = 0
    
    print_formatted_text(FormattedText([("fg:violet", model)]))

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

        # Check if user wants to change model
        if text.startswith('/model '):
            new_model = text[len('/model '):].strip()
            if new_model:
                model = new_model
                print_formatted_text(
                    FormattedText([("fg:green", f"Model changed to: {model}\n")])
                )
            else:
                print_formatted_text(
                    FormattedText([("fg:red", "No model name provided after /model\n")])
                )
            # Skip adding /model to the conversation
            continue
            
        # Check if user wants to set thinking budget
        if text.startswith('/thinking'):
            try:
                # Handle both "/thinking" and "/thinking <number>"
                if text.strip() == '/thinking':
                    new_budget = 1024  # Minimum budget
                else:
                    new_budget = int(text[len('/thinking '):].strip())
                    # Ensure minimum budget of 1024
                    if new_budget < 1024:
                        new_budget = 1024
                
                thinking_budget = new_budget
                if "claude" in model.lower():
                    print_formatted_text(
                        FormattedText([("fg:green", f"Thinking budget set to: {thinking_budget} tokens\n")])
                    )
                else:
                    print_formatted_text(
                        FormattedText([("fg:yellow", f"Note: Thinking budget only applies to Claude models. Current model: {model}\n")])
                    )
            except ValueError:
                print_formatted_text(
                    FormattedText([("fg:red", "Invalid thinking budget. Using minimum of 1024 tokens.\n")])
                )
                thinking_budget = 1024
            # Skip adding /thinking to the conversation
            continue

        try:
            user_message = UserMessage(text)
            messages.append(user_message)
        except Exception as e:
            print(f"Error: {e}")
            continue

        usage = []
        def clog(log: CompletionLog):
            usage.append(log.usage)
            with open("completion.log", "a") as f:
                f.write(str(log.model_dump_json(indent=2)) + "\n")
        
        # Pick the correct toolchat implementation based on current model
        toolchat_impl = choose_toolchat_impl(model)
            
        async def run_toolchat():
            current_assistant_message = ""
            
            # Prepare kwargs for toolchat
            toolchat_kwargs = {
                "messages": [m.model_dump() for m in messages],
                "tools": toolfuncs,
                "model": model,
                "log_func": clog,
                "system_message": system_message
            }
            
            # Add thinking_budget for Claude models
            if "claude" in model.lower() and thinking_budget > 0:
                toolchat_kwargs["thinking_budget"] = thinking_budget
                
            async for txt in toolchat_impl(**toolchat_kwargs):
                if isinstance(txt, ToolMessage):
                    txt = f"\033[35m→  {txt}\033[0m\n"  # Purple color for tool messages
                    current_assistant_message += txt  # store this for reference
                elif isinstance(txt, ThinkingMessage):
                    txt = f"\033[34m{txt}\033[0m"  # Deep blue color for thinking messages
                    # dont store this in the assistant message
                else:
                    current_assistant_message += txt  # store normal assistant output in message history
                sys.stdout.write(txt)
                sys.stdout.flush()
                

            messages.append(AssistantMessage(current_assistant_message))

        asyncio.run(run_toolchat())

        calls = len(usage)
        prompt_tokens = sum(u.prompt_tokens for u in usage)
        completion_tokens = sum(u.completion_tokens for u in usage)
        cost_val = ai_pricing.cost(model, prompt_tokens, completion_tokens)
        session_cost += cost_val

        txt = (
            f'\n\n{calls} calls: '
            f'prompt: {prompt_tokens}, completion: {completion_tokens}, '
            f'cost: ${cost_val:.4f}, session: ${session_cost:.4f}\n'
        )
        print_formatted_text(FormattedText([("fg:violet", txt)]))

    print("GoodBye!")


if __name__ == "__main__":
    from tool_exec import exec
    from tool_psql import psql
    from tool_pdf_to_text import pdf_to_text
    from tool_bash import bash
    from tool_globtool import globtool
    from tool_greptool import greptool
    from tool_ls import ls
    from tool_view import view
    from tool_edit import edit
    from tool_replace import replace
    
    toolfuncs = [
        exec, 
        psql, 
        pdf_to_text, 
        bash, 
        globtool, 
        greptool, 
        ls, 
        view, 
        edit, 
        replace
    ]
    for tool in toolfuncs:
        print_formatted_text(FormattedText([("fg:violet", "Available tool: " + tool.__name__)]))
    main(toolfuncs)

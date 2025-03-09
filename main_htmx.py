#!/usr/bin/env python

from fastapi import FastAPI, Request, Form, Header
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import json
import os
import asyncio
import uvicorn
from datetime import datetime
from typing import List, Dict, Optional, Union, AsyncGenerator, Any
import random
import uuid

# Import our existing toolchat implementations
from ai_toolchat import toolchat as openai_toolchat, ToolMessage, ThinkingMessage, CompletionLog
from ai_toolchat_claude import toolchat as claude_toolchat
import ai_pricing

# Import our tools
from tool_exec import exec
from tool_psql import psql
from tool_pdf_to_text import pdf_to_text

# Create FastAPI app
app = FastAPI()

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
os.makedirs("templates", exist_ok=True)  # Ensure directory exists
os.makedirs("static", exist_ok=True)  # Ensure static directory exists
app.mount("/static", StaticFiles(directory="static"), name="static")

# State for managing a single chat session
class ChatState:
    messages = []
    model = "claude-3-7-sonnet-20250219"
    thinking_budget = 0
    session_cost = 0.0
    usage_stats = []

chat_state = ChatState()

# System message as string
system_message = ("We are assisting the user in a variety of tasks. Use available tools as appropriate. "
                 "Output in markdown format. Use tables for tabular data. "
                 "When using tools, explain to the user what tool you are using and a lay person description of the args. "
                 f"The current date is {datetime.utcnow().strftime('%Y-%m-%d')}.")

# Initialize empty messages list (no system message)
chat_state.messages = []

# Available tools
toolfuncs = [exec, psql, pdf_to_text]

def choose_toolchat_impl(model_name: str):
    """Return the correct toolchat function based on model name."""
    if "claude" in model_name.lower():
        return claude_toolchat
    else:
        return openai_toolchat

async def stream_chat_response(user_message: str) -> AsyncGenerator[str, None]:
    """Process user message and stream response chunks."""
    # Add user message to chat history
    chat_state.messages.append({"role": "user", "content": user_message})
    
    # Track usage for this exchange
    usage = []
    
    def clog(log: CompletionLog):
        usage.append(log.usage)
    
    # Choose the appropriate toolchat implementation
    toolchat_impl = choose_toolchat_impl(chat_state.model)
    
    # Message ID for the response
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    
    # Create a copy of messages to avoid modifying the original
    messages_copy = chat_state.messages.copy()
    
    # Prepare kwargs for toolchat with system message as a separate parameter
    toolchat_kwargs = {
        "messages": messages_copy,
        "tools": toolfuncs,
        "model": chat_state.model,
        "log_func": clog,
        "system_message": system_message
    }
    
    # Add thinking_budget for Claude models
    if "claude" in chat_state.model.lower() and chat_state.thinking_budget > 0:
        toolchat_kwargs["thinking_budget"] = chat_state.thinking_budget
    
    # Create an accumulated response for chat history
    current_assistant_message = ""
    thinking_occurred = False
    
    try:
        async for txt in toolchat_impl(**toolchat_kwargs):
            if isinstance(txt, ToolMessage):
                # Format tool messages
                msg_type = "tool"
                display_txt = str(txt)
                # Tool messages get unique IDs
                curr_id = f"tool_{uuid.uuid4().hex[:8]}"
            elif isinstance(txt, ThinkingMessage):
                # Format thinking messages
                msg_type = "thinking"
                display_txt = str(txt)
                thinking_occurred = True
                curr_id = msg_id
            else:
                # Regular assistant message
                msg_type = "assistant"
                display_txt = txt
                current_assistant_message += txt  # Accumulate for history
                # If we had thinking previously, use a distinct ID for assistant messages
                curr_id = f"{msg_id}_assistant" if thinking_occurred else msg_id
            
            # Yield formatted SSE event
            data = json.dumps({
                "type": msg_type,
                "content": display_txt,
                "id": curr_id
            })
            yield f"event: message\ndata: {data}\n\n"
            
            # Small delay for browser rendering
            await asyncio.sleep(0.01)
        
        # After completion, add the assistant message to chat history
        if current_assistant_message:
            chat_state.messages.append({
                "role": "assistant", 
                "content": current_assistant_message
            })
        
        # Calculate and send usage statistics
        if usage:
            calls = len(usage)
            prompt_tokens = sum(u.prompt_tokens for u in usage)
            completion_tokens = sum(u.completion_tokens for u in usage)
            cost_val = ai_pricing.cost(chat_state.model, prompt_tokens, completion_tokens)
            chat_state.session_cost += cost_val
            
            stats = {
                "calls": calls,
                "promptTokens": prompt_tokens,
                "completionTokens": completion_tokens,
                "cost": f"${cost_val:.4f}",
                "sessionCost": f"${chat_state.session_cost:.4f}"
            }
            
            yield f"event: stats\ndata: {json.dumps(stats)}\n\n"
            
    except Exception as e:
        error_data = json.dumps({
            "type": "error",
            "content": f"Error: {str(e)}",
            "id": msg_id
        })
        yield f"event: message\ndata: {error_data}\n\n"

@app.get("/")
async def get_chat_page(request: Request):
    """Render the main chat interface."""
    # Log available tools for debugging
    tool_names = [tool.__name__ for tool in toolfuncs]
    print(f"Tools available at page load: {tool_names}")
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model": chat_state.model,
        "thinking_budget": chat_state.thinking_budget,
        "session_cost": f"${chat_state.session_cost:.4f}",
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "claude-3-5-sonnet-20241022",
            "claude-3-7-sonnet-20250219",
            "o1-2024-12-17",
            "o3-mini-2025-01-31"
        ],
        # Pre-render tools for immediate display
        "available_tools": ", ".join(tool_names)
    })

@app.get("/chat/stream")
async def stream_chat(message: str):
    """Stream chat responses as server-sent events."""
    return StreamingResponse(
        stream_chat_response(message),
        media_type="text/event-stream"
    )

@app.post("/set-model")
async def set_model(model: str = Form(...)):
    """Update the current model."""
    chat_state.model = model
    return {"success": True, "model": model}

@app.post("/set-thinking")
async def set_thinking(budget: int = Form(...)):
    """Update the thinking budget."""
    # Ensure minimum budget of 1024 tokens for Claude models
    if budget > 0 and budget < 1024 and "claude" in chat_state.model.lower():
        budget = 1024
    
    # Track if this is a significant change to reduce system message noise
    significant_change = abs(chat_state.thinking_budget - budget) > 1024
    
    # Store previous state to determine if turning on/off
    was_on = chat_state.thinking_budget > 0
    is_on = budget > 0
    state_change = was_on != is_on
    
    # Update the budget
    chat_state.thinking_budget = budget
    
    return {
        "success": True, 
        "thinking_budget": budget,
        "model": chat_state.model,
        "is_claude": "claude" in chat_state.model.lower(),
        "significant_change": significant_change or state_change
    }

@app.get("/available-tools")
async def get_available_tools():
    """Return a list of available tools."""
    tool_names = [tool.__name__ for tool in toolfuncs]
    print(f"Available tools: {tool_names}")  # Debug print
    return {
        "tools": tool_names
    }

if __name__ == "__main__":
    uvicorn.run("main_htmx:app", host="0.0.0.0", port=8000, reload=True)

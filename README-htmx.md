# HTMX AI Chat Application

A single-page web application that provides an interactive chat interface for AI models with tool-calling capabilities, built using FastAPI, HTMX, Alpine.js, and Tailwind CSS.

## Overview

This application transforms a command-line AI toolchat interface into a responsive web application while preserving all the original functionality. It connects to various AI models (OpenAI and Anthropic) and allows them to use tools during conversation, with real-time streaming of responses.

## Architecture

### Backend (FastAPI)

The application follows a simple, modern architecture:

1. **Single FastAPI application** (`main_htmx.py`) that:
   - Handles HTTP requests
   - Manages the application state
   - Streams AI responses as server-sent events (SSE)
   - Reuses existing toolchat implementation

2. **Chat Session Management**:
   - The application maintains a single chat session state on the server
   - Tracks message history, model selection, and token usage

3. **Server-Sent Events Streaming**:
   - All AI responses are streamed in real-time using SSE
   - Different message types (assistant text, tool usage, thinking) are differentiated in the stream

### Frontend (HTMX + Alpine.js + Tailwind)

1. **HTMX Integration**:
   - Manages server-sent events connection
   - Processes incoming message chunks

2. **Alpine.js State Management**:
   - Maintains UI state
   - Handles message rendering and display
   - Controls input form and settings

3. **Tailwind CSS Styling**:
   - Provides responsive, modern styling
   - Custom styling for message types and markdown content

## Components

### Message Types

The application handles several types of messages with distinct styling:

1. **User Messages**: Blue background, left-aligned
2. **Assistant Messages**: Gray background with markdown rendering
3. **Tool Messages**: Purple styling to indicate tool execution
4. **Thinking Messages**: Light blue, italicized text (for Claude models)
5. **System Messages**: Yellow background for system notifications
6. **Stats Messages**: Gray banner showing token usage and costs

### Settings Controls

1. **Model Selection**: Dropdown to choose between supported AI models
2. **Thinking Budget**: Input field to configure thinking tokens (Claude models only)
3. **Session Cost Display**: Real-time tracking of API costs

## Data Flow

1. **User Sends Message**:
   - Message is sent to `/chat/stream` endpoint
   - SSE connection is established

2. **Server Processing**:
   - FastAPI passes the message to appropriate toolchat implementation
   - AI generates response, potentially using tools

3. **Response Streaming**:
   - Different message types are streamed as SSE events
   - Client receives and renders each chunk in real-time
   - Message history is updated on both server and client

4. **Settings Updates**:
   - Model changes via `/set-model` endpoint
   - Thinking budget via `/set-thinking` endpoint

## Technologies Used

- **Backend**:
  - FastAPI - High-performance async web framework
  - Uvicorn - ASGI server
  - Jinja2 - Template rendering
  - Existing Python toolchat implementations (OpenAI and Claude)

- **Frontend**:
  - HTMX - HTML extension for AJAX, CSS transitions, WebSockets
  - Alpine.js - Lightweight JavaScript framework
  - Tailwind CSS - Utility-first CSS framework
  - Marked.js - Markdown rendering
  - Highlight.js - Code syntax highlighting


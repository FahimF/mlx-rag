# Plan for Chat Feature in MLX-GUI

This document outlines the plan for adding a new "Chat" tab to the `mlx-gui` admin panel.

### 1. High-Level Goal

The primary objective is to create a new "Chat" tab that serves as an interactive and user-friendly interface for communicating with the installed language models. This feature will be designed to support both standard conversational AI and a more powerful RAG-powered chat that can reason about the user's source code.

### 2. GUI/Frontend Plan (`src/mlx_gui/templates/admin.html`)

1.  **Add a "Chat" Tab:**
    *   A new "Chat" button will be added to the main tab navigation bar in `admin.html`.
    *   A corresponding `div` with `id="chat-tab"` will be created to house the chat interface.

2.  **Chat Tab UI Components:**
    *   **Chat History Display:**
        *   A scrollable `div` will serve as the main chat window, displaying the conversation history.
        *   User messages and model responses will be styled differently for clarity (e.g., different background colors, alignment).
        *   Code blocks within the model's responses will be automatically detected, syntax-highlighted, and equipped with a "copy" button for convenience.
    *   **Chat Input Area:**
        *   A responsive text area at the bottom of the screen for users to compose their messages.
        *   A "Send" button will be included, and the interface will also support sending messages by pressing the "Enter" key.
    *   **Chat Controls:**
        *   **Model Selection:** A dropdown menu will list all the currently loaded "text" and "multimodal" models, allowing the user to switch between them on the fly.
        *   **RAG Collection Selection:** Another dropdown will list all available RAG collections. This will include a "None" option to allow users to disable the RAG functionality and have a standard chat session.
        *   **Conversation Management:**
            *   A "New Chat" button to clear the current conversation and start fresh.
            *   A "Clear History" button to permanently delete the chat history for the current session.

### 3. Backend Plan (`src/mlx_gui/server.py`)

1.  **New API Endpoint:**
    *   `POST /v1/chat`: This will be the single, powerful endpoint that drives the chat functionality. It will be designed to handle streaming responses to create a real-time, interactive experience.

2.  **Endpoint Logic:** The `/v1/chat` endpoint will:
    *   Accept a JSON payload containing the user's message, the selected model, the selected RAG collection (if any), and the conversation history.
    *   **RAG Integration:** If a RAG collection is selected, it will first call the `RAGManager` to retrieve relevant code snippets from the user's source code based on the query.
    *   **Prompt Engineering:** It will dynamically construct a detailed prompt for the language model, including:
        *   A system message to set the context (e.g., "You are a helpful AI assistant specializing in software development.").
        *   The conversation history to maintain context.
        *   The retrieved code snippets from the RAG collection.
        *   The user's latest message.
    *   **Model Interaction:** It will then use the `ModelManager` to call the selected language model and generate a response.
    *   **Streaming:** The endpoint will use FastAPI's `StreamingResponse` to send the model's response back to the frontend token by token, creating a smooth, real-time typing effect in the chat window.

### 4. Implementation Roadmap

1.  **Frontend First:** I will begin by building the chat interface in `admin.html`. This will involve creating the HTML structure and styling the chat components with CSS to ensure a polished and intuitive user experience.
2.  **JavaScript Logic:** Next, I will implement the client-side JavaScript to handle:
    *   User input and button clicks.
    *   API calls to the backend.
    *   Receiving and rendering the streamed response from the server.
    *   Dynamically populating the model and RAG collection dropdowns.
3.  **Backend Endpoint:** I will then implement the `/v1/chat` endpoint in `server.py`, integrating the `RAGManager` and `ModelManager` to orchestrate the full chat logic.
4.  **Streaming Implementation:** I will ensure the backend properly streams the response, and the frontend can correctly handle and display the incoming tokens in real-time.

This plan will deliver a robust and feature-rich chat experience that seamlessly integrates with the existing model and RAG management capabilities of `mlx-gui`.

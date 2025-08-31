/**
 * MLX-RAG Admin Dashboard - Chat Functions
 * Chat session management and messaging functionality
 */

// Enhanced Chat functions with session management
let chatSessions = [];
let currentChatSession = null;
let currentChatMessages = [];

// Load chat sessions from server
async function loadChatSessions() {
    try {
        const response = await apiCall('/v1/chat/sessions');
        chatSessions = response.sessions || [];
        renderChatSessions();
        
        // Initialize with first session if available and no current session
        if (chatSessions.length > 0 && !currentChatSession) {
            await loadChatSession(chatSessions[0].session_id);
        }
    } catch (error) {
        console.error('Failed to load chat sessions:', error);
        showToast('Failed to load chat history', 'error');
    }
}

// Create a new chat session
async function newChatSession() {
    try {
        const model = document.getElementById('chat-model-select').value;
        const ragCollection = document.getElementById('chat-rag-select').value;
        
        const response = await apiCall('/v1/chat/sessions', {
            method: 'POST',
            body: JSON.stringify({
                title: 'New Chat',
                model_name: model,
                rag_collection_name: ragCollection
            })
        });
        
        // Reload sessions and switch to the new one
        await loadChatSessions();
        await loadChatSession(response.session_id);
        
        showToast('New chat session created');
    } catch (error) {
        console.error('Failed to create new chat session:', error);
        showToast('Failed to create new chat session', 'error');
    }
}

// Load a specific chat session
async function loadChatSession(sessionId) {
    try {
        const response = await apiCall(`/v1/chat/sessions/${sessionId}`);
        currentChatSession = response;
        currentChatMessages = response.messages || [];
        
        // Update UI
        document.getElementById('current-chat-title').textContent = response.title;
        document.getElementById('edit-title-btn').classList.remove('hidden');
        document.getElementById('delete-chat-btn').classList.remove('hidden');
        document.getElementById('scroll-to-bottom-btn').classList.remove('hidden');
        
        // Set model and RAG collection if available
        if (response.model_name) {
            document.getElementById('chat-model-select').value = response.model_name;
        }
        if (response.rag_collection_name) {
            document.getElementById('chat-rag-select').value = response.rag_collection_name;
        }
        
        // Enable chat input
        const chatInput = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-chat-btn');
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.placeholder = 'Type your message...';
        
        // Hide placeholder and render messages
        document.getElementById('no-chat-placeholder').style.display = 'none';
        renderChatMessages();
        
        // Ensure we scroll to bottom after loading the chat session
        setTimeout(() => {
            const container = document.getElementById('chat-messages');
            container.scrollTop = container.scrollHeight;
        }, 150);
        
        // Update session list selection
        updateSessionSelection(sessionId);
        
    } catch (error) {
        console.error('Failed to load chat session:', error);
        showToast('Failed to load chat session', 'error');
    }
}

// Render chat sessions in sidebar
function renderChatSessions() {
    const container = document.getElementById('chat-sessions-list');
    
    if (chatSessions.length === 0) {
        container.innerHTML = `
            <div class="text-center text-gray-500 py-8">
                <i class="fas fa-comment text-3xl mb-2"></i>
                <p class="text-sm">No chat history</p>
                <button onclick="newChatSession()" class="mt-2 bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-sm">
                    Start First Chat
                </button>
            </div>
        `;
        return;
    }
    
    container.innerHTML = chatSessions.map(session => {
        const isActive = currentChatSession && currentChatSession.session_id === session.session_id;
        const timeago = formatTimeAgo(session.last_message_at || session.updated_at);
        
        return `
            <div class="chat-session-item p-3 rounded-lg cursor-pointer transition-colors mb-2 ${
                isActive ? 'bg-blue-600 border-blue-400' : 'bg-gray-700 hover:bg-gray-600'
            }" 
                 onclick="loadChatSession('${session.session_id}')">
                <div class="flex items-start justify-between">
                    <div class="flex-1 min-w-0">
                        <h4 class="text-white font-medium text-sm truncate">${escapeHtml(session.title)}</h4>
                        <div class="flex items-center space-x-2 mt-1">
                            <span class="text-xs text-gray-400">${timeago}</span>
                            ${session.message_count > 0 ? `<span class="text-xs text-gray-400">• ${session.message_count} msgs</span>` : ''}
                        </div>
                        ${session.model_name ? `<div class="text-xs text-blue-300 mt-1">${session.model_name}</div>` : ''}
                    </div>
                    <div class="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button onclick="event.stopPropagation(); editSessionTitle('${session.session_id}')" 
                                class="text-gray-400 hover:text-white p-1">
                            <i class="fas fa-edit text-xs"></i>
                        </button>
                        <button onclick="event.stopPropagation(); deleteSession('${session.session_id}')" 
                                class="text-gray-400 hover:text-red-400 p-1">
                            <i class="fas fa-trash text-xs"></i>
                        </button>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// Update session selection in sidebar
function updateSessionSelection(sessionId) {
    document.querySelectorAll('.chat-session-item').forEach(item => {
        item.classList.remove('bg-blue-600', 'border-blue-400');
        item.classList.add('bg-gray-700', 'hover:bg-gray-600');
    });
    
    const activeItem = document.querySelector(`[onclick*="${sessionId}"]`);
    if (activeItem) {
        activeItem.classList.remove('bg-gray-700', 'hover:bg-gray-600');
        activeItem.classList.add('bg-blue-600', 'border-blue-400');
    }
}

// Render messages in current session
function renderChatMessages() {
    const container = document.getElementById('chat-messages');
    
    if (currentChatMessages.length === 0) {
        container.innerHTML = `
            <div class="text-center text-gray-500 py-8">
                <i class="fas fa-comment-dots text-3xl mb-2"></i>
                <p>Start a conversation...</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = currentChatMessages.map((msg, index) => {
        let content;
        if (msg.role === 'user') {
            content = escapeHtml(msg.content);
        } else {
            // For assistant messages, check if streaming
            if (msg.isStreaming) {
                content = escapeHtml(msg.content);
            } else {
                content = parseMarkdown(msg.content);
            }
        }
        
        // Generate images display for user messages
        let imagesHtml = '';
        if (msg.role === 'user' && msg.images && msg.images.length > 0) {
            imagesHtml = `
                <div class="mt-2 flex flex-wrap gap-2">
                    ${msg.images.map(img => `
                        <div class="bg-gray-600 rounded-lg p-2 flex items-center space-x-2 text-xs">
                            <i class="fas fa-image text-blue-400"></i>
                            <span class="text-gray-300">${img.name}</span>
                            <span class="text-gray-500">(${formatFileSize(img.size)})</span>
                        </div>
                    `).join('')}
                </div>
            `;
        }
        
        const timestamp = msg.created_at ? new Date(msg.created_at).toLocaleTimeString() : '';
        
        return `
            <div class="mb-4">
                <div class="flex items-start space-x-3">
                    <div class="flex-shrink-0">
                        <div class="w-8 h-8 rounded-full flex items-center justify-center ${
                            msg.role === 'user' ? 'bg-blue-600' : 'bg-green-600'
                        }">
                            <i class="fas ${
                                msg.role === 'user' ? 'fa-user' : 'fa-robot'
                            } text-white text-sm"></i>
                        </div>
                    </div>
                    <div class="flex-1 min-w-0">
                        <div class="flex items-center justify-between mb-1">
                            <div class="flex items-center space-x-2">
                                <span class="text-sm font-medium text-white capitalize">${msg.role}</span>
                                <span class="text-xs text-gray-400">${timestamp}</span>
                                ${msg.model_name ? `<span class="text-xs text-blue-300">${msg.model_name}</span>` : ''}
                                ${msg.rag_collection_name ? `<span class="text-xs text-purple-300">RAG: ${msg.rag_collection_name}</span>` : ''}
                                ${msg.images && msg.images.length > 0 ? `<span class="text-xs text-yellow-400"><i class="fas fa-image mr-1"></i>${msg.images.length} image${msg.images.length > 1 ? 's' : ''}</span>` : ''}
                            </div>
                            ${msg.role === 'user' ? `
                                <button onclick="resubmitMessage(${index})" 
                                    class="text-gray-400 hover:text-blue-400 p-1 rounded transition-colors" 
                                    title="Re-submit this message">
                                    <i class="fas fa-redo text-xs"></i>
                                </button>
                            ` : ''}
                        </div>
                        <div class="p-3 rounded-lg ${
                            msg.role === 'user' ? 'bg-gray-700' : 'bg-gray-800'
                        }">
                            <div class="text-white ${
                                msg.role === 'user' || msg.isStreaming ? 'whitespace-pre-wrap' : ''
                            }">${content}</div>
                            ${imagesHtml}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }).join('');
    
    // Auto-scroll to bottom
    setTimeout(() => {
        container.scrollTop = container.scrollHeight;
    }, 100);
}

// Send message in current session
async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    
    // Check if we have message or images
    if (!message && selectedImages.length === 0) {
        showToast('Please enter a message or attach images', 'error');
        return;
    }
    
    const model = document.getElementById('chat-model-select').value;
    const ragCollection = document.getElementById('chat-rag-select').value;
    
    if (!model) {
        showToast('Please select a model', 'error');
        return;
    }
    
    // Check if images are selected but model doesn't support vision
    const selectedModel = availableChatModels.find(m => m.name === model);
    const supportsVision = selectedModel && selectedModel.type === 'multimodal';
    
    if (selectedImages.length > 0 && !supportsVision) {
        showToast('Selected model does not support image input. Please select a multimodal model.', 'error');
        return;
    }
    
    if (!currentChatSession) {
        // Create new session if none exists
        await newChatSession();
        if (!currentChatSession) return;
    }
    
    // Store images for local display (we'll need this for the user message)
    const imagesToSend = [...selectedImages];
    
    // Add user message locally (with image info if present)
    const userMessage = {
        role: 'user',
        content: message,
        created_at: new Date().toISOString(),
        isStreaming: false,
        images: imagesToSend.length > 0 ? imagesToSend.map(img => ({ name: img.name, size: img.size })) : undefined
    };
    
    currentChatMessages.push(userMessage);
    renderChatMessages();
    input.value = '';
    
    // Clear selected images from the UI
    clearSelectedImages();
    
    // Save user message to server
    try {
        await apiCall(`/v1/chat/sessions/${currentChatSession.session_id}/messages`, {
            method: 'POST',
            body: JSON.stringify({
                role: 'user',
                content: message
            })
        });
    } catch (error) {
        console.error('Failed to save user message:', error);
    }
    
    // Add placeholder for assistant response
    const assistantMessage = {
        role: 'assistant',
        content: '',
        created_at: new Date().toISOString(),
        model_name: model,
        rag_collection_name: ragCollection,
        isStreaming: true
    };
    
    currentChatMessages.push(assistantMessage);
    renderChatMessages();
    
    try {
        // Convert images to base64 data URLs for OpenAI endpoint
        const processedImages = await Promise.all(
            imagesToSend.map(imageData => {
                return new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onload = () => resolve(reader.result); // This will be a data URL
                    reader.onerror = reject;
                    reader.readAsDataURL(imageData.file);
                });
            })
        );
        
        // Build messages array for OpenAI format
        const messages = [];
        
        // Add chat history
        currentChatMessages.slice(0, -1).forEach(msg => {
            if (msg.role === 'user' || msg.role === 'assistant') {
                messages.push({
                    role: msg.role,
                    content: msg.content
                });
            }
        });
        
        // Add current user message with images if present
        let userMessageContent;
        if (processedImages.length > 0) {
            // Multimodal content format
            userMessageContent = [
                { type: 'text', text: message }
            ];
            processedImages.forEach(imageDataUrl => {
                userMessageContent.push({
                    type: 'image_url',
                    image_url: { url: imageDataUrl }
                });
            });
        } else {
            // Text-only content
            userMessageContent = message;
        }
        
        messages.push({
            role: 'user',
            content: userMessageContent
        });
        
        // Get available tools for the active RAG collection
        let availableTools = [];
        try {
            const toolsResponse = await apiCall('/v1/tools');
            if (toolsResponse.tools && toolsResponse.tools.length > 0) {
                availableTools = toolsResponse.tools;
                console.log(`Found ${availableTools.length} available tools for RAG collection`);
            }
        } catch (error) {
            console.log('No tools available or error fetching tools:', error);
        }
        
        // Prepare OpenAI chat completion request
        const requestBody = {
            model: model,
            messages: messages,
            max_tokens: 2048,
            temperature: 0.7,
            stream: true
        };
        
        // Add tools if available
        if (availableTools.length > 0) {
            requestBody.tools = availableTools;
            requestBody.tool_choice = 'auto';
        }
        
        // Add RAG collection context by adding a system message
        if (ragCollection) {
            // Insert system message at the beginning to include RAG context instruction
            messages.unshift({
                role: 'system',
                content: `You have access to a RAG collection named "${ragCollection}". Use the context from this collection to provide more accurate and detailed responses when relevant.`
            });
        }
        
        console.log('=== OPENAI REQUEST DEBUG ===');
        console.log('Request body:', JSON.stringify(requestBody, null, 2));
        console.log('=== END DEBUG ===');
        
        // Send to OpenAI-compatible endpoint
        const response = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        // Stream the response (OpenAI format)
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value, { stream: true });
            
            // Parse Server-Sent Events format
            const lines = chunk.split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6); // Remove 'data: ' prefix
                    
                    if (data === '[DONE]') {
                        break;
                    }
                    
                    try {
                        const parsed = JSON.parse(data);
                        const choice = parsed.choices?.[0];
                        if (choice?.delta?.content) {
                            fullResponse += choice.delta.content;
                            
                            // Update the assistant message
                            assistantMessage.content = fullResponse;
                            renderChatMessages();
                        }
                        
                        // Handle tool calls if present
                        if (choice?.delta?.tool_calls) {
                            console.log('Tool calls detected:', choice.delta.tool_calls);
                            // Note: For now we'll just log tool calls
                            // Full tool call handling would require more complex state management
                        }
                        
                    } catch (parseError) {
                        // Skip invalid JSON chunks
                        console.debug('Skipping invalid JSON chunk:', data);
                    }
                }
            }
        }
        
        // Mark streaming as complete
        assistantMessage.isStreaming = false;
        renderChatMessages();
        
        // Save assistant message to server
        try {
            await apiCall(`/v1/chat/sessions/${currentChatSession.session_id}/messages`, {
                method: 'POST',
                body: JSON.stringify({
                    role: 'assistant',
                    content: fullResponse,
                    model_name: model,
                    rag_collection_name: ragCollection
                })
            });
            
            // Update session stats
            currentChatSession.message_count += 2; // user + assistant
            currentChatSession.last_message_at = new Date().toISOString();
            
            // Refresh sessions list to show updated info
            await loadChatSessions();
        } catch (error) {
            console.error('Failed to save assistant message:', error);
        }
        
    } catch (error) {
        console.error('Error sending message:', error);
        showToast('Failed to get response from model', 'error');
        
        // Update assistant message with error
        assistantMessage.content = `Error: ${error.message}`;
        assistantMessage.isStreaming = false;
        renderChatMessages();
    }
}

// Handle Enter key in chat input
function handleChatInputKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendChatMessage();
    }
}

// Edit chat session title
async function editChatTitle() {
    if (!currentChatSession) return;
    
    const newTitle = prompt('Enter new title:', currentChatSession.title);
    if (!newTitle || newTitle.trim() === currentChatSession.title) return;
    
    try {
        await apiCall(`/v1/chat/sessions/${currentChatSession.session_id}/title`, {
            method: 'PUT',
            body: JSON.stringify({ title: newTitle.trim() })
        });
        
        currentChatSession.title = newTitle.trim();
        document.getElementById('current-chat-title').textContent = newTitle.trim();
        
        // Refresh sessions list
        await loadChatSessions();
        
        showToast('Chat title updated');
    } catch (error) {
        console.error('Failed to update chat title:', error);
        showToast('Failed to update chat title', 'error');
    }
}

// Delete current chat session (called from delete button in main area)
async function deleteChatSession() {
    if (!currentChatSession) return;
    
    if (!confirm(`Are you sure you want to delete "${currentChatSession.title}"?\n\nThis will permanently delete all messages in this chat.`)) {
        return;
    }
    
    try {
        await apiCall(`/v1/chat/sessions/${currentChatSession.session_id}`, {
            method: 'DELETE'
        });
        
        // Clear current session
        currentChatSession = null;
        currentChatMessages = [];
        
        // Reset UI
        document.getElementById('current-chat-title').textContent = 'RAG Chat';
        document.getElementById('edit-title-btn').classList.add('hidden');
        document.getElementById('delete-chat-btn').classList.add('hidden');
        document.getElementById('scroll-to-bottom-btn').classList.add('hidden');
        document.getElementById('no-chat-placeholder').style.display = 'block';
        document.getElementById('chat-messages').innerHTML = `
            <div class="text-center text-gray-500 py-8" id="no-chat-placeholder">
                <i class="fas fa-comments text-4xl mb-4"></i>
                <p>Select a chat or start a new conversation</p>
            </div>
        `;
        
        // Disable chat input
        const chatInput = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-chat-btn');
        chatInput.disabled = true;
        sendBtn.disabled = true;
        chatInput.placeholder = 'Select a model and start chatting...';
        
        // Refresh sessions list
        await loadChatSessions();
        
        showToast('Chat session deleted');
    } catch (error) {
        console.error('Failed to delete chat session:', error);
        showToast('Failed to delete chat session', 'error');
    }
}

// Delete session from sidebar (called from trash icon in session list)
async function deleteSession(sessionId) {
    // Find the session in the list
    const session = chatSessions.find(s => s.session_id === sessionId);
    if (!session) {
        showToast('Session not found', 'error');
        return;
    }
    
    if (!confirm(`Are you sure you want to delete "${session.title}"?\n\nThis will permanently delete all messages in this chat.`)) {
        return;
    }
    
    try {
        await apiCall(`/v1/chat/sessions/${sessionId}`, {
            method: 'DELETE'
        });
        
        // If this was the current session, clear it
        if (currentChatSession && currentChatSession.session_id === sessionId) {
            currentChatSession = null;
            currentChatMessages = [];
            
            // Reset UI
            document.getElementById('current-chat-title').textContent = 'RAG Chat';
            document.getElementById('edit-title-btn').classList.add('hidden');
            document.getElementById('delete-chat-btn').classList.add('hidden');
            document.getElementById('scroll-to-bottom-btn').classList.add('hidden');
            document.getElementById('no-chat-placeholder').style.display = 'block';
            document.getElementById('chat-messages').innerHTML = `
                <div class="text-center text-gray-500 py-8" id="no-chat-placeholder">
                    <i class="fas fa-comments text-4xl mb-4"></i>
                    <p>Select a chat or start a new conversation</p>
                </div>
            `;
            
            // Disable chat input
            const chatInput = document.getElementById('chat-input');
            const sendBtn = document.getElementById('send-chat-btn');
            chatInput.disabled = true;
            sendBtn.disabled = true;
            chatInput.placeholder = 'Select a model and start chatting...';
        }
        
        // Refresh sessions list
        await loadChatSessions();
        
        showToast('Chat session deleted');
    } catch (error) {
        console.error('Failed to delete chat session:', error);
        showToast('Failed to delete chat session', 'error');
    }
}

// Edit session title from sidebar (called from edit icon in session list)
async function editSessionTitle(sessionId) {
    // Find the session in the list
    const session = chatSessions.find(s => s.session_id === sessionId);
    if (!session) {
        showToast('Session not found', 'error');
        return;
    }
    
    const newTitle = prompt('Enter new title:', session.title);
    if (!newTitle || newTitle.trim() === session.title) return;
    
    try {
        await apiCall(`/v1/chat/sessions/${sessionId}/title`, {
            method: 'PUT',
            body: JSON.stringify({ title: newTitle.trim() })
        });
        
        // Update local session data
        session.title = newTitle.trim();
        
        // If this is the current session, update the main title too
        if (currentChatSession && currentChatSession.session_id === sessionId) {
            currentChatSession.title = newTitle.trim();
            document.getElementById('current-chat-title').textContent = newTitle.trim();
        }
        
        // Refresh sessions list to show updated title
        renderChatSessions();
        
        showToast('Chat title updated');
    } catch (error) {
        console.error('Failed to update chat title:', error);
        showToast('Failed to update chat title', 'error');
    }
}

// Global variable to track available models with their capabilities
let availableChatModels = [];

// Load available models for chat
async function loadChatModels() {
    try {
        const response = await apiCall('/v1/manager/models');
        const models = response.models || [];
        availableChatModels = models.filter(m => m.type === 'text' || m.type === 'multimodal');
        
        const select = document.getElementById('chat-model-select');
        
        if (availableChatModels.length === 0) {
            select.innerHTML = '<option value="">No models available</option>';
            return;
        }
        
        select.innerHTML = availableChatModels.map(model => {
            const statusIcon = model.status === 'loaded' ? '✓' : model.status === 'loading' ? '⏳' : '○';
            const visionIcon = model.type === 'multimodal' ? ' 👁️' : '';
            return `<option value="${model.name}">${statusIcon} ${model.name}${visionIcon}</option>`;
        }).join('');
        
        // Try to maintain current selection or select first loaded model
        const loadedModels = availableChatModels.filter(m => m.status === 'loaded');
        if (loadedModels.length > 0 && !select.value) {
            select.value = loadedModels[0].name;
        }
        
        // Enable/disable input based on model selection
        updateChatInputState();
    } catch (error) {
        console.error('Failed to load chat models:', error);
        document.getElementById('chat-model-select').innerHTML = '<option value="">Error loading models</option>';
        availableChatModels = [];
    }
}

// Load RAG collections for chat
async function loadChatRagCollections() {
    try {
        const response = await apiCall('/v1/rag/collections');
        const collections = response.collections || [];
        const select = document.getElementById('chat-rag-select');
        
        select.innerHTML = '<option value="">No RAG</option>' + collections.map(col => 
            `<option value="${col.name}">${col.name} ${col.is_active ? '(Active)' : ''}</option>`
        ).join('');
        
        // Select active collection by default
        const activeCollection = collections.find(c => c.is_active);
        if (activeCollection) {
            select.value = activeCollection.name;
        }
    } catch (error) {
        console.error('Failed to load RAG collections for chat:', error);
    }
}

// Update chat input state based on selections
function updateChatInputState() {
    const model = document.getElementById('chat-model-select').value;
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-chat-btn');
    const imageUploadBtn = document.getElementById('image-upload-btn');
    
    // Find the selected model to check if it supports vision
    const selectedModel = availableChatModels.find(m => m.name === model);
    const supportsVision = selectedModel && selectedModel.type === 'multimodal';
    
    if (model && currentChatSession) {
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.placeholder = supportsVision ? 'Type your message or attach images...' : 'Type your message...';
        
        // Show/hide image upload button based on vision support
        if (supportsVision) {
            imageUploadBtn.classList.remove('hidden');
            imageUploadBtn.disabled = false;
        } else {
            imageUploadBtn.classList.add('hidden');
            imageUploadBtn.disabled = true;
            // Clear any selected images when switching to non-vision model
            clearSelectedImages();
        }
    } else {
        chatInput.disabled = true;
        sendBtn.disabled = true;
        imageUploadBtn.disabled = true;
        imageUploadBtn.classList.add('hidden');
        chatInput.placeholder = currentChatSession ? 'Select a model to start chatting...' : 'Create or select a chat session first...';
        // Clear any selected images when no model is selected
        clearSelectedImages();
    }
}

// Filter chat sessions based on search
function filterChatSessions(searchTerm) {
    const sessionItems = document.querySelectorAll('.chat-session-item');
    
    sessionItems.forEach(item => {
        const titleElement = item.querySelector('h4');
        const title = titleElement ? titleElement.textContent.toLowerCase() : '';
        
        if (title.includes(searchTerm)) {
            item.style.display = 'block';
        } else {
            item.style.display = 'none';
        }
    });
}

// Initialize chat tab
async function initializeChatTab() {
    await loadChatModels();
    await loadChatRagCollections();
    await loadChatSessions();
    
    // Add event listeners
    document.getElementById('chat-model-select').addEventListener('change', () => {
        updateChatInputState();
        // Also call updateChatInputState again to properly handle vision model UI updates
        setTimeout(updateChatInputState, 50);
    });
    document.getElementById('chat-rag-select').addEventListener('change', () => {
        // Update current session RAG collection preference
        if (currentChatSession) {
            currentChatSession.rag_collection_name = document.getElementById('chat-rag-select').value;
        }
    });
    
    // Chat search functionality
    document.getElementById('chat-search').addEventListener('input', (e) => {
        const searchTerm = e.target.value.toLowerCase();
        filterChatSessions(searchTerm);
    });
}

// Legacy function for backward compatibility
function newChat() {
    newChatSession();
}

// Legacy function for backward compatibility  
function sendMessage() {
    sendChatMessage();
}

// Global variables for image upload
let selectedImages = [];
const MAX_IMAGES = 5;
const MAX_IMAGE_SIZE = 10 * 1024 * 1024; // 10MB
const ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'];

// Trigger image file selection
function triggerImageUpload() {
    const imageInput = document.getElementById('image-input');
    imageInput.click();
}

// Handle image file selection
function handleImageSelection(event) {
    const files = Array.from(event.target.files);
    
    for (const file of files) {
        // Validate file type
        if (!ALLOWED_IMAGE_TYPES.includes(file.type)) {
            showToast(`Invalid file type: ${file.name}. Only JPEG, PNG, GIF, and WebP are supported.`, 'error');
            continue;
        }
        
        // Validate file size
        if (file.size > MAX_IMAGE_SIZE) {
            showToast(`File too large: ${file.name}. Maximum size is 10MB.`, 'error');
            continue;
        }
        
        // Check if we've reached the maximum number of images
        if (selectedImages.length >= MAX_IMAGES) {
            showToast(`Maximum ${MAX_IMAGES} images allowed.`, 'error');
            break;
        }
        
        // Check if image is already selected
        if (selectedImages.some(img => img.name === file.name && img.size === file.size)) {
            showToast(`Image already selected: ${file.name}`, 'warning');
            continue;
        }
        
        // Add the image to the selected list
        const imageData = {
            file: file,
            name: file.name,
            size: file.size,
            type: file.type,
            id: Date.now() + Math.random() // Unique identifier
        };
        
        selectedImages.push(imageData);
    }
    
    // Clear the input so the same file can be selected again if needed
    event.target.value = '';
    
    // Update the preview
    updateImagePreview();
}

// Update image preview display
function updateImagePreview() {
    const container = document.getElementById('image-preview-container');
    const previewsContainer = document.getElementById('image-previews');
    
    if (selectedImages.length === 0) {
        container.classList.add('hidden');
        previewsContainer.innerHTML = '';
        return;
    }
    
    container.classList.remove('hidden');
    
    previewsContainer.innerHTML = selectedImages.map(imageData => {
        const sizeText = formatFileSize(imageData.size);
        return `
            <div class="relative bg-gray-700 rounded-lg p-2 flex items-center space-x-2 max-w-xs">
                <div class="flex-shrink-0">
                    <img id="preview-${imageData.id}" src="" alt="${imageData.name}" 
                        class="w-12 h-12 object-cover rounded border border-gray-600">
                </div>
                <div class="flex-1 min-w-0">
                    <div class="text-white text-sm truncate">${imageData.name}</div>
                    <div class="text-gray-400 text-xs">${sizeText}</div>
                </div>
                <button onclick="removeImage('${imageData.id}')" 
                    class="flex-shrink-0 text-gray-400 hover:text-red-400 p-1">
                    <i class="fas fa-times text-xs"></i>
                </button>
            </div>
        `;
    }).join('');
    
    // Load image previews
    selectedImages.forEach(imageData => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = document.getElementById(`preview-${imageData.id}`);
            if (img) {
                img.src = e.target.result;
            }
        };
        reader.readAsDataURL(imageData.file);
    });
}

// Remove an image from the selection
function removeImage(imageId) {
    selectedImages = selectedImages.filter(img => img.id != imageId);
    updateImagePreview();
}

// Clear all selected images
function clearSelectedImages() {
    selectedImages = [];
    updateImagePreview();
}

// Re-submit a previous user message
function resubmitMessage(messageIndex) {
    if (messageIndex < 0 || messageIndex >= currentChatMessages.length) {
        showToast('Invalid message index', 'error');
        return;
    }
    
    const message = currentChatMessages[messageIndex];
    if (message.role !== 'user') {
        showToast('Can only re-submit user messages', 'error');
        return;
    }
    
    // Copy the message content to the chat input
    const chatInput = document.getElementById('chat-input');
    chatInput.value = message.content;
    
    // Focus the input so user can see it and modify if needed
    chatInput.focus();
    
    showToast('Message copied to input field', 'success');
}

// Scroll to the bottom of the chat (to the input area)
function scrollToBottom() {
    // With the new layout, we only need to scroll the chat messages container
    const chatContainer = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    
    if (chatContainer) {
        // Smoothly scroll the chat messages container to the bottom
        chatContainer.scrollTo({
            top: chatContainer.scrollHeight,
            behavior: 'smooth'
        });
    }
    
    // Focus the input field after a short delay
    if (chatInput) {
        setTimeout(() => {
            chatInput.focus();
        }, 300);
    }
}

// Format file size for display
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Export functions for global use
window.loadChatSessions = loadChatSessions;
window.newChatSession = newChatSession;
window.loadChatSession = loadChatSession;
window.sendChatMessage = sendChatMessage;
window.handleChatInputKeydown = handleChatInputKeydown;
window.editChatTitle = editChatTitle;
window.deleteChatSession = deleteChatSession;
window.deleteSession = deleteSession;
window.editSessionTitle = editSessionTitle;
window.loadChatModels = loadChatModels;
window.loadChatRagCollections = loadChatRagCollections;
window.initializeChatTab = initializeChatTab;
window.newChat = newChat;
window.sendMessage = sendMessage;
window.triggerImageUpload = triggerImageUpload;
window.handleImageSelection = handleImageSelection;
window.removeImage = removeImage;
window.clearSelectedImages = clearSelectedImages;
window.resubmitMessage = resubmitMessage;
window.scrollToBottom = scrollToBottom;

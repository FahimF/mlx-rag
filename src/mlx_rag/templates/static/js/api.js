/**
 * MLX-RAG Admin Dashboard - API Utilities
 * Core API functions and helper utilities
 */

// API functions
async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        showToast(`API Error: ${error.message}`, 'error');
        throw error;
    }
}

// Utility functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 GB';
    const gb = bytes / (1024 * 1024 * 1024);
    return gb.toFixed(1) + ' GB';
}

function formatLocalTime(utcTimeString) {
    if (!utcTimeString) return '';
    try {
        // Debug: Log the original timestamp
        console.debug('Formatting timestamp:', utcTimeString);

        // Ensure the UTC time string is properly formatted for parsing
        let dateString = utcTimeString;

        // If it doesn't end with 'Z' or have timezone info, assume it's UTC
        if (!dateString.includes('Z') && !dateString.includes('+') && !dateString.includes('-', 10)) {
            dateString = dateString + 'Z';
            console.debug('Added Z suffix:', dateString);
        }

        // Parse the UTC time string and convert to local time
        const utcDate = new Date(dateString);

        // Check if the date is valid
        if (isNaN(utcDate.getTime())) {
            console.error('Invalid date:', utcTimeString);
            return utcTimeString;
        }

        // Debug: Log the parsed date
        console.debug('Parsed date (local):', utcDate.toString());
        console.debug('Current time:', new Date().toString());

        // Format as local date/time with a more readable format
        const options = {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            hour12: true
        };

        const formatted = utcDate.toLocaleString(undefined, options);
        console.debug('Formatted result:', formatted);

        return formatted;
    } catch (error) {
        console.error('Error formatting time:', error, 'Input:', utcTimeString);
        return utcTimeString; // Fallback to original string
    }
}

function formatTimeAgo(timestamp) {
    if (!timestamp) return 'Just now';
    
    const now = new Date();
    const time = new Date(timestamp);
    const diffMs = now - time;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    
    return time.toLocaleDateString();
}

// Status and type helper functions
function getStatusClass(status) {
    switch (status) {
        case 'loaded': return 'status-loaded';
        case 'loading': return 'status-loading';
        case 'downloading': return 'status-downloading';
        case 'failed': return 'status-failed';
        default: return 'status-unloaded';
    }
}

function getRagStatusClass(status) {
    switch (status) {
        case 'READY': return 'status-ready';
        case 'PROCESSING': return 'status-processing';
        case 'ERROR': return 'status-error';
        default: return 'status-unloaded';
    }
}

function getRagStatusText(status) {
    switch (status) {
        case 'READY': return 'Ready';
        case 'PROCESSING': return 'Processing...';
        case 'ERROR': return 'Error';
        default: return status || 'Unknown';
    }
}

function getModelTypeClass(modelType) {
    switch (modelType) {
        case 'text': return 'bg-blue-700 text-blue-300';
        case 'multimodal': return 'bg-purple-700 text-purple-300';
        case 'vision': return 'bg-pink-700 text-pink-300';
        case 'audio': return 'bg-orange-700 text-orange-300';
        case 'embedding': return 'bg-emerald-700 text-emerald-300';
        default: return 'bg-gray-700 text-gray-300';
    }
}

function getModelTypeLabel(modelType) {
    switch (modelType) {
        case 'text': return 'Text';
        case 'multimodal': return 'Multimodal';
        case 'vision': return 'Vision';
        case 'audio': return 'Audio';
        case 'embedding': return 'Embedding';
        default: return modelType || 'Unknown';
    }
}

// Simple markdown parser for basic formatting
function parseMarkdown(text) {
    if (!text) return text;
    
    // Escape HTML first to prevent XSS
    let html = escapeHtml(text);
    
    // Convert markdown to HTML
    html = html
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')  // Bold
        .replace(/\*(.+?)\*/g, '<em>$1</em>')             // Italic
        .replace(/`(.+?)`/g, '<code class="bg-gray-600 px-1 py-0.5 rounded text-sm">$1</code>')  // Inline code
        .replace(/```([\s\S]+?)```/g, '<pre class="bg-gray-900 p-3 rounded-lg overflow-x-auto mt-2 mb-2"><code>$1</code></pre>')  // Code blocks
        .replace(/\n/g, '<br>');  // Line breaks
    
    return html;
}

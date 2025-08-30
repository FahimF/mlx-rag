/**
 * MLX-RAG Admin Dashboard - UI Utilities
 * Modal, toast, loading, and other UI management functions
 */

// Global UI state
let progressPollingInterval = null;
let loadingStartTime = null;

// Enhanced progress tracking functions
function showProgressModal(text = 'Loading...', showProgress = false) {
    document.getElementById('loading-text').textContent = text;
    document.getElementById('loading-modal').classList.remove('hidden');

    if (showProgress) {
        document.getElementById('loading-details').classList.remove('hidden');
        document.getElementById('loading-stage').textContent = 'Preparing...';
        document.getElementById('loading-time').textContent = '0s';
        document.getElementById('loading-progress').style.width = '0%';
        document.getElementById('loading-status').textContent = '';
        loadingStartTime = Date.now();
    } else {
        document.getElementById('loading-details').classList.add('hidden');
    }
}

function hideProgressModal() {
    document.getElementById('loading-modal').classList.add('hidden');
    document.getElementById('loading-details').classList.add('hidden');
    if (progressPollingInterval) {
        clearInterval(progressPollingInterval);
        progressPollingInterval = null;
    }
    loadingStartTime = null;
}

function updateProgress(stage, progress, status, elapsed) {
    document.getElementById('loading-stage').textContent = stage;
    document.getElementById('loading-progress').style.width = `${progress}%`;
    document.getElementById('loading-status').textContent = status;

    if (elapsed !== undefined) {
        document.getElementById('loading-time').textContent = `${elapsed}s`;
    } else if (loadingStartTime) {
        const elapsedSeconds = Math.floor((Date.now() - loadingStartTime) / 1000);
        document.getElementById('loading-time').textContent = `${elapsedSeconds}s`;
    }
}

// Legacy functions for compatibility
function showLoading(text = 'Loading...') {
    showProgressModal(text, false);
}

function hideLoading() {
    hideProgressModal();
}

function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    const icon = document.getElementById('toast-icon');
    const messageEl = document.getElementById('toast-message');

    let iconClass = 'fas fa-check-circle text-green-400 mr-3';
    if (type === 'error') {
        iconClass = 'fas fa-exclamation-circle text-red-400 mr-3';
    } else if (type === 'warning') {
        iconClass = 'fas fa-exclamation-triangle text-yellow-400 mr-3';
    }

    icon.className = iconClass;
    messageEl.textContent = message;

    toast.classList.remove('hidden');
    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3000);
}

// Tab switching functionality
function switchTab(tabName) {
    // Stop RAG auto-refresh when switching away from RAG tab
    if (window.ragRefreshInterval && tabName !== 'rag') {
        stopRagAutoRefresh();
    }
    
    // Update tab buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active', 'border-blue-400', 'text-blue-400');
        btn.classList.add('border-transparent', 'text-gray-300');
    });

    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active', 'border-blue-400', 'text-blue-400');
    document.querySelector(`[data-tab="${tabName}"]`).classList.remove('border-transparent', 'text-gray-300');

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.add('hidden');
    });
    document.getElementById(`${tabName}-tab`).classList.remove('hidden');

    // Load tab-specific data
    if (tabName === 'models') {
        refreshModels();
    } else if (tabName === 'monitor') {
        loadSystemInfo();
    } else if (tabName === 'api-test') {
        loadTestModels();
    } else if (tabName === 'transcribe') {
        loadTranscriptionModels();
    } else if (tabName === 'settings') {
        loadSettings();
    } else if (tabName === 'rag') {
        loadRagCollections();
        startRagAutoRefresh();
    } else if (tabName === 'chat') {
        loadChatModels();
        loadChatRagCollections();
        loadChatSessions();
    }
}

// Restart dialog functions
function showRestartDialog() {
    document.getElementById('restart-modal').classList.remove('hidden');
}

function closeRestartDialog() {
    document.getElementById('restart-modal').classList.add('hidden');
}

async function restartServerNow() {
    try {
        showLoading('Restarting server...');
        closeRestartDialog();

        // Call restart endpoint
        await apiCall('/v1/system/restart', {
            method: 'POST'
        });

        // Show success message
        showToast('Server restarting... Please wait a moment then refresh the page.', 'success');

        // Auto-refresh page after delay
        setTimeout(() => {
            window.location.reload();
        }, 5000);

    } catch (error) {
        showToast('Failed to restart server. Please restart manually.', 'error');
    } finally {
        hideLoading();
    }
}

// Initialize tab switching
function initializeTabs() {
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;
            switchTab(tabName);
        });
    });
}

// Export functions for global use
window.showProgressModal = showProgressModal;
window.hideProgressModal = hideProgressModal;
window.updateProgress = updateProgress;
window.showLoading = showLoading;
window.hideLoading = hideLoading;
window.showToast = showToast;
window.switchTab = switchTab;
window.showRestartDialog = showRestartDialog;
window.closeRestartDialog = closeRestartDialog;
window.restartServerNow = restartServerNow;
window.initializeTabs = initializeTabs;

/**
 * MLX-RAG Admin Dashboard - Dashboard Functions
 * System status, monitoring, and settings functionality
 */

// Global system state
let systemStatus = {};

// Dashboard functions
async function updateDashboard() {
    try {
        const status = await apiCall('/v1/system/status');
        systemStatus = status;

        // Update dashboard cards
        document.getElementById('system-health').textContent = status.status === 'running' ? 'Healthy' : 'Unhealthy';
        document.getElementById('loaded-count').textContent = status.model_manager.loaded_models_count;
        document.getElementById('memory-percent').textContent = status.model_manager.memory_usage_percent.toFixed(1) + '%';
        document.getElementById('queue-size').textContent = status.model_manager.queue_size;

        // Update header
        document.getElementById('memory-usage').textContent =
            `${status.model_manager.total_model_memory_gb.toFixed(1)}GB / ${status.system.memory.total_gb.toFixed(1)}GB`;
    } catch (error) {
        console.error('Failed to update dashboard:', error);
    }
}

// Monitor functions
async function loadSystemInfo() {
    try {
        const status = await apiCall('/v1/system/status');
        const systemInfo = document.getElementById('system-info');
        const modelStats = document.getElementById('model-stats');

        systemInfo.innerHTML = `
            <div><strong>Platform:</strong> ${status.system.platform} ${status.system.architecture}</div>
            <div><strong>Processor:</strong> ${status.system.processor}</div>
            <div><strong>Apple Silicon:</strong> ${status.system.is_apple_silicon ? 'Yes' : 'No'}</div>
            <div><strong>MLX Compatible:</strong> ${status.system.mlx_compatible ? 'Yes' : 'No'}</div>
            <div><strong>Total Memory:</strong> ${status.system.memory.total_gb.toFixed(1)} GB</div>
            <div><strong>Available Memory:</strong> ${status.system.memory.available_gb.toFixed(1)} GB</div>
            <div><strong>Memory Usage:</strong> ${status.system.memory.percent_used.toFixed(1)}%</div>
        `;

        modelStats.innerHTML = `
            <div><strong>Loaded Models:</strong> ${status.model_manager.loaded_models_count}</div>
            <div><strong>Max Concurrent:</strong> ${status.model_manager.max_concurrent_models}</div>
            <div><strong>Queue Size:</strong> ${status.model_manager.queue_size}</div>
            <div><strong>Model Memory:</strong> ${status.model_manager.total_model_memory_gb.toFixed(1)} GB</div>
            <div><strong>Memory Usage:</strong> ${status.model_manager.memory_usage_percent.toFixed(1)}%</div>
            <div><strong>Auto-unload:</strong> ${status.model_manager.auto_unload_enabled ? 'Enabled' : 'Disabled'}</div>
            <div><strong>Timeout:</strong> ${status.model_manager.inactivity_timeout_minutes} min</div>
        `;
    } catch (error) {
        showToast('Failed to load system info', 'error');
    }
}

// Settings functions
async function loadSettings() {
    try {
        const settings = await apiCall('/v1/settings');
        document.getElementById('auto-unload').checked = settings.auto_unload_inactive_models;
        document.getElementById('timeout-minutes').value = settings.model_inactivity_timeout_minutes;
        document.getElementById('max-models').value = settings.max_concurrent_models;
        document.getElementById('log-level').value = settings.log_level;
        document.getElementById('server-port').value = settings.server_port;
        document.getElementById('bind-all-interfaces').checked = settings.bind_to_all_interfaces;
    } catch (error) {
        showToast('Failed to load settings', 'error');
    }
}

async function saveSettings() {
    try {
        showLoading('Saving settings...');

        // Get current settings to check for changes
        const currentSettings = await apiCall('/v1/settings');

        const newSettings = {
            auto_unload_inactive_models: document.getElementById('auto-unload').checked,
            model_inactivity_timeout_minutes: parseInt(document.getElementById('timeout-minutes').value),
            max_concurrent_models: parseInt(document.getElementById('max-models').value),
            log_level: document.getElementById('log-level').value,
            server_port: parseInt(document.getElementById('server-port').value),
            bind_to_all_interfaces: document.getElementById('bind-all-interfaces').checked
        };

        // Check if network settings changed
        const networkSettingsChanged =
            currentSettings.server_port !== newSettings.server_port ||
            currentSettings.bind_to_all_interfaces !== newSettings.bind_to_all_interfaces;

        // Save each setting
        for (const [key, value] of Object.entries(newSettings)) {
            await apiCall(`/v1/settings/${key}`, {
                method: 'PUT',
                body: JSON.stringify({ value })
            });
        }

        showToast('Settings saved successfully');

        // If network settings changed, offer immediate restart
        if (networkSettingsChanged) {
            setTimeout(() => {
                showRestartDialog();
            }, 500);
        }

    } catch (error) {
        showToast('Failed to save settings', 'error');
    } finally {
        hideLoading();
    }
}

// Initialize dashboard with auto-refresh
function initializeDashboard() {
    updateDashboard();
    
    // Auto-refresh dashboard every 30 seconds
    setInterval(updateDashboard, 30000);
}

// RAG collection management functions
let ragRefreshInterval = null;

// Folder picker functionality
function selectFolder() {
    const folderPicker = document.getElementById('folder-picker');
    folderPicker.click();
    
    folderPicker.onchange = function(event) {
        const files = event.target.files;
        if (files.length > 0) {
            // Get the common path from the first file
            const firstFile = files[0];
            const relativePath = firstFile.webkitRelativePath;
            
            // Extract the folder name (top-level folder in selection)
            const pathParts = relativePath.split('/');
            const selectedFolderName = pathParts[0];
            
            // Since browsers don't provide full system paths due to security,
            // we'll prompt the user to provide the full path
            const userProvidedPath = prompt(
                `Browser security prevents us from getting the full path.\n\n` +
                `You selected the folder: "${selectedFolderName}"\n` +
                `Found ${files.length} files in the selection.\n\n` +
                `Please enter the full path to this folder:`,
                `/path/to/${selectedFolderName}`
            );
            
            if (userProvidedPath && userProvidedPath.trim()) {
                // Set the full path provided by the user
                document.getElementById('rag-folder-path').value = userProvidedPath.trim();
                showToast(`Folder path set: ${userProvidedPath.trim()}`);
            } else {
                // Fallback to just the folder name if user cancels
                document.getElementById('rag-folder-path').value = selectedFolderName;
                showToast(`Selected folder: ${selectedFolderName} (${files.length} files found)`);
            }
        }
    };
}

// Load RAG collections
async function loadRagCollections() {
    try {
        const response = await apiCall('/v1/rag/collections');
        const collections = response.collections || [];
        const container = document.getElementById('rag-collections-list');
        
        if (collections.length === 0) {
            container.innerHTML = `
                <div class="text-center text-gray-500 py-8">
                    <i class="fas fa-folder-open text-3xl mb-2"></i>
                    <p class="text-sm">No RAG collections found</p>
                    <p class="text-xs text-gray-400 mt-1">Create your first collection to get started</p>
                </div>
            `;
            return;
        }
        
        container.innerHTML = collections.map(collection => {
            const statusClass = collection.status === 'ready' ? 'text-green-400' : 
                               collection.status === 'processing' ? 'text-yellow-400' : 'text-red-400';
            const statusIcon = collection.status === 'ready' ? 'fa-check-circle' : 
                              collection.status === 'processing' ? 'fa-clock' : 'fa-exclamation-circle';
                              
            return `
                <div class="bg-gray-700 rounded-lg p-4 hover:bg-gray-600 transition-colors">
                    <div class="flex items-start justify-between">
                        <div class="flex-1 min-w-0">
                            <h4 class="text-white font-medium text-sm truncate">${escapeHtml(collection.name)}</h4>
                            <div class="flex items-center mt-1">
                                <i class="fas ${statusIcon} ${statusClass} mr-1 text-xs"></i>
                                <span class="text-xs ${statusClass} capitalize">${collection.status}</span>
                                ${collection.is_active ? '<span class="ml-2 px-2 py-1 bg-blue-600 text-white text-xs rounded">Active</span>' : ''}
                            </div>
                            <p class="text-xs text-gray-400 mt-1 truncate" title="${collection.path}">
                                <i class="fas fa-folder mr-1"></i>${collection.path}
                            </p>
                        </div>
                        <div class="flex items-center space-x-1 ml-2">
                            ${!collection.is_active ? `
                                <button onclick="activateRagCollection('${collection.name}')" 
                                        class="text-gray-400 hover:text-green-400 p-1" title="Activate">
                                    <i class="fas fa-play text-xs"></i>
                                </button>
                            ` : ''}
                            <button onclick="reprocessRagCollection('${collection.name}')" 
                                    class="text-gray-400 hover:text-blue-400 p-1" title="Reprocess">
                                <i class="fas fa-sync-alt text-xs"></i>
                            </button>
                            <button onclick="deleteRagCollection('${collection.name}')" 
                                    class="text-gray-400 hover:text-red-400 p-1" title="Delete">
                                <i class="fas fa-trash text-xs"></i>
                            </button>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    } catch (error) {
        console.error('Failed to load RAG collections:', error);
        document.getElementById('rag-collections-list').innerHTML = `
            <div class="text-center text-red-500 py-8">
                <i class="fas fa-exclamation-triangle text-3xl mb-2"></i>
                <p class="text-sm">Failed to load RAG collections</p>
            </div>
        `;
    }
}

// Create new RAG collection
async function createRagCollection() {
    const name = document.getElementById('rag-collection-name').value.trim();
    const path = document.getElementById('rag-folder-path').value.trim();
    
    if (!name || !path) {
        showToast('Please provide both collection name and folder path', 'error');
        return;
    }
    
    try {
        showLoading('Creating RAG collection...');
        
        // Use query parameters as expected by the server endpoint
        const url = new URL('/v1/rag/collections', window.location.origin);
        url.searchParams.append('name', name);
        url.searchParams.append('path', path);
        
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        // Clear the form
        document.getElementById('rag-collection-name').value = '';
        document.getElementById('rag-folder-path').value = '';
        
        // Refresh the collections list
        await loadRagCollections();
        
        showToast('RAG collection created successfully');
    } catch (error) {
        console.error('Failed to create RAG collection:', error);
        showToast('Failed to create RAG collection: ' + (error.message || 'Unknown error'), 'error');
    } finally {
        hideLoading();
    }
}

// Refresh RAG collections
async function refreshRagCollections() {
    await loadRagCollections();
    showToast('RAG collections refreshed');
}

// Activate RAG collection
async function activateRagCollection(collectionName) {
    try {
        showLoading('Activating collection...');
        
        await apiCall(`/v1/rag/collections/${encodeURIComponent(collectionName)}/activate`, {
            method: 'POST'
        });
        
        await loadRagCollections();
        showToast(`Collection '${collectionName}' activated successfully`);
    } catch (error) {
        console.error('Failed to activate RAG collection:', error);
        showToast('Failed to activate collection: ' + (error.message || 'Unknown error'), 'error');
    } finally {
        hideLoading();
    }
}

// Reprocess RAG collection
async function reprocessRagCollection(collectionName) {
    if (!confirm(`Are you sure you want to reprocess the '${collectionName}' collection?\n\nThis will re-index all documents and may take some time.`)) {
        return;
    }
    
    try {
        showLoading('Starting reprocessing...');
        
        await apiCall(`/v1/rag/collections/${encodeURIComponent(collectionName)}/reprocess`, {
            method: 'POST'
        });
        
        await loadRagCollections();
        showToast(`Reprocessing started for '${collectionName}'`);
    } catch (error) {
        console.error('Failed to reprocess RAG collection:', error);
        showToast('Failed to reprocess collection: ' + (error.message || 'Unknown error'), 'error');
    } finally {
        hideLoading();
    }
}

// Delete RAG collection
async function deleteRagCollection(collectionName) {
    if (!confirm(`Are you sure you want to delete the '${collectionName}' collection?\n\nThis action cannot be undone.`)) {
        return;
    }
    
    try {
        showLoading('Deleting collection...');
        
        await apiCall(`/v1/rag/collections/${encodeURIComponent(collectionName)}`, {
            method: 'DELETE'
        });
        
        await loadRagCollections();
        showToast(`Collection '${collectionName}' deleted successfully`);
    } catch (error) {
        console.error('Failed to delete RAG collection:', error);
        showToast('Failed to delete collection: ' + (error.message || 'Unknown error'), 'error');
    } finally {
        hideLoading();
    }
}

// Start auto-refresh for RAG collections
function startRagAutoRefresh() {
    if (ragRefreshInterval) {
        clearInterval(ragRefreshInterval);
    }
    
    // Auto-refresh RAG collections every 10 seconds when the tab is active
    ragRefreshInterval = setInterval(loadRagCollections, 10000);
}

// Stop auto-refresh for RAG collections
function stopRagAutoRefresh() {
    if (ragRefreshInterval) {
        clearInterval(ragRefreshInterval);
        ragRefreshInterval = null;
    }
}

// HTML escape utility function
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Export functions for global use
window.updateDashboard = updateDashboard;
window.loadSystemInfo = loadSystemInfo;
window.loadSettings = loadSettings;
window.saveSettings = saveSettings;
window.initializeDashboard = initializeDashboard;
window.systemStatus = systemStatus;

// RAG functions
window.selectFolder = selectFolder;
window.loadRagCollections = loadRagCollections;
window.createRagCollection = createRagCollection;
window.refreshRagCollections = refreshRagCollections;
window.activateRagCollection = activateRagCollection;
window.reprocessRagCollection = reprocessRagCollection;
window.deleteRagCollection = deleteRagCollection;
window.startRagAutoRefresh = startRagAutoRefresh;
window.stopRagAutoRefresh = stopRagAutoRefresh;
window.escapeHtml = escapeHtml;

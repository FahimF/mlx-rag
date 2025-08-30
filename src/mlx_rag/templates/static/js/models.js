/**
 * MLX-RAG Admin Dashboard - Models Functions
 * Model management, discovery, loading, and installation functionality
 */

// Global models state
let currentModels = [];

// Model management functions
async function refreshModels(silent = false) {
    try {
        if (!silent) {
            showLoading('Refreshing models...');
        }
        // First, trigger a rescan of the cache
        await apiCall('/v1/manager/models/rescan', { method: 'POST' });

        const response = await apiCall('/v1/manager/models');
        currentModels = response.models || [];
        renderModels();

        // Auto-refresh if any models are in loading state
        const hasLoadingModels = currentModels.some(m => m.status === 'loading' || m.status === 'downloading');
        if (hasLoadingModels && !window.progressPollingInterval) {
            setTimeout(() => refreshModels(true), 3000); // Refresh again in 3 seconds
        }
    } catch (error) {
        if (!silent) {
            showToast('Failed to load models', 'error');
        }
    } finally {
        if (!silent) {
            hideLoading();
        }
    }
}

function renderModels() {
    const container = document.getElementById('models-list');

    if (currentModels.length === 0) {
        container.innerHTML = `
            <div class="text-center py-12">
                <i class="fas fa-cube text-gray-500 text-4xl mb-4"></i>
                <p class="text-gray-400">No models installed</p>
                <p class="text-gray-500 text-sm">Use the Discover tab to find and install models</p>
            </div>
        `;
        return;
    }

    container.innerHTML = currentModels.map(model => `
        <div class="bg-gray-800 rounded-lg p-4 card-hover">
            <div class="flex items-center justify-between">
                <div class="flex-1">
                    <div class="flex items-center mb-2">
                        <span class="status-indicator ${getStatusClass(model.status)}"
                              ${model.status === 'loading' ? 'title="Model is currently loading..."' : ''}
                              ${model.status === 'downloading' ? 'title="Model is downloading from HuggingFace..."' : ''}></span>
                        <h3 class="text-lg font-medium text-white">
                            ${model.huggingface_id ?
                                `<a href="https://huggingface.co/${model.huggingface_id}" target="_blank" rel="noopener noreferrer"
                                   class="hover:text-blue-300 transition-colors cursor-pointer">
                                    ${model.name}
                                </a>` :
                                model.name
                            }
                        </h3>
                        <span class="ml-2 px-2 py-1 text-xs bg-gray-700 text-gray-300 rounded">${model.type}</span>
                        ${model.author && model.author !== 'unknown' ?
                            `<span class="ml-2 px-2 py-1 text-xs bg-purple-700 text-purple-300 rounded">
                                <i class="fas fa-user mr-1"></i>${model.author}
                            </span>` : ''
                        }
                    </div>
                    <div class="flex items-center space-x-4 text-sm text-gray-400">
                        <span><i class="fas fa-memory mr-1"></i>${model.memory_required_gb} GB</span>
                        <span><i class="fas fa-clock mr-1"></i>Used ${model.use_count} times</span>
                        ${model.last_used_at ? `<span><i class="fas fa-history mr-1"></i>${formatLocalTime(model.last_used_at)}</span>` : ''}
                    </div>
                </div>
                <div class="flex space-x-2">
                    ${model.status === 'loaded' ?
                        `<button onclick="unloadModel('${model.name}')" class="bg-yellow-600 hover:bg-yellow-700 text-white px-3 py-1 rounded text-sm">
                            <i class="fas fa-stop mr-1"></i>Unload
                        </button>` :
                        `<button onclick="loadModel('${model.name}')" class="bg-green-600 hover:bg-green-700 text-white px-3 py-1 rounded text-sm">
                            <i class="fas fa-play mr-1"></i>Load
                        </button>`
                    }
                    <button onclick="removeModel('${model.name}')" class="bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded text-sm">
                        <i class="fas fa-trash mr-1"></i>Remove
                    </button>
                </div>
            </div>
        </div>
    `).join('');
}

async function loadModel(modelName) {
    try {
        // Show enhanced progress modal
        showProgressModal(`Loading ${modelName}...`, true);
        updateProgress('Initializing...', 5, 'Starting model loading process');

        // Start the loading request (non-blocking)
        const loadPromise = apiCall(`/v1/models/${modelName}/load`, { method: 'POST' });

        // Start progress polling
        await pollModelLoadingProgress(modelName, loadPromise);

        showToast(`Model ${modelName} loaded successfully`);
        refreshModels();
        updateDashboard();
    } catch (error) {
        showToast(`Failed to load ${modelName}`, 'error');
    } finally {
        hideProgressModal();
    }
}

async function pollModelLoadingProgress(modelName, loadPromise) {
    return new Promise((resolve, reject) => {
        let pollCount = 0;
        let lastStatus = null;
        let lastProgress = 0;

        window.progressPollingInterval = setInterval(async () => {
            try {
                pollCount++;
                const elapsed = Math.floor((Date.now() - window.loadingStartTime) / 1000);

                // Check model status with real progress data
                const statusResponse = await apiCall(`/v1/manager/models/${modelName}/status`);
                const status = statusResponse.status;

                if (status !== lastStatus) {
                    lastStatus = status;
                    console.log(`Model ${modelName} status changed to: ${status}`);
                }

                // Update progress based on REAL server data
                if (status === 'loading') {
                    // Use real download progress if available
                    const realProgress = statusResponse.download_progress || 0;
                    const downloadStage = statusResponse.download_stage || 'Loading...';
                    const downloadStatus = statusResponse.download_status || 'unknown';
                    const downloadedMB = statusResponse.downloaded_mb || 0;
                    const totalMB = statusResponse.total_mb || 0;
                    const speedMbps = statusResponse.download_speed_mbps || 0;
                    const etaSeconds = statusResponse.download_eta_seconds || 0;

                    // Build detailed status message
                    let statusMessage = downloadStage;
                    if (downloadedMB > 0 && totalMB > 0) {
                        statusMessage = `Downloaded ${downloadedMB}MB/${totalMB}MB`;
                        if (speedMbps > 0) {
                            statusMessage += ` (${speedMbps.toFixed(1)}MB/s)`;
                        }
                        if (etaSeconds > 0) {
                            const minutes = Math.floor(etaSeconds / 60);
                            const seconds = Math.floor(etaSeconds % 60);
                            statusMessage += ` - ETA: ${minutes}m ${seconds}s`;
                        }
                    } else if (elapsed > 0) {
                        statusMessage += ` (${elapsed}s elapsed)`;
                    }

                    // Use real progress or fallback to minimal simulation
                    const progress = realProgress > 0 ? realProgress : Math.min(pollCount * 0.5, 5);

                    // Only update if progress actually changed (avoid fake updates)
                    if (Math.abs(progress - lastProgress) > 0.1 || downloadStage !== lastStatus) {
                        updateProgress(downloadStage, progress, statusMessage, elapsed);
                        lastProgress = progress;
                    }

                } else if (status === 'loaded') {
                    updateProgress('Completed!', 100, 'Model loaded successfully', elapsed);
                    clearInterval(window.progressPollingInterval);
                    window.progressPollingInterval = null;
                    resolve();
                    return;
                } else if (status === 'failed') {
                    clearInterval(window.progressPollingInterval);
                    window.progressPollingInterval = null;
                    reject(new Error('Model loading failed'));
                    return;
                }

                // Check if the main loading promise completed
                if (loadPromise && loadPromise.then) {
                    loadPromise.then(() => {
                        updateProgress('Completed!', 100, 'Model loaded successfully');
                        clearInterval(window.progressPollingInterval);
                        window.progressPollingInterval = null;
                        resolve();
                    }).catch((error) => {
                        clearInterval(window.progressPollingInterval);
                        window.progressPollingInterval = null;
                        reject(error);
                    });
                    loadPromise = null; // Prevent multiple handlers
                }

            } catch (error) {
                console.warn('Progress polling error:', error);
                // Continue polling unless it's a critical error
                if (pollCount > 300) { // 5 minutes timeout (more realistic for large downloads)
                    clearInterval(window.progressPollingInterval);
                    window.progressPollingInterval = null;
                    reject(new Error('Loading timeout - please check server logs for details'));
                }
            }
        }, 1000); // Poll every second
    });
}

async function unloadModel(modelName) {
    try {
        showLoading(`Unloading ${modelName}...`);
        await apiCall(`/v1/models/${modelName}/unload`, { method: 'POST' });
        showToast(`Model ${modelName} unloaded successfully`);
        refreshModels();
        updateDashboard();
    } catch (error) {
        showToast(`Failed to unload ${modelName}`, 'error');
    } finally {
        hideLoading();
    }
}

async function removeModel(modelName) {
    if (!confirm(`Are you sure you want to remove ${modelName}?\n\nThis will:\n• Unload the model from memory\n• Remove it from the database\n• Delete downloaded files (use Shift+Click to keep files)`)) return;

    try {
        showLoading(`Removing ${modelName}...`);

        // Check if Shift key was held to keep files
        const keepFiles = event.shiftKey;
        const url = keepFiles ?
            `/v1/models/${modelName}?remove_files=false` :
            `/v1/models/${modelName}`;

        await apiCall(url, { method: 'DELETE' });

        const message = keepFiles ?
            `Model ${modelName} removed successfully (files kept)` :
            `Model ${modelName} and files removed successfully`;

        showToast(message);
        refreshModels();
        updateDashboard();
    } catch (error) {
        showToast(`Failed to remove ${modelName}`, 'error');
    } finally {
        hideLoading();
    }
}

// Model discovery functions
async function searchModels() {
    const query = document.getElementById('search-input').value;
    try {
        showLoading('Searching models...');
        const response = await apiCall(`/v1/discover/models?query=${encodeURIComponent(query)}&limit=20`);
        renderDiscoveryResults(response.models || []);
    } catch (error) {
        showToast('Failed to search models', 'error');
    } finally {
        hideLoading();
    }
}

async function loadPopularChat() {
    try {
        showLoading('Loading popular chat models...');
        const response = await apiCall('/v1/discover/categories');
        const chatModels = response.categories['Popular Chat'] || [];
        const modelDetails = await Promise.all(
            chatModels.slice(0, 10).map(async (modelId) => {
                try {
                    const detail = await apiCall(`/v1/discover/models/${encodeURIComponent(modelId)}`);
                    return detail;
                } catch (e) {
                    return null;
                }
            })
        );
        renderDiscoveryResults(modelDetails.filter(m => m !== null));
    } catch (error) {
        showToast('Failed to load popular chat models', 'error');
    } finally {
        hideLoading();
    }
}

async function loadPopularSTT() {
    try {
        showLoading('Loading STT models...');
        // Use the dedicated STT models endpoint
        const response = await apiCall('/v1/discover/stt?limit=10');
        const sttModels = response.models || [];
        renderDiscoveryResults(sttModels);
    } catch (error) {
        showToast('Failed to load STT models', 'error');
    } finally {
        hideLoading();
    }
}

async function loadVisionModels() {
    try {
        showLoading('Loading vision models...');
        // Use the dedicated vision models endpoint
        const response = await apiCall('/v1/discover/vision?limit=10');
        const visionModels = response.models || [];
        renderDiscoveryResults(visionModels);
    } catch (error) {
        showToast('Failed to load vision models', 'error');
    } finally {
        hideLoading();
    }
}

async function loadCompatibleModels() {
    try {
        showLoading('Loading compatible models...');
        const response = await apiCall('/v1/discover/compatible?limit=20');
        renderDiscoveryResults(response.models || []);
    } catch (error) {
        showToast('Failed to load compatible models', 'error');
    } finally {
        hideLoading();
    }
}

async function loadSmallModels() {
    try {
        showLoading('Loading small models...');
        const response = await apiCall('/v1/discover/models?query=4bit&limit=20');
        const smallModels = (response.models || []).filter(m => m.size_gb && m.size_gb < 10);
        renderDiscoveryResults(smallModels);
    } catch (error) {
        showToast('Failed to load small models', 'error');
    } finally {
        hideLoading();
    }
}

async function loadTrendingModels() {
    try {
        showLoading('Loading trending models...');
        const response = await apiCall('/v1/discover/trending?limit=20');
        renderDiscoveryResults(response.models || []);
    } catch (error) {
        showToast('Failed to load trending models', 'error');
    } finally {
        hideLoading();
    }
}

async function loadEmbeddingModels() {
    try {
        showLoading('Loading embedding models...');
        const response = await apiCall('/v1/discover/embeddings?limit=20');
        renderDiscoveryResults(response.models || []);
    } catch (error) {
        showToast('Failed to load embedding models', 'error');
    } finally {
        hideLoading();
    }
}

function renderDiscoveryResults(models) {
    const container = document.getElementById('discover-results');

    if (models.length === 0) {
        container.innerHTML = `
            <div class="text-center py-12">
                <i class="fas fa-search text-gray-500 text-4xl mb-4"></i>
                <p class="text-gray-400">No models found</p>
                <p class="text-gray-500 text-sm">Try a different search term</p>
            </div>
        `;
        return;
    }

    container.innerHTML = models.map(model => `
        <div class="bg-gray-800 rounded-lg p-4 card-hover">
            <div class="flex items-center justify-between">
                <div class="flex-1">
                    <div class="flex items-center mb-2">
                        <h3 class="text-lg font-medium text-white">
                            <a href="https://huggingface.co/${model.id}" target="_blank" rel="noopener noreferrer"
                               class="hover:text-blue-300 transition-colors cursor-pointer">
                                ${model.name}
                            </a>
                        </h3>
                        <span class="ml-2 px-2 py-1 text-xs ${getModelTypeClass(model.model_type)} rounded">${getModelTypeLabel(model.model_type)}</span>
                        ${model.mlx_compatible ? '<span class="ml-2 px-2 py-1 text-xs bg-green-700 text-green-300 rounded">MLX</span>' : ''}
                    </div>
                    <p class="text-sm text-gray-400 mb-2">${model.description || 'No description available'}</p>
                    <div class="flex items-center space-x-4 text-sm text-gray-400">
                        <span><i class="fas fa-download mr-1"></i>${model.downloads?.toLocaleString() || '0'}</span>
                        <span><i class="fas fa-heart mr-1"></i>${model.likes || '0'}</span>
                        ${model.size_gb && model.size_gb > 0 ? `<span><i class="fas fa-memory mr-1"></i>${model.size_gb.toFixed(1)} GB</span>` : ''}
                        <span><i class="fas fa-user mr-1"></i>${model.author}</span>
                    </div>
                </div>
                <div class="flex space-x-2">
                    <button onclick="installModel('${model.id}', '${model.name.replace(/'/g, "\\'")}')" class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded">
                        <i class="fas fa-download mr-1"></i>Install
                    </button>
                </div>
            </div>
        </div>
    `).join('');
}

async function installModel(modelId, modelName) {
    try {
        // Show enhanced progress modal for installation
        showProgressModal(`Installing ${modelName}...`, true);
        updateProgress('Preparing installation...', 5, 'Validating model information');

        const sanitizedName = modelName.toLowerCase().replace(/[^a-z0-9-]/g, '-');

        // Start installation
        updateProgress('Downloading model files...', 15, 'Contacting HuggingFace repository');

        const installPromise = apiCall('/v1/models/install', {
            method: 'POST',
            body: JSON.stringify({
                model_id: modelId,
                name: sanitizedName
            })
        });

        // Simulate installation progress
        const progressSteps = [
            { progress: 25, stage: 'Downloading model files...', status: 'Fetching model configuration' },
            { progress: 45, stage: 'Downloading model files...', status: 'Downloading model weights' },
            { progress: 70, stage: 'Processing files...', status: 'Validating downloaded files' },
            { progress: 85, stage: 'Installing model...', status: 'Setting up model in database' },
            { progress: 95, stage: 'Finalizing...', status: 'Completing installation' }
        ];

        let stepIndex = 0;
        const progressInterval = setInterval(() => {
            if (stepIndex < progressSteps.length) {
                const step = progressSteps[stepIndex];
                updateProgress(step.stage, step.progress, step.status);
                stepIndex++;
            }
        }, 2000); // Update every 2 seconds

        // Wait for installation to complete
        await installPromise;

        clearInterval(progressInterval);
        updateProgress('Installation complete!', 100, 'Model installed successfully');

        showToast(`Model ${modelName} installed successfully`);
        // Switch to models tab and refresh
        setTimeout(() => {
            switchTab('models');
        }, 1000);

    } catch (error) {
        showToast(`Failed to install ${modelName}`, 'error');
    } finally {
        setTimeout(() => {
            hideProgressModal();
        }, 1500); // Brief delay to show completion
    }
}

// Initialize search functionality
function initializeModelsSearch() {
    document.getElementById('search-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchModels();
        }
    });
}

// Export functions for global use
window.refreshModels = refreshModels;
window.loadModel = loadModel;
window.unloadModel = unloadModel;
window.removeModel = removeModel;
window.searchModels = searchModels;
window.loadPopularChat = loadPopularChat;
window.loadPopularSTT = loadPopularSTT;
window.loadVisionModels = loadVisionModels;
window.loadCompatibleModels = loadCompatibleModels;
window.loadSmallModels = loadSmallModels;
window.loadTrendingModels = loadTrendingModels;
window.loadEmbeddingModels = loadEmbeddingModels;
window.installModel = installModel;
window.initializeModelsSearch = initializeModelsSearch;
window.currentModels = currentModels;

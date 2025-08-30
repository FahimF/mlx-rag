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

// Export functions for global use
window.updateDashboard = updateDashboard;
window.loadSystemInfo = loadSystemInfo;
window.loadSettings = loadSettings;
window.saveSettings = saveSettings;
window.initializeDashboard = initializeDashboard;
window.systemStatus = systemStatus;

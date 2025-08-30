/**
 * MLX-RAG Admin Dashboard - Main Application
 * Main initialization and application orchestration
 */

// Application initialization
document.addEventListener('DOMContentLoaded', function() {
    console.log('MLX-RAG Admin Dashboard initializing...');
    
    // Initialize core components
    initializeTabs();
    initializeDashboard();
    initializeModelsSearch();
    
    // Start with dashboard tab
    switchTab('models');
    
    console.log('MLX-RAG Admin Dashboard initialized successfully');
});

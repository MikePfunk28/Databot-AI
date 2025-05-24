// Global variables
let sessionId = 'default';
let activeDatasets = [];
let allDatasets = [];

// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const datasetList = document.getElementById('datasetList');
const activeDatasetList = document.getElementById('activeDatasetList');
const uploadBtn = document.getElementById('uploadBtn');
const urlBtn = document.getElementById('urlBtn');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const useRagCheckbox = document.getElementById('useRag');
const retrieveExternalCheckbox = document.getElementById('retrieveExternal');

// Modal Elements
const uploadModal = document.getElementById('uploadModal');
const closeUploadModal = document.getElementById('closeUploadModal');
const uploadForm = document.getElementById('uploadForm');
const fileTypeSelect = document.getElementById('fileType');
const delimiterGroup = document.getElementById('delimiterGroup');
const sheetNameGroup = document.getElementById('sheetNameGroup');
const chunkSizeGroup = document.getElementById('chunkSizeGroup');

const urlModal = document.getElementById('urlModal');
const closeUrlModal = document.getElementById('closeUrlModal');
const urlForm = document.getElementById('urlForm');

const datasetInfoModal = document.getElementById('datasetInfoModal');
const closeDatasetInfoModal = document.getElementById('closeDatasetInfoModal');
const datasetInfoTitle = document.getElementById('datasetInfoTitle');
const datasetInfo = document.getElementById('datasetInfo');

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    // Load datasets
    loadDatasets();
    
    // Load active datasets
    loadActiveDatasets();
    
    // Load chat history
    loadChatHistory();
    
    // Set up event listeners
    setupEventListeners();
});

// Set up event listeners
function setupEventListeners() {
    // Send message
    sendBtn.addEventListener('click', sendMessage);
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Upload modal
    uploadBtn.addEventListener('click', () => {
        uploadModal.style.display = 'block';
    });
    
    closeUploadModal.addEventListener('click', () => {
        uploadModal.style.display = 'none';
    });
    
    // URL modal
    urlBtn.addEventListener('click', () => {
        urlModal.style.display = 'block';
    });
    
    closeUrlModal.addEventListener('click', () => {
        urlModal.style.display = 'none';
    });
    
    // Dataset info modal
    closeDatasetInfoModal.addEventListener('click', () => {
        datasetInfoModal.style.display = 'none';
    });
    
    // File type change
    fileTypeSelect.addEventListener('change', () => {
        const fileType = fileTypeSelect.value;
        
        // Show/hide relevant form groups
        delimiterGroup.style.display = fileType === 'csv' ? 'block' : 'none';
        sheetNameGroup.style.display = fileType === 'excel' ? 'block' : 'none';
        chunkSizeGroup.style.display = fileType === 'text' ? 'block' : 'none';
    });
    
    // Upload form submit
    uploadForm.addEventListener('submit', (e) => {
        e.preventDefault();
        uploadFile();
    });
    
    // URL form submit
    urlForm.addEventListener('submit', (e) => {
        e.preventDefault();
        addUrl();
    });
    
    // Clear history
    clearHistoryBtn.addEventListener('click', clearHistory);
    
    // Close modals when clicking outside
    window.addEventListener('click', (e) => {
        if (e.target === uploadModal) {
            uploadModal.style.display = 'none';
        }
        if (e.target === urlModal) {
            urlModal.style.display = 'none';
        }
        if (e.target === datasetInfoModal) {
            datasetInfoModal.style.display = 'none';
        }
    });
}

// Load datasets
async function loadDatasets() {
    try {
        const response = await fetch(`/api/datasets?session_id=${sessionId}`);
        const data = await response.json();
        
        allDatasets = data.datasets || [];
        
        renderDatasetList();
    } catch (error) {
        console.error('Error loading datasets:', error);
        datasetList.innerHTML = '<p>Error loading datasets</p>';
    }
}

// Load active datasets
async function loadActiveDatasets() {
    try {
        const response = await fetch(`/api/datasets/active?session_id=${sessionId}`);
        const data = await response.json();
        
        activeDatasets = data.active_datasets || [];
        
        renderActiveDatasetList();
    } catch (error) {
        console.error('Error loading active datasets:', error);
        activeDatasetList.innerHTML = '<p>Error loading active datasets</p>';
    }
}

// Load chat history
async function loadChatHistory() {
    try {
        const response = await fetch(`/api/history?session_id=${sessionId}`);
        const data = await response.json();
        
        const messages = data.history || [];
        
        // Clear chat messages
        chatMessages.innerHTML = '';
        
        // Add messages to chat
        messages.forEach(message => {
            if (message.role === 'user') {
                addUserMessage(message.content);
            } else if (message.role === 'assistant') {
                addAssistantMessage(message.content, message.metadata);
            }
        });
        
        // Scroll to bottom
        scrollToBottom();
    } catch (error) {
        console.error('Error loading chat history:', error);
    }
}

// Render dataset list
function renderDatasetList() {
    if (allDatasets.length === 0) {
        datasetList.innerHTML = '<p>No datasets available</p>';
        return;
    }
    
    let html = '';
    
    allDatasets.forEach(dataset => {
        const isActive = activeDatasets.includes(dataset.id);
        const buttonText = isActive ? 'Remove' : 'Add';
        const buttonAction = isActive ? 'remove' : 'add';
        
        html += `
            <div class="dataset-item">
                <div class="dataset-item-info" title="${dataset.id}">
                    ${dataset.id.substring(0, 20)}${dataset.id.length > 20 ? '...' : ''}
                </div>
                <div class="dataset-item-actions">
                    <button onclick="toggleActiveDataset('${dataset.id}', '${buttonAction}')">${buttonText}</button>
                    <button onclick="showDatasetInfo('${dataset.id}')">Info</button>
                </div>
            </div>
        `;
    });
    
    datasetList.innerHTML = html;
}

// Render active dataset list
function renderActiveDatasetList() {
    if (activeDatasets.length === 0) {
        activeDatasetList.innerHTML = '<p>No active datasets</p>';
        return;
    }
    
    let html = '';
    
    activeDatasets.forEach(datasetId => {
        const dataset = allDatasets.find(d => d.id === datasetId) || { id: datasetId };
        
        html += `
            <div class="dataset-item">
                <div class="dataset-item-info" title="${dataset.id}">
                    ${dataset.id.substring(0, 20)}${dataset.id.length > 20 ? '...' : ''}
                </div>
                <div class="dataset-item-actions">
                    <button onclick="toggleActiveDataset('${dataset.id}', 'remove')">Remove</button>
                </div>
            </div>
        `;
    });
    
    activeDatasetList.innerHTML = html;
}

// Toggle active dataset
async function toggleActiveDataset(datasetId, action) {
    try {
        const response = await fetch('/api/datasets/active', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: sessionId,
                action: action,
                dataset_id: datasetId
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to update active datasets');
        }
        
        // Update active datasets
        if (action === 'add') {
            activeDatasets.push(datasetId);
        } else {
            activeDatasets = activeDatasets.filter(id => id !== datasetId);
        }
        
        // Update UI
        renderDatasetList();
        renderActiveDatasetList();
    } catch (error) {
        console.error('Error updating active datasets:', error);
        alert('Error updating active datasets');
    }
}

// Show dataset info
async function showDatasetInfo(datasetId) {
    try {
        // Show loading
        datasetInfoTitle.textContent = 'Dataset Information';
        datasetInfo.innerHTML = '<p>Loading dataset information...</p>';
        datasetInfoModal.style.display = 'block';
        
        const response = await fetch(`/api/datasets/${datasetId}?session_id=${sessionId}`);
        
        if (!response.ok) {
            throw new Error('Failed to fetch dataset info');
        }
        
        const data = await response.json();
        
        // Update modal title
        datasetInfoTitle.textContent = `Dataset: ${datasetId}`;
        
        // Format dataset info
        let html = '<div class="dataset-info-section">';
        
        // Basic info
        html += '<h3>Basic Information</h3>';
        html += `<div class="dataset-info-item"><span class="dataset-info-label">Source Type:</span> ${data.info.source_type || 'Unknown'}</div>`;
        html += `<div class="dataset-info-item"><span class="dataset-info-label">Number of Rows:</span> ${data.info.num_rows || 'Unknown'}</div>`;
        html += `<div class="dataset-info-item"><span class="dataset-info-label">Has Embeddings:</span> ${data.info.has_embeddings ? 'Yes' : 'No'}</div>`;
        
        if (data.info.original_path) {
            html += `<div class="dataset-info-item"><span class="dataset-info-label">Original Path:</span> ${data.info.original_path}</div>`;
        }
        
        if (data.info.source_url) {
            html += `<div class="dataset-info-item"><span class="dataset-info-label">Source URL:</span> <a href="${data.info.source_url}" target="_blank">${data.info.source_url}</a></div>`;
        }
        
        html += '</div>';
        
        // Schema
        if (data.schema) {
            html += '<div class="dataset-info-section">';
            html += '<h3>Schema</h3>';
            
            if (data.schema.columns) {
                html += '<div class="dataset-info-item"><span class="dataset-info-label">Columns:</span></div>';
                html += '<ul>';
                data.schema.columns.forEach(column => {
                    const type = data.schema.dtypes && data.schema.dtypes[column] ? ` (${data.schema.dtypes[column]})` : '';
                    html += `<li>${column}${type}</li>`;
                });
                html += '</ul>';
            } else if (data.schema.tables) {
                html += '<div class="dataset-info-item"><span class="dataset-info-label">Tables:</span></div>';
                html += '<ul>';
                data.schema.tables.forEach(table => {
                    const rowCount = data.schema.table_rows && data.schema.table_rows[table] ? ` (${data.schema.table_rows[table]} rows)` : '';
                    html += `<li>${table}${rowCount}</li>`;
                });
                html += '</ul>';
            }
            
            html += '</div>';
        }
        
        // Sample data
        if (data.sample) {
            html += '<div class="dataset-info-section">';
            html += '<h3>Sample Data</h3>';
            
            if (Array.isArray(data.sample)) {
                // Simple dataset
                html += formatSampleTable(data.sample);
            } else {
                // Multi-component dataset
                for (const [name, sample] of Object.entries(data.sample)) {
                    html += `<h4>${name}</h4>`;
                    html += formatSampleTable(sample);
                }
            }
            
            html += '</div>';
        }
        
        datasetInfo.innerHTML = html;
    } catch (error) {
        console.error('Error fetching dataset info:', error);
        datasetInfo.innerHTML = `<p>Error fetching dataset information: ${error.message}</p>`;
    }
}

// Format sample data as table
function formatSampleTable(sample) {
    if (!sample || sample.length === 0) {
        return '<p>No sample data available</p>';
    }
    
    const columns = Object.keys(sample[0]);
    
    let html = '<div class="dataset-sample">';
    html += '<table>';
    
    // Table header
    html += '<tr>';
    columns.forEach(column => {
        html += `<th>${column}</th>`;
    });
    html += '</tr>';
    
    // Table rows
    sample.forEach(row => {
        html += '<tr>';
        columns.forEach(column => {
            const value = row[column];
            const displayValue = typeof value === 'object' ? JSON.stringify(value) : value;
            html += `<td>${displayValue}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</table>';
    html += '</div>';
    
    return html;
}

// Upload file
async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const fileType = document.getElementById('fileType').value;
    const delimiter = document.getElementById('delimiter').value;
    const sheetName = document.getElementById('sheetName').value;
    const chunkSize = document.getElementById('chunkSize').value;
    
    if (!fileInput.files || fileInput.files.length === 0) {
        alert('Please select a file');
        return;
    }
    
    const file = fileInput.files[0];
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    formData.append('file_type', fileType);
    formData.append('session_id', sessionId);
    
    if (fileType === 'csv' && delimiter) {
        formData.append('delimiter', delimiter);
    }
    
    if (fileType === 'excel' && sheetName) {
        formData.append('sheet_name', sheetName);
    }
    
    if (fileType === 'text' && chunkSize) {
        formData.append('chunk_size', chunkSize);
    }
    
    try {
        // Show loading
        uploadForm.innerHTML = '<p>Uploading and processing file... This may take a moment.</p>';
        
        const response = await fetch('/api/ingest', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Failed to upload file');
        }
        
        const data = await response.json();
        
        // Close modal
        uploadModal.style.display = 'none';
        
        // Reset form
        uploadForm.reset();
        uploadForm.innerHTML = `
            <div class="form-group">
                <label for="fileInput">Select File:</label>
                <input type="file" id="fileInput" required>
            </div>
            <div class="form-group">
                <label for="fileType">File Type:</label>
                <select id="fileType">
                    <option value="csv">CSV</option>
                    <option value="json">JSON</option>
                    <option value="excel">Excel</option>
                    <option value="text">Text</option>
                </select>
            </div>
            <div class="form-group" id="delimiterGroup">
                <label for="delimiter">Delimiter (for CSV):</label>
                <input type="text" id="delimiter" value="," maxlength="1">
            </div>
            <div class="form-group" id="sheetNameGroup" style="display: none;">
                <label for="sheetName">Sheet Name (for Excel, optional):</label>
                <input type="text" id="sheetName">
            </div>
            <div class="form-group" id="chunkSizeGroup" style="display: none;">
                <label for="chunkSize">Chunk Size (for Text):</label>
                <input type="number" id="chunkSize" value="1000" min="100">
            </div>
            <button type="submit">Upload</button>
        `;
        
        // Set up event listeners again
        fileTypeSelect.addEventListener('change', () => {
            const fileType = fileTypeSelect.value;
            
            // Show/hide relevant form groups
            delimiterGroup.style.display = fileType === 'csv' ? 'block' : 'none';
            sheetNameGroup.style.display = fileType === 'excel' ? 'block' : 'none';
            chunkSizeGroup.style.display = fileType === 'text' ? 'block' : 'none';
        });
        
        // Show success message
        alert(`File uploaded successfully. Dataset ID: ${data.dataset_id}`);
        
        // Reload datasets
        loadDatasets();
        loadActiveDatasets();
    } catch (error) {
        console.error('Error uploading file:', error);
        alert('Error uploading file: ' + error.message);
        
        // Reset form
        uploadForm.reset();
    }
}

// Add URL
async function addUrl() {
    const urlInput = document.getElementById('urlInput').value;
    
    if (!urlInput) {
        alert('Please enter a URL');
        return;
    }
    
    try {
        // Show loading
        urlForm.innerHTML = '<p>Processing URL... This may take a moment.</p>';
        
        const response = await fetch('/api/ingest/url', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                url: urlInput,
                session_id: sessionId
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to process URL');
        }
        
        const data = await response.json();
        
        // Close modal
        urlModal.style.display = 'none';
        
        // Reset form
        urlForm.reset();
        urlForm.innerHTML = `
            <div class="form-group">
                <label for="urlInput">URL:</label>
                <input type="url" id="urlInput" required placeholder="https://example.com">
            </div>
            <button type="submit">Add</button>
        `;
        
        // Show success message
        alert(`URL processed successfully. Dataset ID: ${data.dataset_id}`);
        
        // Reload datasets
        loadDatasets();
        loadActiveDatasets();
    } catch (error) {
        console.error('Error processing URL:', error);
        alert('Error processing URL: ' + error.message);
        
        // Reset form
        urlForm.reset();
    }
}

// Send message
async function sendMessage() {
    const message = messageInput.value.trim();
    
    if (!message) {
        return;
    }
    
    // Add user message to chat
    addUserMessage(message);
    
    // Clear input
    messageInput.value = '';
    
    // Add loading message
    const loadingElement = document.createElement('div');
    loadingElement.className = 'assistant-message-container';
    loadingElement.innerHTML = `
        <div class="assistant-message">
            <div class="loading"></div> Thinking...
        </div>
    `;
    chatMessages.appendChild(loadingElement);
    
    // Scroll to bottom
    scrollToBottom();
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                session_id: sessionId,
                use_rag: useRagCheckbox.checked,
                retrieve_external: retrieveExternalCheckbox.checked
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to send message');
        }
        
        const data = await response.json();
        
        // Remove loading message
        chatMessages.removeChild(loadingElement);
        
        // Add assistant message to chat
        addAssistantMessage(data.response, data.metadata);
        
        // Scroll to bottom
        scrollToBottom();
    } catch (error) {
        console.error('Error sending message:', error);
        
        // Remove loading message
        chatMessages.removeChild(loadingElement);
        
        // Add error message
        addAssistantMessage(`Error: ${error.message}`, { error: true });
        
        // Scroll to bottom
        scrollToBottom();
    }
}

// Add user message to chat
function addUserMessage(message) {
    const messageElement = document.createElement('div');
    messageElement.className = 'message user-message';
    messageElement.textContent = message;
    
    chatMessages.appendChild(messageElement);
    
    // Scroll to bottom
    scrollToBottom();
}

// Add assistant message to chat
function addAssistantMessage(message, metadata) {
    const messageContainer = document.createElement('div');
    messageContainer.className = 'assistant-message-container';
    
    const messageElement = document.createElement('div');
    messageElement.className = 'assistant-message';
    messageElement.textContent = message;
    
    messageContainer.appendChild(messageElement);
    
    // Add metadata if available
    if (metadata) {
        let metadataText = '';
        
        if (metadata.model_used) {
            metadataText += `Model: ${metadata.model_used} `;
        }
        
        if (metadata.fallback_used) {
            metadataText += 'Fallback model used ';
        }
        
        if (metadata.rag_used) {
            metadataText += 'RAG augmentation used ';
        }
        
        if (metadata.external_retrieved) {
            metadataText += 'External data retrieved ';
        }
        
        if (metadata.error) {
            metadataText += 'Error occurred ';
        }
        
        if (metadataText) {
            const metadataElement = document.createElement('div');
            metadataElement.className = 'message-metadata';
            metadataElement.textContent = metadataText;
            
            messageContainer.appendChild(metadataElement);
        }
    }
    
    chatMessages.appendChild(messageContainer);
    
    // Scroll to bottom
    scrollToBottom();
}

// Clear history
async function clearHistory() {
    if (!confirm('Are you sure you want to clear the conversation history?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/history?session_id=${sessionId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error('Failed to clear history');
        }
        
        // Clear chat messages
        chatMessages.innerHTML = `
            <div class="welcome-message">
                <h2>Welcome to Data AI Chatbot</h2>
                <p>Your interactive data analysis assistant</p>
                <p>Upload data or add URLs to get started, then ask questions about your data.</p>
            </div>
        `;
    } catch (error) {
        console.error('Error clearing history:', error);
        alert('Error clearing history: ' + error.message);
    }
}

// Scroll to bottom of chat
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Close modals when pressing Escape
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        uploadModal.style.display = 'none';
        urlModal.style.display = 'none';
        datasetInfoModal.style.display = 'none';
    }
});

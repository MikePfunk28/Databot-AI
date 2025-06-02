#!/bin/bash
set -e

echo "Starting DataBot AI System..."

# Start Ollama server in background
echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
sleep 10

# Function to check if Ollama is ready
wait_for_ollama() {
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "Ollama is ready!"
            return 0
        fi
        echo "Attempt $attempt/$max_attempts: Waiting for Ollama..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "Failed to connect to Ollama after $max_attempts attempts"
    return 1
}

wait_for_ollama

# Pull required models if not present
echo "Checking for required models..."

# Check and pull phi4-mini-reasoning
if ! ollama list | grep -q "phi4-mini-reasoning"; then
    echo "Pulling phi4-mini-reasoning model..."
    ollama pull phi4-mini-reasoning
fi

# Check and pull databot-embed
if ! ollama list | grep -q "mikepfunk28/databot-embed"; then
    echo "Pulling databot-embed model..."
    ollama pull mikepfunk28/databot-embed
fi

# Create databot-instruct from Modelfile if not present
if ! ollama list | grep -q "databot-instruct"; then
    echo "Creating databot-instruct model from Modelfile..."
    if [ -f "/app/instruct/Modelfile-databot-instruct" ]; then
        cd /app/instruct
        ollama create databot-instruct -f Modelfile-databot-instruct
        cd /app
    else
        echo "Warning: Modelfile not found, databot-instruct model not created"
    fi
fi

# Verify models are available
echo "Available models:"
ollama list

# Run setup script if it exists
if [ -f "/app/setup_databot.py" ]; then
    echo "Running DataBot setup..."
    python setup_databot.py --skip-ollama
fi

# Create necessary directories
mkdir -p /app/data/{datasets,embeddings,conversations,workflows,learning}
mkdir -p /app/logs

echo "DataBot AI System initialization complete!"

# Execute the main command
exec "$@"
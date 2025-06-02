# DataBot AI - Comprehensive Data Analysis System

A sophisticated AI-powered data analysis system built with Phi4-mini-reasoning and enhanced with advanced workflow management, vector embeddings, and intelligent data insights generation.

## Features

- **Local Model Inference**: Uses phi4-mini-reasoning via Ollama for privacy and control
- **Data Analysis**: Analyze CSV, JSON, Excel, and text files
- **RAG (Retrieval-Augmented Generation)**: Enhances responses with context from your data
- **Vector Embeddings**: Efficiently stores and retrieves information (with fallbacks for limited resources)
- **Web Interface**: User-friendly chat interface with data visualization
- **API Access**: REST API for integration with other applications
- **Continuous Learning**: Learns from your data to provide better insights over time

## Installation

### Prerequisites

- Python 3.8+ 
- Ollama (https://ollama.com/download)

### Installation Steps

#### Linux/macOS

1. Extract the package:

   ```bash
   tar -xzvf phi4-data-ai-chatbot.tar.gz -C /your/target/directory
   cd /your/target/directory
   ```
1. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
1. Install Ollama:

   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
1. Pull the phi4-mini-reasoning model:

   ```bash
   ollama pull phi4-mini-reasoning:latest
   ```

#### Windows

1. Extract the package to your desired location
1. Create a virtual environment:

   ```
   python -m venv venv
   venv\Scripts\activate
   ```
1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```
1. Install Ollama from https://ollama.com/download
1. Pull the phi4-mini-reasoning model:

   ```
   ollama pull phi4-mini-reasoning:latest
   ```

### Optional Dependencies

For production use with large datasets, install these optional dependencies:

```bash
pip install sentence-transformers faiss-cpu
```

## Usage

### Starting the Application

```bash
python src/main.py
```

Then open your browser to http://localhost:5000

### Using the Web Interface

1. **Upload Data**: Use the upload button to add CSV, JSON, Excel, or text files
2. **Ask Questions**: Type questions about your data in the chat interface
3. **View Insights**: The AI will analyze your data and provide insights
4. **Visualize Data**: Request visualizations by asking for charts or graphs

### API Access

The application provides a REST API for integration:

- `POST /api/chat`: Send messages to the AI
- `POST /api/ingest`: Upload data files
- `GET /api/datasets`: List available datasets
- `GET /api/datasets/<id>`: Get information about a specific dataset

## Application Structure

```
data-ai-chatbot/
├── config/                 # Configuration files
│   └── system_prompt.md    # System prompt for the AI
├── data/                   # Data storage (created on first run)
│   ├── datasets/           # Ingested datasets
│   ├── embeddings/         # Vector embeddings
│   └── conversations/      # Saved conversations
├── src/                    # Source code
│   ├── models/             # Database models
│   ├── routes/             # API routes
│   ├── static/             # Static assets (CSS, JS)
│   ├── templates/          # HTML templates
│   ├── data_ingestion.py   # Data ingestion module
│   ├── model_wrapper.py    # Ollama model wrapper
│   ├── rag_pipeline.py     # RAG implementation
│   ├── vector_embedding.py # Vector embedding module
│   └── main.py             # Main application entry point
├── venv/                   # Virtual environment
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Customization

### Changing the System Prompt

Edit `config/system_prompt.md` to customize how the AI responds to queries.

### Using a Different Model

To use a different Ollama model, edit `src/main.py` and change:

```python
model_wrapper = ModelWrapper(
    model_name="your-model-name:tag",
    system_prompt_path=SYSTEM_PROMPT_PATH
)
```

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port in `src/main.py` (line with `app.run`)
2. **Ollama not running**: Start Ollama with `ollama serve`
3. **Model not found**: Ensure you've pulled the model with `ollama pull phi4-mini-reasoning:latest`

### Resource Usage

- The application is designed to work with limited resources
- For large datasets, consider installing the optional dependencies
- Adjust chunk sizes in data ingestion for memory optimization

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Uses Ollama for local model inference
- Built with Flask, pandas, and other open-source libraries
- phi4-mini-reasoning model developed by Microsoft
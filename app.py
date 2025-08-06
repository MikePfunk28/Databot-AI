import os
import json
import pandas as pd
import sqlite3
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from datetime import datetime
import logging

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Configuration
UPLOAD_FOLDER = 'data/uploads'
ALLOWED_EXTENSIONS = {'csv', 'json', 'xlsx', 'xls', 'txt', 'sql'}
OLLAMA_URL = 'http://localhost:11434'
MODEL_NAME = 'databot'

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('data/embeddings', exist_ok=True)

# Initialize sentence transformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def query_ollama(prompt, context=""):
    try:
        full_prompt = f"Context: {context}\n\nUser Question: {prompt}" if context else prompt

        response = requests.post(f"{OLLAMA_URL}/api/generate",
                                 json={
            "model": MODEL_NAME,
            "prompt": full_prompt,
            "stream": False
        })

        if response.status_code == 200:
            return response.json().get('response', 'No response generated')
        else:
            return f"Error: Ollama service returned status {response.status_code}"

    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama service. Please ensure Ollama is running."
    except Exception as e:
        return f"Error querying Ollama: {str(e)}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Process the uploaded file
            data_summary = process_uploaded_file(filepath)
            session['current_file'] = filepath

            return jsonify({
                'success': True,
                'filename': filename,
                'summary': data_summary
            })
        else:
            return jsonify({'error': 'Invalid file type'}), 400

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


def process_uploaded_file(filepath):
    try:
        file_ext = filepath.split('.')[-1].lower()

        if file_ext == 'csv':
            df = pd.read_csv(filepath)
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(filepath)
        elif file_ext == 'json':
            df = pd.read_json(filepath)
        else:
            return "Unsupported file type"

        # Generate summary
        summary = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'data_types': df.dtypes.to_dict(),
            'head': df.head().to_dict(),
            'missing_values': df.isnull().sum().to_dict()
        }

        # Create embeddings for RAG
        create_embeddings(df, filepath)

        return summary

    except Exception as e:
        logger.error(f"File processing error: {str(e)}")
        return f"Error processing file: {str(e)}"


def create_embeddings(df, filepath):
    try:
        # Convert dataframe to text chunks for embedding
        text_chunks = []

        # Add column information
        col_info = f"Dataset has {len(df.columns)} columns: {', '.join(df.columns)}"
        text_chunks.append(col_info)

        # Add sample rows as text
        for idx, row in df.head(10).iterrows():
            row_text = f"Row {idx}: " + \
                ", ".join([f"{col}: {val}" for col, val in row.items()])
            text_chunks.append(row_text)

        # Add statistical summary
        stats_text = f"Dataset statistics: {len(df)} rows, data types: {df.dtypes.to_dict()}"
        text_chunks.append(stats_text)

        # Generate embeddings
        embeddings = embedding_model.encode(text_chunks)

        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))

        # Save index and chunks
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        index_path = f'data/embeddings/{base_name}_index.faiss'
        chunks_path = f'data/embeddings/{base_name}_chunks.pkl'

        faiss.write_index(index, index_path)
        with open(chunks_path, 'wb') as f:
            pickle.dump(text_chunks, f)

        logger.info(f"Created embeddings for {filepath}")

    except Exception as e:
        logger.error(f"Embedding creation error: {str(e)}")


@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Get context from current file if available
        context = ""
        if 'current_file' in session:
            context = get_relevant_context(
                user_message, session['current_file'])

        # Query Ollama with context
        response = query_ollama(user_message, context)

        return jsonify({
            'response': response,
            'context_used': bool(context)
        })

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500


def get_relevant_context(query, filepath):
    try:
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        index_path = f'data/embeddings/{base_name}_index.faiss'
        chunks_path = f'data/embeddings/{base_name}_chunks.pkl'

        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            return ""

        # Load index and chunks
        index = faiss.read_index(index_path)
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)

        # Encode query and search
        query_embedding = embedding_model.encode([query])
        D, I = index.search(query_embedding.astype('float32'), k=3)

        # Get relevant chunks
        relevant_chunks = [chunks[i] for i in I[0] if i < len(chunks)]

        return "\n".join(relevant_chunks)

    except Exception as e:
        logger.error(f"Context retrieval error: {str(e)}")
        return ""


@app.route('/url', methods=['POST'])
def process_url():
    try:
        url = request.json.get('url', '')

        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        # Fetch data from URL
        response = requests.get(url)
        response.raise_for_status()

        # Try to parse as JSON first, then CSV
        try:
            data = response.json()
            df = pd.json_normalize(data)
        except:
            try:
                df = pd.read_csv(url)
            except:
                return jsonify({'error': 'Unable to parse URL data as JSON or CSV'}), 400

        # Process the data
        summary = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'head': df.head().to_dict()
        }

        # Save temporary file and create embeddings
        temp_file = f"data/uploads/url_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(temp_file, index=False)
        create_embeddings(df, temp_file)
        session['current_file'] = temp_file

        return jsonify({
            'success': True,
            'summary': summary
        })

    except Exception as e:
        logger.error(f"URL processing error: {str(e)}")
        return jsonify({'error': f'URL processing failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# Generated by Copilot

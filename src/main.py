#!/usr/bin/env python3
"""
Main Flask application for Data AI Chatbot
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # DON'T CHANGE THIS !!!

import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import uuid
import logging

# Import chatbot components
from src.model_wrapper import ModelWrapper
from src.data_ingestion import DataIngestion
from src.rag_pipeline import RAGPipeline
from src.vector_embedding import VectorEmbedding, SENTENCE_TRANSFORMERS_AVAILABLE, FAISS_AVAILABLE
from src.workflow_manager import WorkflowManager
from src.analysis_learning import AnalysisLearningSystem

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CONFIG_DIR = os.path.join(BASE_DIR, 'config')

# Ensure directories exist
os.makedirs(os.path.join(DATA_DIR, 'datasets'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'embeddings'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'conversations'), exist_ok=True)

# Load system prompt
SYSTEM_PROMPT_PATH = os.path.join(CONFIG_DIR, 'system_prompt.md')
if not os.path.exists(SYSTEM_PROMPT_PATH):
    with open(SYSTEM_PROMPT_PATH, 'w') as f:
        f.write("""You are a Data Analysis AI Assistant. Your purpose is to help users analyze and understand their data.

Your capabilities:
1. Learn from the user's data to provide insights
2. Answer questions about the data
3. Perform complex data analysis
4. Generate visualizations when needed
5. Retrieve external information to supplement the data
6. Continuously improve your understanding of the data

When analyzing data:
- Be thorough and detailed in your analysis
- Identify patterns, trends, and anomalies
- Provide context and explanations for your findings
- Suggest follow-up questions or analyses
- Be honest about limitations or uncertainties

Always maintain a helpful, informative tone and focus on providing value through data-driven insights.""")

# Initialize components with DataBot models
model_wrapper = ModelWrapper(
    model_name="databot-instruct",
    system_prompt_path=SYSTEM_PROMPT_PATH
)

data_ingestion = DataIngestion(data_dir=DATA_DIR)
vector_embedding = VectorEmbedding(data_dir=DATA_DIR, model_name="mikepfunk28/databot-embed")
rag_pipeline = RAGPipeline(
    data_dir=DATA_DIR, vector_embedding=vector_embedding)

# Initialize analysis learning system
analysis_learning = AnalysisLearningSystem(data_dir=DATA_DIR)

# Initialize workflow manager
workflow_components = {
    'model_wrapper': model_wrapper,
    'data_ingestion': data_ingestion,
    'vector_embedding': vector_embedding,
    'rag_pipeline': rag_pipeline,
    'analysis_learning': analysis_learning
}
workflow_manager = WorkflowManager(data_dir=DATA_DIR, components=workflow_components)

# Session management
active_sessions = {}


def get_or_create_session(session_id=None):
    """Get or create a session"""
    if not session_id:
        session_id = str(uuid.uuid4())

    if session_id not in active_sessions:
        active_sessions[session_id] = {
            'history': [],
            'active_datasets': []
        }

    return session_id, active_sessions[session_id]

# Routes


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Process a chat message"""
    try:
        data = request.json
        message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        use_rag = data.get('use_rag', True)
        retrieve_external = data.get('retrieve_external', True)

        # Get or create session
        session_id, session = get_or_create_session(session_id)

        # Add user message to history
        session['history'].append({
            'role': 'user',
            'content': message
        })

        # Process message
        context = ""
        metadata = {}

        # Apply RAG if enabled
        if use_rag and session['active_datasets']:
            context = rag_pipeline.retrieve_context(
                query=message,
                dataset_ids=session['active_datasets'],
                retrieve_external=retrieve_external
            )
            metadata['rag_used'] = True
            if retrieve_external and rag_pipeline.external_data_retrieved:
                # Generate response
                metadata['external_retrieved'] = True
        response = model_wrapper.generate_response(
            prompt=message,
            context=context
        )

        metadata['model_used'] = model_wrapper.model_name

        # Add assistant message to history
        session['history'].append({
            'role': 'assistant',
            'content': response,
            'metadata': metadata
        })

        # Save conversation history
        save_conversation_history(session_id, session['history'])

        return jsonify({
            'response': response,
            'metadata': metadata
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'response': f"An error occurred: {str(e)}",
            'metadata': {'error': True}
        }), 500


@app.route('/api/ingest', methods=['POST'])
def ingest_file():
    """Ingest a file"""
    try:
        session_id = request.form.get('session_id', 'default')
        file_type = request.form.get('file_type', 'csv')

        # Get file
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join('/tmp', filename)
        file.save(temp_path)

        # Process file based on type
        if file_type == 'csv':
            delimiter = request.form.get('delimiter', ',')
            dataset_id = data_ingestion.ingest_csv(
                temp_path, delimiter=delimiter)

        elif file_type == 'json':
            dataset_id = data_ingestion.ingest_json(temp_path)

        elif file_type == 'excel':
            sheet_name = request.form.get('sheet_name', None)
            dataset_id = data_ingestion.ingest_excel(
                temp_path, sheet_name=sheet_name)

        elif file_type == 'text':
            chunk_size = int(request.form.get('chunk_size', 1000))
            dataset_id = data_ingestion.ingest_text(
                temp_path, chunk_size=chunk_size)

        else:
            os.remove(temp_path)
            return jsonify({'error': f'Unsupported file type: {file_type}'}), 400

        # Clean up
        os.remove(temp_path)

        # Generate embeddings
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            vector_embedding.generate_embeddings(dataset_id)

        # Add to active datasets
        session_id, session = get_or_create_session(session_id)
        if dataset_id not in session['active_datasets']:
            session['active_datasets'].append(dataset_id)

        return jsonify({
            'dataset_id': dataset_id,
            'message': f'File ingested successfully as {dataset_id}'
        })

    except Exception as e:
        logger.error(f"Error in ingest endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/ingest/url', methods=['POST'])
def ingest_url():
    """Ingest a URL"""
    try:
        data = request.json
        url = data.get('url', '')
        session_id = data.get('session_id', 'default')

        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        # Process URL
        dataset_id = data_ingestion.ingest_url(url)

        # Generate embeddings
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            vector_embedding.generate_embeddings(dataset_id)

        # Add to active datasets
        session_id, session = get_or_create_session(session_id)
        if dataset_id not in session['active_datasets']:
            session['active_datasets'].append(dataset_id)

        return jsonify({
            'dataset_id': dataset_id,
            'message': f'URL ingested successfully as {dataset_id}'
        })

    except Exception as e:
        logger.error(f"Error in ingest URL endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Get all datasets"""
    try:
        datasets = data_ingestion.list_datasets()
        return jsonify({'datasets': datasets})

    except Exception as e:
        logger.error(
            f"Error in get datasets endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/active', methods=['GET'])
def get_active_datasets():
    """Get active datasets for a session"""
    try:
        session_id = request.args.get('session_id', 'default')
        session_id, session = get_or_create_session(session_id)

        return jsonify({'active_datasets': session['active_datasets']})

    except Exception as e:
        logger.error(
            f"Error in get active datasets endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/active', methods=['POST'])
def update_active_datasets():
    """Update active datasets for a session"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        action = data.get('action', 'add')
        dataset_id = data.get('dataset_id', '')

        if not dataset_id:
            return jsonify({'error': 'No dataset ID provided'}), 400

        # Get or create session
        session_id, session = get_or_create_session(session_id)

        # Update active datasets
        if action == 'add':
            if dataset_id not in session['active_datasets']:
                session['active_datasets'].append(dataset_id)

        elif action == 'remove':
            if dataset_id in session['active_datasets']:
                session['active_datasets'].remove(dataset_id)

        else:
            return jsonify({'error': f'Unsupported action: {action}'}), 400

        return jsonify({'active_datasets': session['active_datasets']})

    except Exception as e:
        logger.error(
            f"Error in update active datasets endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/<dataset_id>', methods=['GET'])
def get_dataset_info(dataset_id):
    """Get information about a dataset"""
    try:
        # Get dataset info
        info = data_ingestion.get_dataset_info(dataset_id)

        # Get dataset schema
        schema = data_ingestion.get_dataset_schema(dataset_id)

        # Get dataset sample
        sample = data_ingestion.get_dataset_sample(dataset_id, max_rows=5)

        return jsonify({
            'info': info,
            'schema': schema,
            'sample': sample
        })

    except Exception as e:
        logger.error(
            f"Error in get dataset info endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history for a session"""
    try:
        session_id = request.args.get('session_id', 'default')
        session_id, session = get_or_create_session(session_id)

        return jsonify({'history': session['history']})

    except Exception as e:
        logger.error(f"Error in get history endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['DELETE'])
def clear_history():
    """Clear conversation history for a session"""
    try:
        session_id = request.args.get('session_id', 'default')
        session_id, session = get_or_create_session(session_id)

        session['history'] = []

        # Save empty conversation history
        save_conversation_history(session_id, [])

        return jsonify({'message': 'History cleared successfully'})

    except Exception as e:
        logger.error(
            f"Error in clear history endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


def save_conversation_history(session_id, history):
    """Save conversation history to file"""
    try:
        history_path = os.path.join(
            DATA_DIR, 'conversations', f'{session_id}.json')

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    except Exception as e:
        logger.error(
            f"Error saving conversation history: {str(e)}", exc_info=True)


def load_conversation_history(session_id):
    """Load conversation history from file"""
    try:
        history_path = os.path.join(
            DATA_DIR, 'conversations', f'{session_id}.json')

        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                return json.load(f)

    except Exception as e:
        logger.error(
            f"Error loading conversation history: {str(e)}", exc_info=True)

    return []

# Load existing sessions


def load_existing_sessions():
    """Load existing sessions from conversation history files"""
    try:
        conversations_dir = os.path.join(DATA_DIR, 'conversations')

        if not os.path.exists(conversations_dir):
            return

        for filename in os.listdir(conversations_dir):
            if filename.endswith('.json'):
                session_id = filename[:-5]  # Remove .json extension
                history = load_conversation_history(session_id)

                if history:
                    active_sessions[session_id] = {
                        'history': history,
                        'active_datasets': []  # Active datasets will be empty on load
                    }

    except Exception as e:
        logger.error(
            f"Error loading existing sessions: {str(e)}", exc_info=True)


# Initialize sessions
load_existing_sessions()


# Workflow API endpoints
@app.route('/api/workflows', methods=['GET'])
def list_workflows():
    """List all workflows"""
    try:
        workflows = workflow_manager.list_workflows()
        return jsonify({
            'success': True,
            'workflows': workflows
        })
    except Exception as e:
        logger.error(f"Error listing workflows: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/workflows', methods=['POST'])
def create_workflow():
    """Create a new workflow"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        name = data.get('name')
        description = data.get('description', '')
        tasks = data.get('tasks', [])
        
        if not name or not tasks:
            return jsonify({
                'success': False,
                'error': 'Name and tasks are required'
            }), 400
        
        workflow_id = workflow_manager.create_workflow(name, description, tasks)
        
        return jsonify({
            'success': True,
            'workflow_id': workflow_id
        })
        
    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/workflows/<workflow_id>/execute', methods=['POST'])
def execute_workflow(workflow_id):
    """Execute a workflow"""
    try:
        success = workflow_manager.execute_workflow(workflow_id)
        
        return jsonify({
            'success': success,
            'workflow_id': workflow_id
        })
        
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/workflows/<workflow_id>/status', methods=['GET'])
def get_workflow_status(workflow_id):
    """Get workflow status"""
    try:
        status = workflow_manager.get_workflow_status(workflow_id)
        
        if status is None:
            return jsonify({
                'success': False,
                'error': 'Workflow not found'
            }), 404
        
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        logger.error(f"Error getting workflow status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/workflows/analyze', methods=['POST'])
def create_analysis_workflow():
    """Create a predefined data analysis workflow"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        dataset_path = data.get('dataset_path')
        analysis_type = data.get('analysis_type', 'comprehensive')
        
        if not dataset_path:
            return jsonify({
                'success': False,
                'error': 'dataset_path is required'
            }), 400
        
        workflow_id = workflow_manager.create_data_analysis_workflow(
            dataset_path, analysis_type
        )
        
        # Optionally auto-execute the workflow
        if data.get('auto_execute', False):
            workflow_manager.execute_workflow(workflow_id)
        
        return jsonify({
            'success': True,
            'workflow_id': workflow_id
        })
        
    except Exception as e:
        logger.error(f"Error creating analysis workflow: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/learning/insights', methods=['GET'])
def get_learning_insights():
    """Get insights about what the system has learned"""
    try:
        insights = analysis_learning.get_learning_insights()
        return jsonify({
            'success': True,
            'insights': insights
        })
    except Exception as e:
        logger.error(f"Error getting learning insights: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/learning/recommend', methods=['POST'])
def get_analysis_recommendations():
    """Get analysis recommendations for a dataset"""
    try:
        data = request.get_json()
        dataset_characteristics = data.get('dataset_characteristics', {})
        
        if not dataset_characteristics:
            return jsonify({
                'success': False,
                'error': 'Dataset characteristics required'
            }), 400
        
        recommendations = analysis_learning.optimize_for_dataset(dataset_characteristics)
        method, confidence = analysis_learning.get_recommended_method(dataset_characteristics)
        
        return jsonify({
            'success': True,
            'recommended_method': method,
            'confidence': confidence,
            'detailed_recommendations': recommendations
        })
        
    except Exception as e:
        logger.error(f"Error getting analysis recommendations: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/learning/validate', methods=['POST'])
def validate_insight():
    """Validate an insight with actual outcomes"""
    try:
        data = request.get_json()
        insight_id = data.get('insight_id')
        validation_score = data.get('validation_score')
        impact_score = data.get('impact_score')
        
        if not insight_id or validation_score is None:
            return jsonify({
                'success': False,
                'error': 'insight_id and validation_score required'
            }), 400
        
        analysis_learning.validate_insight(insight_id, validation_score, impact_score)
        
        return jsonify({
            'success': True,
            'message': 'Insight validated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error validating insight: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for Docker"""
    try:
        # Check if all components are working
        status = {
            'status': 'healthy',
            'components': {
                'model_wrapper': model_wrapper is not None,
                'vector_embedding': vector_embedding is not None,
                'analysis_learning': analysis_learning is not None,
                'workflow_manager': workflow_manager is not None
            },
            'models': {
                'databot_instruct': 'databot-instruct' in str(model_wrapper.model_name) if model_wrapper else False,
                'databot_embed': 'databot-embed' in str(vector_embedding.model_name) if vector_embedding else False
            }
        }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

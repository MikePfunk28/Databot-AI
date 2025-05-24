#!/usr/bin/env python3
"""
Chatbot Interface for Data AI Chatbot

This module provides a user-friendly interface for interacting with the data
analysis chatbot, integrating the model wrapper and RAG pipeline.
"""

from src.rag_pipeline import RAGPipeline
from src.data_ingestion import DataIngestion, VectorEmbedding
from src.model_wrapper import ModelWrapper, GeminiWrapper, ModelManager
import os
import json
import time
import argparse
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import threading
import queue
import cmd
import readline
import shlex
import sys
import re
import textwrap

# Import local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ChatSession:
    """
    Class for managing a chat session with the data analysis chatbot.
    """

    def __init__(
        self,
        model_name: str = "databot",
        system_prompt_path: str = "/home/ubuntu/data-ai-chatbot/config/system_prompt.md",
        data_dir: str = "/home/ubuntu/data-ai-chatbot/data",
        gemini_api_key: Optional[str] = None,
        conversation_id: Optional[str] = None
    ):
        """
        Initialize the chat session.

        Args:
            model_name: Name of the Ollama model to use
            system_prompt_path: Path to the system prompt file
            data_dir: Directory containing data
            gemini_api_key: Gemini API key for fallback
            conversation_id: Unique identifier for the conversation
        """
        self.model_name = model_name
        self.system_prompt_path = system_prompt_path
        self.data_dir = data_dir
        self.gemini_api_key = gemini_api_key

        # Generate conversation ID if not provided
        if conversation_id is None:
            self.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        else:
            self.conversation_id = conversation_id

        # Initialize components
        self._init_components()

        # Create conversation directory
        self.conversation_dir = os.path.join(
            data_dir, "conversations", self.conversation_id)
        os.makedirs(self.conversation_dir, exist_ok=True)

        # Initialize conversation history
        self.history_path = os.path.join(self.conversation_dir, "history.json")
        self._init_history()

        # Initialize active datasets
        self.active_datasets = []

    def _init_components(self):
        """Initialize the model, data ingestion, and RAG components."""
        try:
            # Initialize primary model
            self.primary_model = ModelWrapper(
                model_name=self.model_name,
                system_prompt_path=self.system_prompt_path
            )

            # Initialize fallback model if API key is provided
            self.fallback_model = None
            if self.gemini_api_key:
                self.fallback_model = GeminiWrapper(
                    api_key=self.gemini_api_key)

            # Initialize model manager
            self.model_manager = ModelManager(
                primary_model=self.primary_model,
                fallback_model=self.fallback_model
            )

            # Initialize data ingestion and vector embedding
            self.data_ingestion = DataIngestion(self.data_dir)
            self.vector_embedding = VectorEmbedding(self.data_dir)

            # Initialize RAG pipeline
            self.rag_pipeline = RAGPipeline(self.data_dir)

        except Exception as e:
            print(f"Error initializing components: {e}")
            raise

    def _init_history(self):
        """Initialize the conversation history."""
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r') as f:
                    self.history = json.load(f)
            except Exception as e:
                print(f"Error loading conversation history: {e}")
                self.history = {"messages": []}
        else:
            self.history = {"messages": []}

    def _save_history(self):
        """Save the conversation history."""
        try:
            with open(self.history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Error saving conversation history: {e}")

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a message to the conversation history.

        Args:
            role: Role of the message sender (user or assistant)
            content: Message content
            metadata: Additional metadata
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        self.history["messages"].append(message)
        self._save_history()

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the conversation history.

        Args:
            limit: Maximum number of messages to return

        Returns:
            List of messages
        """
        messages = self.history["messages"]

        if limit is not None:
            messages = messages[-limit:]

        return messages

    def clear_history(self):
        """Clear the conversation history."""
        self.history = {"messages": []}
        self._save_history()

        # Also clear model conversation history
        self.primary_model.clear_conversation()

    def process_message(
        self,
        message: str,
        use_rag: bool = True,
        retrieve_external: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user message and generate a response.

        Args:
            message: User message
            use_rag: Whether to use RAG for context augmentation
            retrieve_external: Whether to retrieve external data

        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Add user message to history
            self.add_message("user", message)

            # Process with RAG if enabled
            context = None
            rag_results = None

            if use_rag:
                rag_results = self.rag_pipeline.process_query(
                    message,
                    use_existing_data=True,
                    retrieve_external=retrieve_external,
                    dataset_ids=self.active_datasets if self.active_datasets else None
                )

                context = rag_results.get("formatted_context")

            # Generate response
            response_data = self.model_manager.generate_response(
                message, context)

            # Add assistant message to history
            metadata = {
                "model_used": response_data.get("model_used", "unknown"),
                "fallback_used": response_data.get("fallback_used", False),
                "rag_used": use_rag,
                "external_retrieved": rag_results.get("external_retrieved", False) if rag_results else False,
                "sources": rag_results.get("sources", []) if rag_results else []
            }

            self.add_message("assistant", response_data["response"], metadata)

            return {
                "response": response_data["response"],
                "metadata": metadata
            }

        except Exception as e:
            error_message = f"Error processing message: {str(e)}"
            print(error_message)

            # Add error message to history
            self.add_message(
                "assistant",
                error_message,
                {"error": True}
            )

            return {
                "response": error_message,
                "metadata": {"error": True}
            }

    def analyze_data(
        self,
        message: str,
        dataset_id: str
    ) -> Dict[str, Any]:
        """
        Analyze data based on a user message.

        Args:
            message: User message
            dataset_id: Dataset ID to analyze

        Returns:
            Dictionary containing the analysis and metadata
        """
        try:
            # Get dataset info
            dataset_info = self.data_ingestion.get_dataset_info(dataset_id)

            if not dataset_info:
                raise ValueError(f"Dataset '{dataset_id}' not found")

            # Get dataset schema
            schema = self.data_ingestion.get_dataset_schema(dataset_id)

            # Get dataset sample
            sample = self.data_ingestion.get_dataset_sample(
                dataset_id, num_rows=5)

            # Format data context
            data_context = f"""
Dataset ID: {dataset_id}
Source Type: {dataset_info.get('source_type', 'unknown')}
Number of Rows: {dataset_info.get('num_rows', 'unknown')}

Schema:
{json.dumps(schema, indent=2)}

Sample Data:
{json.dumps(sample, indent=2)}
"""

            # Add user message to history
            self.add_message(
                "user",
                message,
                {"dataset_id": dataset_id}
            )

            # Generate analysis
            response_data = self.model_manager.analyze_data(
                message, data_context)

            # Add assistant message to history
            metadata = {
                "model_used": response_data.get("model_used", "unknown"),
                "fallback_used": response_data.get("fallback_used", False),
                "dataset_id": dataset_id
            }

            self.add_message("assistant", response_data["response"], metadata)

            return {
                "response": response_data["response"],
                "metadata": metadata
            }

        except Exception as e:
            error_message = f"Error analyzing data: {str(e)}"
            print(error_message)

            # Add error message to history
            self.add_message(
                "assistant",
                error_message,
                {"error": True}
            )

            return {
                "response": error_message,
                "metadata": {"error": True}
            }

    def set_active_datasets(self, dataset_ids: List[str]):
        """
        Set the active datasets for RAG.

        Args:
            dataset_ids: List of dataset IDs
        """
        self.active_datasets = dataset_ids

    def add_active_dataset(self, dataset_id: str):
        """
        Add a dataset to the active datasets.

        Args:
            dataset_id: Dataset ID to add
        """
        if dataset_id not in self.active_datasets:
            self.active_datasets.append(dataset_id)

    def remove_active_dataset(self, dataset_id: str):
        """
        Remove a dataset from the active datasets.

        Args:
            dataset_id: Dataset ID to remove
        """
        if dataset_id in self.active_datasets:
            self.active_datasets.remove(dataset_id)

    def get_active_datasets(self) -> List[str]:
        """
        Get the active datasets.

        Returns:
            List of active dataset IDs
        """
        return self.active_datasets


class ChatbotCLI(cmd.Cmd):
    """
    Command-line interface for the data analysis chatbot.
    """

    intro = """
=================================================================
Data AI Chatbot - Your Interactive Data Analysis Assistant
=================================================================
Type 'help' or '?' to list commands.
Type 'chat <message>' to start a conversation.
Type 'exit' to quit.
"""
    prompt = "DataChat> "

    def __init__(
        self,
        model_name: str = "databot",
        system_prompt_path: str = "/home/ubuntu/data-ai-chatbot/config/system_prompt.md",
        data_dir: str = "/home/ubuntu/data-ai-chatbot/data",
        gemini_api_key: Optional[str] = None
    ):
        """
        Initialize the chatbot CLI.

        Args:
            model_name: Name of the Ollama model to use
            system_prompt_path: Path to the system prompt file
            data_dir: Directory containing data
            gemini_api_key: Gemini API key for fallback
        """
        super().__init__()

        # Initialize chat session
        self.chat_session = ChatSession(
            model_name=model_name,
            system_prompt_path=system_prompt_path,
            data_dir=data_dir,
            gemini_api_key=gemini_api_key
        )

        # Set up command history
        self.history_file = os.path.join(data_dir, "cli_history.txt")
        try:
            readline.read_history_file(self.history_file)
        except FileNotFoundError:
            pass

    def preloop(self):
        """Set up before entering the command loop."""
        print(self.intro)

    def postloop(self):
        """Clean up after exiting the command loop."""
        try:
            readline.write_history_file(self.history_file)
        except Exception as e:
            print(f"Error writing history file: {e}")

    def emptyline(self):
        """Do nothing on empty line."""
        pass

    def default(self, line):
        """Default behavior for unknown commands."""
        print(f"Unknown command: {line}")
        print("Type 'help' or '?' to list available commands.")

    def do_exit(self, arg):
        """Exit the chatbot CLI."""
        print("Goodbye!")
        return True

    def do_quit(self, arg):
        """Exit the chatbot CLI."""
        return self.do_exit(arg)

    def do_EOF(self, arg):
        """Exit on EOF (Ctrl+D)."""
        print()
        return self.do_exit(arg)

    def do_chat(self, arg):
        """
        Chat with the data analysis assistant.

        Usage: chat <message>
        """
        if not arg:
            print("Please provide a message.")
            return

        print(f"\nYou: {arg}")
        print("\nProcessing...")

        # Process message
        response_data = self.chat_session.process_message(arg)

        # Print response
        print(f"\nAssistant: {response_data['response']}")

        # Print metadata if there was an error
        if response_data["metadata"].get("error", False):
            print("\nAn error occurred. Please try again.")

    def do_analyze(self, arg):
        """
        Analyze a specific dataset.

        Usage: analyze <dataset_id> <message>
        """
        args = shlex.split(arg)

        if len(args) < 2:
            print("Please provide a dataset ID and a message.")
            return

        dataset_id = args[0]
        message = " ".join(args[1:])

        print(f"\nYou: {message}")
        print(f"Dataset: {dataset_id}")
        print("\nAnalyzing...")

        # Analyze data
        response_data = self.chat_session.analyze_data(message, dataset_id)

        # Print response
        print(f"\nAssistant: {response_data['response']}")

        # Print metadata if there was an error
        if response_data["metadata"].get("error", False):
            print("\nAn error occurred. Please try again.")

    def do_list_datasets(self, arg):
        """
        List available datasets.

        Usage: list_datasets
        """
        datasets = self.chat_session.data_ingestion.list_datasets()

        if not datasets:
            print("No datasets available.")
            return

        print("\nAvailable Datasets:")
        print("=" * 80)
        print(f"{'ID':<40} {'Type':<10} {'Rows':<10} {'Has Embeddings':<15}")
        print("-" * 80)

        for dataset in datasets:
            dataset_id = dataset["id"]
            source_type = dataset.get("source_type", "unknown")
            num_rows = dataset.get("num_rows", "unknown")
            has_embeddings = "Yes" if dataset.get(
                "has_embeddings", False) else "No"

            print(
                f"{dataset_id:<40} {source_type:<10} {num_rows:<10} {has_embeddings:<15}")

    def do_dataset_info(self, arg):
        """
        Show information about a dataset.

        Usage: dataset_info <dataset_id>
        """
        if not arg:
            print("Please provide a dataset ID.")
            return

        dataset_id = arg.strip()
        dataset_info = self.chat_session.data_ingestion.get_dataset_info(
            dataset_id)

        if not dataset_info:
            print(f"Dataset '{dataset_id}' not found.")
            return

        print(f"\nDataset: {dataset_id}")
        print("=" * 80)

        # Print basic info
        print(f"Source Type: {dataset_info.get('source_type', 'unknown')}")
        print(f"Original Path: {dataset_info.get('original_path', 'unknown')}")
        print(f"Number of Rows: {dataset_info.get('num_rows', 'unknown')}")
        print(
            f"Has Embeddings: {'Yes' if dataset_info.get('has_embeddings', False) else 'No'}")

        # Print schema
        schema = self.chat_session.data_ingestion.get_dataset_schema(
            dataset_id)
        print("\nSchema:")
        print(json.dumps(schema, indent=2))

        # Print sample
        sample = self.chat_session.data_ingestion.get_dataset_sample(
            dataset_id, num_rows=3)
        print("\nSample Data:")
        print(json.dumps(sample, indent=2))

    def do_ingest(self, arg):
        """
        Ingest a data file.

        Usage: ingest <file_type> <file_path>

        File types:
          csv: CSV file
          json: JSON file
          excel: Excel file
          text: Text file
          url: Web URL
        """
        args = shlex.split(arg)

        if len(args) < 2:
            print("Please provide a file type and a file path.")
            return

        file_type = args[0].lower()
        file_path = args[1]

        print(f"\nIngesting {file_type} file: {file_path}")

        try:
            dataset_id = None

            if file_type == "csv":
                delimiter = "," if len(args) < 3 else args[2]
                dataset_id = self.chat_session.data_ingestion.ingest_csv(
                    file_path, delimiter=delimiter)
            elif file_type == "json":
                dataset_id = self.chat_session.data_ingestion.ingest_json(
                    file_path)
            elif file_type == "excel":
                sheet_name = None if len(args) < 3 else args[2]
                dataset_id = self.chat_session.data_ingestion.ingest_excel(
                    file_path, sheet_name=sheet_name)
            elif file_type == "text":
                chunk_size = 1000 if len(args) < 3 else int(args[2])
                dataset_id = self.chat_session.data_ingestion.ingest_text(
                    file_path, chunk_size=chunk_size)
            elif file_type == "url":
                dataset_id = self.chat_session.rag_pipeline.add_url(file_path)
            else:
                print(f"Unsupported file type: {file_type}")
                return

            if dataset_id:
                print(
                    f"\nSuccessfully ingested data. Dataset ID: {dataset_id}")

                # Generate embeddings
                print("\nGenerating embeddings...")
                self.chat_session.vector_embedding.generate_embeddings(
                    dataset_id)
                print("Embeddings generated successfully.")

                # Add to active datasets
                self.chat_session.add_active_dataset(dataset_id)
                print(f"Dataset '{dataset_id}' added to active datasets.")
            else:
                print("\nFailed to ingest data.")

        except Exception as e:
            print(f"\nError ingesting data: {e}")

    def do_active_datasets(self, arg):
        """
        List or modify active datasets for RAG.

        Usage:
          active_datasets                 - List active datasets
          active_datasets add <id>        - Add a dataset
          active_datasets remove <id>     - Remove a dataset
          active_datasets clear           - Clear all active datasets
        """
        args = shlex.split(arg)

        if not args:
            # List active datasets
            active_datasets = self.chat_session.get_active_datasets()

            if not active_datasets:
                print("No active datasets.")
                return

            print("\nActive Datasets:")
            for dataset_id in active_datasets:
                print(f"- {dataset_id}")

        elif args[0] == "add" and len(args) >= 2:
            # Add dataset
            dataset_id = args[1]

            # Check if dataset exists
            dataset_info = self.chat_session.data_ingestion.get_dataset_info(
                dataset_id)

            if not dataset_info:
                print(f"Dataset '{dataset_id}' not found.")
                return

            self.chat_session.add_active_dataset(dataset_id)
            print(f"Dataset '{dataset_id}' added to active datasets.")

        elif args[0] == "remove" and len(args) >= 2:
            # Remove dataset
            dataset_id = args[1]

            if dataset_id not in self.chat_session.get_active_datasets():
                print(f"Dataset '{dataset_id}' is not in active datasets.")
                return

            self.chat_session.remove_active_dataset(dataset_id)
            print(f"Dataset '{dataset_id}' removed from active datasets.")

        elif args[0] == "clear":
            # Clear all active datasets
            self.chat_session.set_active_datasets([])
            print("All active datasets cleared.")

        else:
            print("Invalid command. Type 'help active_datasets' for usage.")

    def do_history(self, arg):
        """
        Show conversation history.

        Usage:
          history                 - Show all history
          history <n>             - Show last n messages
          history clear           - Clear history
        """
        args = shlex.split(arg)

        if not args:
            # Show all history
            messages = self.chat_session.get_history()
            self._print_history(messages)

        elif args[0] == "clear":
            # Clear history
            self.chat_session.clear_history()
            print("Conversation history cleared.")

        elif args[0].isdigit():
            # Show last n messages
            limit = int(args[0])
            messages = self.chat_session.get_history(limit=limit)
            self._print_history(messages)

        else:
            print("Invalid command. Type 'help history' for usage.")

    def _print_history(self, messages):
        """Print conversation history messages."""
        if not messages:
            print("No conversation history.")
            return

        print("\nConversation History:")
        print("=" * 80)

        for i, message in enumerate(messages):
            role = message["role"].capitalize()
            content = message["content"]
            timestamp = datetime.fromisoformat(
                message["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")

            print(f"\n[{i+1}] {role} ({timestamp}):")

            # Wrap text for better readability
            wrapped_content = textwrap.fill(
                content, width=76, initial_indent="  ", subsequent_indent="  ")
            print(wrapped_content)

            # Print metadata for assistant messages
            if role.lower() == "assistant" and message.get("metadata"):
                metadata = message["metadata"]

                if metadata.get("model_used"):
                    print(f"  Model: {metadata['model_used']}")

                if metadata.get("fallback_used"):
                    print("  Fallback model used")

                if metadata.get("rag_used"):
                    print("  RAG augmentation used")

                if metadata.get("external_retrieved"):
                    print("  External data retrieved")

                if metadata.get("error"):
                    print("  Error occurred")

        print("\n" + "=" * 80)


class WebInterface:
    """
    Web interface for the data analysis chatbot using Flask.
    """

    def __init__(
        self,
        model_name: str = "databot",
        system_prompt_path: str = "/home/ubuntu/data-ai-chatbot/config/system_prompt.md",
        data_dir: str = "/home/ubuntu/data-ai-chatbot/data",
        gemini_api_key: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 5000
    ):
        """
        Initialize the web interface.

        Args:
            model_name: Name of the Ollama model to use
            system_prompt_path: Path to the system prompt file
            data_dir: Directory containing data
            gemini_api_key: Gemini API key for fallback
            host: Host to bind to
            port: Port to bind to
        """
        self.model_name = model_name
        self.system_prompt_path = system_prompt_path
        self.data_dir = data_dir
        self.gemini_api_key = gemini_api_key
        self.host = host
        self.port = port

        # Initialize chat sessions
        self.chat_sessions = {}

        # Set up Flask app
        self._setup_app()

    def _setup_app(self):
        """Set up the Flask app."""
        try:
            from flask import Flask, request, jsonify, render_template, send_from_directory
            from flask_cors import CORS

            app = Flask(__name__,
                        template_folder=os.path.join(
                            os.path.dirname(__file__), "templates"),
                        static_folder=os.path.join(os.path.dirname(__file__), "static"))
            CORS(app)

            self.app = app

            # Define routes
            @app.route('/')
            def index():
                return render_template('index.html')

            @app.route('/api/chat', methods=['POST'])
            def chat():
                data = request.json

                if not data or 'message' not in data:
                    return jsonify({"error": "Message is required"}), 400

                message = data['message']
                session_id = data.get('session_id', 'default')
                use_rag = data.get('use_rag', True)
                retrieve_external = data.get('retrieve_external', True)

                # Get or create chat session
                chat_session = self._get_chat_session(session_id)

                # Process message
                response_data = chat_session.process_message(
                    message, use_rag, retrieve_external
                )

                return jsonify({
                    "response": response_data["response"],
                    "metadata": response_data["metadata"],
                    "session_id": session_id
                })

            @app.route('/api/analyze', methods=['POST'])
            def analyze():
                data = request.json

                if not data or 'message' not in data or 'dataset_id' not in data:
                    return jsonify({"error": "Message and dataset_id are required"}), 400

                message = data['message']
                dataset_id = data['dataset_id']
                session_id = data.get('session_id', 'default')

                # Get or create chat session
                chat_session = self._get_chat_session(session_id)

                # Analyze data
                response_data = chat_session.analyze_data(message, dataset_id)

                return jsonify({
                    "response": response_data["response"],
                    "metadata": response_data["metadata"],
                    "session_id": session_id
                })

            @app.route('/api/datasets', methods=['GET'])
            def list_datasets():
                session_id = request.args.get('session_id', 'default')

                # Get or create chat session
                chat_session = self._get_chat_session(session_id)

                # List datasets
                datasets = chat_session.data_ingestion.list_datasets()

                return jsonify({"datasets": datasets})

            @app.route('/api/datasets/<dataset_id>', methods=['GET'])
            def dataset_info(dataset_id):
                session_id = request.args.get('session_id', 'default')

                # Get or create chat session
                chat_session = self._get_chat_session(session_id)

                # Get dataset info
                dataset_info = chat_session.data_ingestion.get_dataset_info(
                    dataset_id)

                if not dataset_info:
                    return jsonify({"error": f"Dataset '{dataset_id}' not found"}), 404

                # Get schema and sample
                schema = chat_session.data_ingestion.get_dataset_schema(
                    dataset_id)
                sample = chat_session.data_ingestion.get_dataset_sample(
                    dataset_id, num_rows=5)

                return jsonify({
                    "dataset_id": dataset_id,
                    "info": dataset_info,
                    "schema": schema,
                    "sample": sample
                })

            @app.route('/api/datasets/active', methods=['GET'])
            def active_datasets():
                session_id = request.args.get('session_id', 'default')

                # Get or create chat session
                chat_session = self._get_chat_session(session_id)

                # Get active datasets
                active_datasets = chat_session.get_active_datasets()

                return jsonify({"active_datasets": active_datasets})

            @app.route('/api/datasets/active', methods=['POST'])
            def update_active_datasets():
                data = request.json

                if not data:
                    return jsonify({"error": "Request body is required"}), 400

                session_id = data.get('session_id', 'default')
                action = data.get('action')
                dataset_id = data.get('dataset_id')

                # Get or create chat session
                chat_session = self._get_chat_session(session_id)

                if action == 'add' and dataset_id:
                    # Check if dataset exists
                    dataset_info = chat_session.data_ingestion.get_dataset_info(
                        dataset_id)

                    if not dataset_info:
                        return jsonify({"error": f"Dataset '{dataset_id}' not found"}), 404

                    chat_session.add_active_dataset(dataset_id)
                    return jsonify({"message": f"Dataset '{dataset_id}' added to active datasets"})

                elif action == 'remove' and dataset_id:
                    chat_session.remove_active_dataset(dataset_id)
                    return jsonify({"message": f"Dataset '{dataset_id}' removed from active datasets"})

                elif action == 'clear':
                    chat_session.set_active_datasets([])
                    return jsonify({"message": "All active datasets cleared"})

                else:
                    return jsonify({"error": "Invalid action"}), 400

            @app.route('/api/ingest', methods=['POST'])
            def ingest_data():
                # Check if file is in request
                if 'file' not in request.files:
                    return jsonify({"error": "No file provided"}), 400

                file = request.files['file']

                if file.filename == '':
                    return jsonify({"error": "No file selected"}), 400

                # Get parameters
                file_type = request.form.get('file_type', '').lower()
                session_id = request.form.get('session_id', 'default')

                if not file_type:
                    # Try to infer from filename
                    ext = os.path.splitext(file.filename)[1].lower()

                    if ext in ['.csv']:
                        file_type = 'csv'
                    elif ext in ['.json']:
                        file_type = 'json'
                    elif ext in ['.xlsx', '.xls']:
                        file_type = 'excel'
                    elif ext in ['.txt', '.md', '.rst']:
                        file_type = 'text'
                    else:
                        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

                # Get or create chat session
                chat_session = self._get_chat_session(session_id)

                # Save file temporarily
                temp_dir = os.path.join(self.data_dir, "temp")
                os.makedirs(temp_dir, exist_ok=True)

                temp_path = os.path.join(temp_dir, file.filename)
                file.save(temp_path)

                try:
                    dataset_id = None

                    if file_type == 'csv':
                        delimiter = request.form.get('delimiter', ',')
                        dataset_id = chat_session.data_ingestion.ingest_csv(
                            temp_path, delimiter=delimiter)
                    elif file_type == 'json':
                        dataset_id = chat_session.data_ingestion.ingest_json(
                            temp_path)
                    elif file_type == 'excel':
                        sheet_name = request.form.get('sheet_name')
                        dataset_id = chat_session.data_ingestion.ingest_excel(
                            temp_path, sheet_name=sheet_name)
                    elif file_type == 'text':
                        chunk_size = int(request.form.get('chunk_size', 1000))
                        dataset_id = chat_session.data_ingestion.ingest_text(
                            temp_path, chunk_size=chunk_size)
                    else:
                        return jsonify({"error": f"Unsupported file type: {file_type}"}), 400

                    # Generate embeddings
                    if dataset_id:
                        chat_session.vector_embedding.generate_embeddings(
                            dataset_id)

                        # Add to active datasets
                        chat_session.add_active_dataset(dataset_id)

                        return jsonify({
                            "message": "File ingested successfully",
                            "dataset_id": dataset_id
                        })
                    else:
                        return jsonify({"error": "Failed to ingest file"}), 500

                except Exception as e:
                    return jsonify({"error": f"Error ingesting file: {str(e)}"}), 500

                finally:
                    # Clean up
                    try:
                        os.remove(temp_path)
                    except:
                        pass

            @app.route('/api/ingest/url', methods=['POST'])
            def ingest_url():
                data = request.json

                if not data or 'url' not in data:
                    return jsonify({"error": "URL is required"}), 400

                url = data['url']
                session_id = data.get('session_id', 'default')

                # Get or create chat session
                chat_session = self._get_chat_session(session_id)

                try:
                    # Process URL
                    dataset_id = chat_session.rag_pipeline.add_url(url)

                    if dataset_id:
                        # Add to active datasets
                        chat_session.add_active_dataset(dataset_id)

                        return jsonify({
                            "message": "URL ingested successfully",
                            "dataset_id": dataset_id
                        })
                    else:
                        return jsonify({"error": "Failed to ingest URL"}), 500

                except Exception as e:
                    return jsonify({"error": f"Error ingesting URL: {str(e)}"}), 500

            @app.route('/api/history', methods=['GET'])
            def get_history():
                session_id = request.args.get('session_id', 'default')
                limit = request.args.get('limit')

                # Get or create chat session
                chat_session = self._get_chat_session(session_id)

                # Get history
                if limit and limit.isdigit():
                    messages = chat_session.get_history(limit=int(limit))
                else:
                    messages = chat_session.get_history()

                return jsonify({"history": messages})

            @app.route('/api/history', methods=['DELETE'])
            def clear_history():
                session_id = request.args.get('session_id', 'default')

                # Get or create chat session
                chat_session = self._get_chat_session(session_id)

                # Clear history
                chat_session.clear_history()

                return jsonify({"message": "Conversation history cleared"})

        except ImportError:
            print("Flask not installed. Installing...")
            import subprocess
            subprocess.run(["pip", "install", "flask",
                           "flask-cors"], check=True)
            self._setup_app()

    def _get_chat_session(self, session_id: str) -> ChatSession:
        """
        Get or create a chat session.

        Args:
            session_id: Session identifier

        Returns:
            Chat session
        """
        if session_id not in self.chat_sessions:
            self.chat_sessions[session_id] = ChatSession(
                model_name=self.model_name,
                system_prompt_path=self.system_prompt_path,
                data_dir=self.data_dir,
                gemini_api_key=self.gemini_api_key,
                conversation_id=session_id
            )

        return self.chat_sessions[session_id]

    def run(self):
        """Run the web interface."""
        print(f"Starting web interface on http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port)


def create_web_templates():
    """Create web templates for the Flask app."""
    # Create templates directory
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    os.makedirs(templates_dir, exist_ok=True)

    # Create static directory
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(os.path.join(static_dir, "css"), exist_ok=True)
    os.makedirs(os.path.join(static_dir, "js"), exist_ok=True)

    # Create index.html
    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data AI Chatbot</title>
    <link rel="stylesheet" href="/static/css/styles.css">
</head>
<body>
    <div class="app-container">
        <div class="sidebar">
            <div class="sidebar-header">
                <h2>Data AI Chatbot</h2>
            </div>
            <div class="sidebar-content">
                <div class="section">
                    <h3>Datasets</h3>
                    <div class="dataset-list" id="datasetList">
                        <p>Loading datasets...</p>
                    </div>
                    <div class="dataset-actions">
                        <button id="uploadBtn">Upload Data</button>
                        <button id="urlBtn">Add URL</button>
                    </div>
                </div>
                <div class="section">
                    <h3>Active Datasets</h3>
                    <div class="active-dataset-list" id="activeDatasetList">
                        <p>No active datasets</p>
                    </div>
                </div>
                <div class="section">
                    <h3>Settings</h3>
                    <div class="settings">
                        <label>
                            <input type="checkbox" id="useRag" checked>
                            Use RAG
                        </label>
                        <label>
                            <input type="checkbox" id="retrieveExternal" checked>
                            Retrieve External Data
                        </label>
                    </div>
                </div>
            </div>
            <div class="sidebar-footer">
                <button id="clearHistoryBtn">Clear History</button>
            </div>
        </div>
        <div class="main-content">
            <div class="chat-container">
                <div class="chat-messages" id="chatMessages">
                    <div class="welcome-message">
                        <h2>Welcome to Data AI Chatbot</h2>
                        <p>Your interactive data analysis assistant</p>
                        <p>Upload data or add URLs to get started, then ask questions about your data.</p>
                    </div>
                </div>
                <div class="chat-input">
                    <textarea id="messageInput" placeholder="Ask a question about your data..."></textarea>
                    <button id="sendBtn">Send</button>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Upload Modal -->
    <div class="modal" id="uploadModal">
        <div class="modal-content">
            <span class="close-btn" id="closeUploadModal">&times;</span>
            <h2>Upload Data</h2>
            <form id="uploadForm">
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
            </form>
        </div>
    </div>
    
    <!-- URL Modal -->
    <div class="modal" id="urlModal">
        <div class="modal-content">
            <span class="close-btn" id="closeUrlModal">&times;</span>
            <h2>Add URL</h2>
            <form id="urlForm">
                <div class="form-group">
                    <label for="urlInput">URL:</label>
                    <input type="url" id="urlInput" required placeholder="https://example.com">
                </div>
                <button type="submit">Add</button>
            </form>
        </div>
    </div>
    
    <!-- Dataset Info Modal -->
    <div class="modal" id="datasetInfoModal">
        <div class="modal-content dataset-info-content">
            <span class="close-btn" id="closeDatasetInfoModal">&times;</span>
            <h2 id="datasetInfoTitle">Dataset Information</h2>
            <div class="dataset-info" id="datasetInfo">
                <p>Loading dataset information...</p>
            </div>
        </div>
    </div>
    
    <script src="/static/js/app.js"></script>
</body>
</html>
"""

    with open(os.path.join(templates_dir, "index.html"), "w") as f:
        f.write(index_html)

    # Create CSS
    css = """/* General Styles */
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f5f5f5;
}

button {
    cursor: pointer;
    background-color: #4a6fa5;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    font-size: 14px;
    transition: background-color 0.3s;
}

button:hover {
    background-color: #3a5a80;
}

/* App Layout */
.app-container {
    display: flex;
    height: 100vh;
    overflow: hidden;
}

/* Sidebar */
.sidebar {
    width: 300px;
    background-color: #2c3e50;
    color: #ecf0f1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.sidebar-header {
    padding: 20px;
    background-color: #1a2530;
    border-bottom: 1px solid #34495e;
}

.sidebar-content {
    flex: 1;
    overflow-y: auto;
    padding: 10px;
}

.sidebar-footer {
    padding: 15px;
    background-color: #1a2530;
    border-top: 1px solid #34495e;
    text-align: center;
}

.section {
    margin-bottom: 20px;
}

.section h3 {
    margin-bottom: 10px;
    padding-bottom: 5px;
    border-bottom: 1px solid #34495e;
}

/* Dataset Lists */
.dataset-list, .active-dataset-list {
    margin-bottom: 15px;
    max-height: 200px;
    overflow-y: auto;
}

.dataset-item {
    padding: 8px;
    margin-bottom: 5px;
    background-color: #34495e;
    border-radius: 4px;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.dataset-item:hover {
    background-color: #3d5871;
}

.dataset-item-info {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.dataset-item-actions {
    display: flex;
    gap: 5px;
}

.dataset-item-actions button {
    padding: 2px 5px;
    font-size: 12px;
    background-color: #5a7a9f;
}

.dataset-actions {
    display: flex;
    gap: 10px;
}

/* Settings */
.settings {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.settings label {
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
}

/* Main Content */
.main-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

/* Chat Container */
.chat-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background-color: #fff;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
}

.welcome-message {
    text-align: center;
    margin: 50px auto;
    max-width: 600px;
    padding: 30px;
    background-color: #f8f9fa;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.welcome-message h2 {
    margin-bottom: 15px;
    color: #2c3e50;
}

.welcome-message p {
    margin-bottom: 10px;
    color: #7f8c8d;
}

.message {
    margin-bottom: 15px;
    max-width: 80%;
}

.user-message {
    margin-left: auto;
    background-color: #4a6fa5;
    color: white;
    border-radius: 18px 18px 0 18px;
    padding: 12px 18px;
}

.assistant-message-container {
    display: flex;
    flex-direction: column;
    margin-bottom: 20px;
}

.assistant-message {
    background-color: #f1f1f1;
    border-radius: 18px 18px 18px 0;
    padding: 12px 18px;
    align-self: flex-start;
}

.message-metadata {
    font-size: 12px;
    color: #7f8c8d;
    margin-top: 5px;
    padding-left: 10px;
}

.chat-input {
    display: flex;
    padding: 15px;
    background-color: #f8f9fa;
    border-top: 1px solid #e9ecef;
}

.chat-input textarea {
    flex: 1;
    padding: 12px;
    border: 1px solid #ced4da;
    border-radius: 4px;
    resize: none;
    height: 50px;
    font-family: inherit;
}

.chat-input button {
    margin-left: 10px;
    align-self: flex-end;
}

/* Modals */
.modal {
    display: none;
    position: fixed;
    z-index: 1000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
}

.modal-content {
    background-color: #fff;
    margin: 10% auto;
    padding: 20px;
    border-radius: 8px;
    width: 80%;
    max-width: 500px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    position: relative;
}

.dataset-info-content {
    max-width: 800px;
    width: 90%;
    max-height: 80vh;
    overflow-y: auto;
}

.close-btn {
    position: absolute;
    top: 10px;
    right: 15px;
    font-size: 24px;
    cursor: pointer;
    color: #aaa;
}

.close-btn:hover {
    color: #333;
}

.form-group {
    margin-bottom: 15px;
}

.form-group label {
    display: block;
    margin-bottom: 5px;
    font-weight: 500;
}

.form-group input, .form-group select {
    width: 100%;
    padding: 8px;
    border: 1px solid #ced4da;
    border-radius: 4px;
}

/* Dataset Info Styling */
.dataset-info {
    margin-top: 15px;
}

.dataset-info h3 {
    margin: 15px 0 10px;
    padding-bottom: 5px;
    border-bottom: 1px solid #e9ecef;
}

.dataset-info-section {
    margin-bottom: 20px;
}

.dataset-info-item {
    margin-bottom: 8px;
}

.dataset-info-label {
    font-weight: 500;
    margin-right: 10px;
}

.dataset-sample {
    overflow-x: auto;
    margin-top: 10px;
}

.dataset-sample table {
    border-collapse: collapse;
    width: 100%;
}

.dataset-sample th, .dataset-sample td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}

.dataset-sample th {
    background-color: #f2f2f2;
}

.dataset-sample tr:nth-child(even) {
    background-color: #f9f9f9;
}

/* Loading Indicator */
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    border-top-color: #fff;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Responsive Adjustments */
@media (max-width: 768px) {
    .app-container {
        flex-direction: column;
    }
    
    .sidebar {
        width: 100%;
        height: 200px;
    }
    
    .sidebar-content {
        display: flex;
        overflow-x: auto;
    }
    
    .section {
        min-width: 250px;
        margin-right: 15px;
    }
    
    .modal-content {
        width: 95%;
        margin: 5% auto;
    }
}
"""

    with open(os.path.join(static_dir, "css", "styles.css"), "w") as f:
        f.write(css)

    # Create JavaScript
    js = """// Global variables
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
"""

    with open(os.path.join(static_dir, "js", "app.js"), "w") as f:
        f.write(js)


# Main function
def main():
    """Main function for the chatbot interface."""
    parser = argparse.ArgumentParser(description="Data AI Chatbot Interface")
    parser.add_argument("--model", default="databot",
                        help="Name of the Ollama model to use")
    parser.add_argument("--system-prompt", default="/home/ubuntu/data-ai-chatbot/config/system_prompt.md",
                        help="Path to the system prompt file")
    parser.add_argument(
        "--data-dir", default="/home/ubuntu/data-ai-chatbot/data", help="Directory containing data")
    parser.add_argument("--gemini-api-key", help="Gemini API key for fallback")
    parser.add_argument(
        "--interface", choices=["cli", "web"], default="web", help="Interface type (cli or web)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind to for web interface")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to bind to for web interface")

    args = parser.parse_args()

    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)

    # Create web templates if using web interface
    if args.interface == "web":
        create_web_templates()

    # Start interface
    if args.interface == "cli":
        # Start CLI interface
        cli = ChatbotCLI(
            model_name=args.model,
            system_prompt_path=args.system_prompt,
            data_dir=args.data_dir,
            gemini_api_key=args.gemini_api_key
        )
        cli.cmdloop()
    else:
        # Start web interface
        web = WebInterface(
            model_name=args.model,
            system_prompt_path=args.system_prompt,
            data_dir=args.data_dir,
            gemini_api_key=args.gemini_api_key,
            host=args.host,
            port=args.port
        )
        web.run()


if __name__ == "__main__":
    main()

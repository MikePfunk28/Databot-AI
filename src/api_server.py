#!/usr/bin/env python3
"""
API Server Module for Data AI Chatbot

This module provides a REST API for interacting with the data analysis chatbot,
as well as integration with Model Context Protocol (MCP) servers.
"""

from src.rag_pipeline import RAGPipeline
from src.data_ingestion import DataIngestion, VectorEmbedding
from src.model_wrapper import ModelWrapper, GeminiWrapper, ModelManager
import os
import json
import time
import argparse
import threading
import uuid
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import logging
import socket
import asyncio
import websockets

# Import local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class APIServer:
    """
    REST API server for the data analysis chatbot.
    """

    def __init__(
        self,
        model_name: str = "databot",
        system_prompt_path: str = "/home/ubuntu/data-ai-chatbot/config/system_prompt.md",
        data_dir: str = "/home/ubuntu/data-ai-chatbot/data",
        gemini_api_key: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8000
    ):
        """
        Initialize the API server.

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

        # Initialize components
        self._init_components()

        # Initialize session storage
        self.sessions = {}

        # Set up FastAPI app
        self._setup_app()

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
            logger.error(f"Error initializing components: {e}")
            raise

    def _setup_app(self):
        """Set up the FastAPI app."""
        try:
            from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form, Query, BackgroundTasks
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import JSONResponse
            from fastapi.staticfiles import StaticFiles
            from pydantic import BaseModel, Field
            import uvicorn

            app = FastAPI(
                title="Data AI Chatbot API",
                description="API for interacting with the data analysis chatbot",
                version="1.0.0"
            )

            # Add CORS middleware
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            self.app = app

            # Define Pydantic models for request/response
            class ChatRequest(BaseModel):
                message: str
                session_id: Optional[str] = "default"
                use_rag: Optional[bool] = True
                retrieve_external: Optional[bool] = True

            class AnalyzeRequest(BaseModel):
                message: str
                dataset_id: str
                session_id: Optional[str] = "default"

            class ActiveDatasetRequest(BaseModel):
                session_id: Optional[str] = "default"
                action: str
                dataset_id: Optional[str] = None

            class URLIngestRequest(BaseModel):
                url: str
                session_id: Optional[str] = "default"

            class StreamRequest(BaseModel):
                message: str
                session_id: Optional[str] = "default"
                use_rag: Optional[bool] = True
                retrieve_external: Optional[bool] = True

            # Define routes
            @app.get("/")
            async def root():
                return {"message": "Data AI Chatbot API", "version": "1.0.0"}

            @app.get("/health")
            async def health():
                return {"status": "ok"}

            @app.post("/api/chat")
            async def chat(request: ChatRequest):
                try:
                    # Get or create session
                    session = self._get_session(request.session_id)

                    # Process message
                    response_data = session.process_message(
                        request.message,
                        use_rag=request.use_rag,
                        retrieve_external=request.retrieve_external
                    )

                    return {
                        "response": response_data["response"],
                        "metadata": response_data["metadata"],
                        "session_id": request.session_id
                    }
                except Exception as e:
                    logger.error(f"Error in chat endpoint: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            @app.post("/api/analyze")
            async def analyze(request: AnalyzeRequest):
                try:
                    # Get or create session
                    session = self._get_session(request.session_id)

                    # Analyze data
                    response_data = session.analyze_data(
                        request.message,
                        request.dataset_id
                    )

                    return {
                        "response": response_data["response"],
                        "metadata": response_data["metadata"],
                        "session_id": request.session_id
                    }
                except Exception as e:
                    logger.error(f"Error in analyze endpoint: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            @app.get("/api/datasets")
            async def list_datasets(session_id: str = "default"):
                try:
                    # Get or create session
                    session = self._get_session(session_id)

                    # List datasets
                    datasets = session.data_ingestion.list_datasets()

                    return {"datasets": datasets}
                except Exception as e:
                    logger.error(f"Error in list_datasets endpoint: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            @app.get("/api/datasets/{dataset_id}")
            async def dataset_info(dataset_id: str, session_id: str = "default"):
                try:
                    # Get or create session
                    session = self._get_session(session_id)

                    # Get dataset info
                    dataset_info = session.data_ingestion.get_dataset_info(
                        dataset_id)

                    if not dataset_info:
                        raise HTTPException(
                            status_code=404, detail=f"Dataset '{dataset_id}' not found")

                    # Get schema and sample
                    schema = session.data_ingestion.get_dataset_schema(
                        dataset_id)
                    sample = session.data_ingestion.get_dataset_sample(
                        dataset_id, num_rows=5)

                    return {
                        "dataset_id": dataset_id,
                        "info": dataset_info,
                        "schema": schema,
                        "sample": sample
                    }
                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"Error in dataset_info endpoint: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            @app.get("/api/datasets/active")
            async def active_datasets(session_id: str = "default"):
                try:
                    # Get or create session
                    session = self._get_session(session_id)

                    # Get active datasets
                    active_datasets = session.get_active_datasets()

                    return {"active_datasets": active_datasets}
                except Exception as e:
                    logger.error(f"Error in active_datasets endpoint: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            @app.post("/api/datasets/active")
            async def update_active_datasets(request: ActiveDatasetRequest):
                try:
                    # Get or create session
                    session = self._get_session(request.session_id)

                    if request.action == "add" and request.dataset_id:
                        # Check if dataset exists
                        dataset_info = session.data_ingestion.get_dataset_info(
                            request.dataset_id)

                        if not dataset_info:
                            raise HTTPException(
                                status_code=404, detail=f"Dataset '{request.dataset_id}' not found")

                        session.add_active_dataset(request.dataset_id)
                        return {"message": f"Dataset '{request.dataset_id}' added to active datasets"}

                    elif request.action == "remove" and request.dataset_id:
                        session.remove_active_dataset(request.dataset_id)
                        return {"message": f"Dataset '{request.dataset_id}' removed from active datasets"}

                    elif request.action == "clear":
                        session.set_active_datasets([])
                        return {"message": "All active datasets cleared"}

                    else:
                        raise HTTPException(
                            status_code=400, detail="Invalid action")
                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(
                        f"Error in update_active_datasets endpoint: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            @app.post("/api/ingest")
            async def ingest_data(
                background_tasks: BackgroundTasks,
                file: UploadFile = File(...),
                file_type: str = Form(...),
                session_id: str = Form("default"),
                delimiter: Optional[str] = Form(","),
                sheet_name: Optional[str] = Form(None),
                chunk_size: Optional[int] = Form(1000)
            ):
                try:
                    # Get or create session
                    session = self._get_session(session_id)

                    # Save file temporarily
                    temp_dir = os.path.join(self.data_dir, "temp")
                    os.makedirs(temp_dir, exist_ok=True)

                    temp_path = os.path.join(temp_dir, file.filename)
                    with open(temp_path, "wb") as f:
                        f.write(await file.read())

                    # Process file based on type
                    dataset_id = None

                    if file_type == "csv":
                        dataset_id = session.data_ingestion.ingest_csv(
                            temp_path, delimiter=delimiter)
                    elif file_type == "json":
                        dataset_id = session.data_ingestion.ingest_json(
                            temp_path)
                    elif file_type == "excel":
                        dataset_id = session.data_ingestion.ingest_excel(
                            temp_path, sheet_name=sheet_name)
                    elif file_type == "text":
                        dataset_id = session.data_ingestion.ingest_text(
                            temp_path, chunk_size=chunk_size)
                    else:
                        raise HTTPException(
                            status_code=400, detail=f"Unsupported file type: {file_type}")

                    if not dataset_id:
                        raise HTTPException(
                            status_code=500, detail="Failed to ingest file")

                    # Generate embeddings in the background
                    def generate_embeddings_task():
                        try:
                            session.vector_embedding.generate_embeddings(
                                dataset_id)
                            session.add_active_dataset(dataset_id)
                        except Exception as e:
                            logger.error(f"Error generating embeddings: {e}")

                    background_tasks.add_task(generate_embeddings_task)

                    # Clean up
                    os.remove(temp_path)

                    return {
                        "message": "File ingested successfully, embeddings being generated",
                        "dataset_id": dataset_id
                    }
                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"Error in ingest_data endpoint: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            @app.post("/api/ingest/url")
            async def ingest_url(
                background_tasks: BackgroundTasks,
                request: URLIngestRequest
            ):
                try:
                    # Get or create session
                    session = self._get_session(request.session_id)

                    # Process URL
                    dataset_id = session.rag_pipeline.add_url(request.url)

                    if not dataset_id:
                        raise HTTPException(
                            status_code=500, detail="Failed to ingest URL")

                    # Add to active datasets
                    session.add_active_dataset(dataset_id)

                    return {
                        "message": "URL ingested successfully",
                        "dataset_id": dataset_id
                    }
                except Exception as e:
                    logger.error(f"Error in ingest_url endpoint: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            @app.get("/api/history")
            async def get_history(
                session_id: str = "default",
                limit: Optional[int] = None
            ):
                try:
                    # Get or create session
                    session = self._get_session(session_id)

                    # Get history
                    messages = session.get_history(limit=limit)

                    return {"history": messages}
                except Exception as e:
                    logger.error(f"Error in get_history endpoint: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            @app.delete("/api/history")
            async def clear_history(session_id: str = "default"):
                try:
                    # Get or create session
                    session = self._get_session(session_id)

                    # Clear history
                    session.clear_history()

                    return {"message": "Conversation history cleared"}
                except Exception as e:
                    logger.error(f"Error in clear_history endpoint: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            # Streaming endpoint
            @app.websocket("/api/stream")
            async def stream(websocket):
                await websocket.accept()

                try:
                    # Get request data
                    data = await websocket.receive_json()

                    message = data.get("message")
                    session_id = data.get("session_id", "default")
                    use_rag = data.get("use_rag", True)
                    retrieve_external = data.get("retrieve_external", True)

                    if not message:
                        await websocket.send_json({"error": "Message is required"})
                        await websocket.close()
                        return

                    # Get or create session
                    session = self._get_session(session_id)

                    # Add user message to history
                    session.add_message("user", message)

                    # Process with RAG if enabled
                    context = None
                    rag_results = None

                    if use_rag:
                        # Send processing message
                        await websocket.send_json({
                            "type": "status",
                            "status": "retrieving_context"
                        })

                        rag_results = session.rag_pipeline.process_query(
                            message,
                            use_existing_data=True,
                            retrieve_external=retrieve_external,
                            dataset_ids=session.active_datasets if session.active_datasets else None
                        )

                        context = rag_results.get("formatted_context")

                        # Send context retrieved message
                        await websocket.send_json({
                            "type": "status",
                            "status": "context_retrieved",
                            "context_count": len(rag_results.get("context", [])),
                            "external_retrieved": rag_results.get("external_retrieved", False)
                        })

                    # Send generating message
                    await websocket.send_json({
                        "type": "status",
                        "status": "generating"
                    })

                    # Generate response
                    response_data = session.model_manager.generate_response(
                        message, context)

                    # Add assistant message to history
                    metadata = {
                        "model_used": response_data.get("model_used", "unknown"),
                        "fallback_used": response_data.get("fallback_used", False),
                        "rag_used": use_rag,
                        "external_retrieved": rag_results.get("external_retrieved", False) if rag_results else False,
                        "sources": rag_results.get("sources", []) if rag_results else []
                    }

                    session.add_message(
                        "assistant", response_data["response"], metadata)

                    # Send response
                    await websocket.send_json({
                        "type": "response",
                        "response": response_data["response"],
                        "metadata": metadata
                    })

                    # Close connection
                    await websocket.close()

                except Exception as e:
                    logger.error(f"Error in stream endpoint: {e}")

                    try:
                        await websocket.send_json({
                            "type": "error",
                            "error": str(e)
                        })
                        await websocket.close()
                    except:
                        pass

            # MCP endpoints
            @app.post("/api/mcp/register")
            async def mcp_register(
                host: str = Form(...),
                port: int = Form(...),
                name: str = Form(...),
                description: Optional[str] = Form(None)
            ):
                try:
                    # Register MCP server
                    mcp_server = {
                        "host": host,
                        "port": port,
                        "name": name,
                        "description": description,
                        "registered_at": datetime.now().isoformat()
                    }

                    # Save to MCP servers file
                    mcp_servers_dir = os.path.join(
                        self.data_dir, "mcp_servers")
                    os.makedirs(mcp_servers_dir, exist_ok=True)

                    mcp_servers_file = os.path.join(
                        mcp_servers_dir, "servers.json")

                    if os.path.exists(mcp_servers_file):
                        with open(mcp_servers_file, "r") as f:
                            mcp_servers = json.load(f)
                    else:
                        mcp_servers = []

                    # Check if server already exists
                    for i, server in enumerate(mcp_servers):
                        if server["host"] == host and server["port"] == port:
                            # Update existing server
                            mcp_servers[i] = mcp_server
                            break
                    else:
                        # Add new server
                        mcp_servers.append(mcp_server)

                    with open(mcp_servers_file, "w") as f:
                        json.dump(mcp_servers, f, indent=2)

                    return {"message": "MCP server registered successfully"}
                except Exception as e:
                    logger.error(f"Error in mcp_register endpoint: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            @app.get("/api/mcp/servers")
            async def mcp_servers():
                try:
                    # Get MCP servers
                    mcp_servers_dir = os.path.join(
                        self.data_dir, "mcp_servers")
                    mcp_servers_file = os.path.join(
                        mcp_servers_dir, "servers.json")

                    if os.path.exists(mcp_servers_file):
                        with open(mcp_servers_file, "r") as f:
                            mcp_servers = json.load(f)
                    else:
                        mcp_servers = []

                    return {"servers": mcp_servers}
                except Exception as e:
                    logger.error(f"Error in mcp_servers endpoint: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            @app.post("/api/mcp/query")
            async def mcp_query(
                server_name: str = Form(...),
                query: str = Form(...),
                session_id: str = Form("default")
            ):
                try:
                    # Get MCP servers
                    mcp_servers_dir = os.path.join(
                        self.data_dir, "mcp_servers")
                    mcp_servers_file = os.path.join(
                        mcp_servers_dir, "servers.json")

                    if not os.path.exists(mcp_servers_file):
                        raise HTTPException(
                            status_code=404, detail="No MCP servers registered")

                    with open(mcp_servers_file, "r") as f:
                        mcp_servers = json.load(f)

                    # Find server
                    server = None
                    for s in mcp_servers:
                        if s["name"] == server_name:
                            server = s
                            break

                    if not server:
                        raise HTTPException(
                            status_code=404, detail=f"MCP server '{server_name}' not found")

                    # Query server
                    result = await self._query_mcp_server(
                        server["host"],
                        server["port"],
                        query,
                        session_id
                    )

                    return result
                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"Error in mcp_query endpoint: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            # Model information endpoint
            @app.get("/api/model/info")
            async def model_info():
                try:
                    # Get model info
                    model_info = self.primary_model.get_model_info()

                    return model_info
                except Exception as e:
                    logger.error(f"Error in model_info endpoint: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            self.uvicorn_config = uvicorn.Config(
                app,
                host=self.host,
                port=self.port,
                log_level="info"
            )
            self.uvicorn_server = uvicorn.Server(self.uvicorn_config)

        except ImportError:
            logger.error("FastAPI not installed. Installing...")
            import subprocess
            subprocess.run(["pip", "install", "fastapi", "uvicorn",
                           "python-multipart", "websockets"], check=True)
            self._setup_app()

    def _get_session(self, session_id: str):
        """
        Get or create a session.

        Args:
            session_id: Session identifier

        Returns:
            Session object
        """
        from src.chatbot_interface import ChatSession

        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession(
                model_name=self.model_name,
                system_prompt_path=self.system_prompt_path,
                data_dir=self.data_dir,
                gemini_api_key=self.gemini_api_key,
                conversation_id=session_id
            )

        return self.sessions[session_id]

    async def _query_mcp_server(
        self,
        host: str,
        port: int,
        query: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Query an MCP server.

        Args:
            host: Server host
            port: Server port
            query: Query string
            session_id: Session identifier

        Returns:
            Server response
        """
        try:
            # Connect to server
            uri = f"ws://{host}:{port}/mcp"

            async with websockets.connect(uri) as websocket:
                # Send query
                await websocket.send(json.dumps({
                    "type": "query",
                    "query": query,
                    "session_id": session_id
                }))

                # Get response
                response = await websocket.recv()

                return json.loads(response)
        except Exception as e:
            logger.error(f"Error querying MCP server: {e}")
            raise

    def run(self):
        """Run the API server."""
        logger.info(f"Starting API server on http://{self.host}:{self.port}")
        self.uvicorn_server.run()


class MCPServer:
    """
    Model Context Protocol (MCP) server for the data analysis chatbot.
    """

    def __init__(
        self,
        model_name: str = "databot",
        system_prompt_path: str = "/home/ubuntu/data-ai-chatbot/config/system_prompt.md",
        data_dir: str = "/home/ubuntu/data-ai-chatbot/data",
        gemini_api_key: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8765,
        name: str = "default",
        description: Optional[str] = None,
        api_server_host: str = "localhost",
        api_server_port: int = 8000
    ):
        """
        Initialize the MCP server.

        Args:
            model_name: Name of the Ollama model to use
            system_prompt_path: Path to the system prompt file
            data_dir: Directory containing data
            gemini_api_key: Gemini API key for fallback
            host: Host to bind to
            port: Port to bind to
            name: Server name
            description: Server description
            api_server_host: API server host
            api_server_port: API server port
        """
        self.model_name = model_name
        self.system_prompt_path = system_prompt_path
        self.data_dir = data_dir
        self.gemini_api_key = gemini_api_key
        self.host = host
        self.port = port
        self.name = name
        self.description = description
        self.api_server_host = api_server_host
        self.api_server_port = api_server_port

        # Initialize components
        self._init_components()

        # Initialize session storage
        self.sessions = {}

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
            logger.error(f"Error initializing components: {e}")
            raise

    def _get_session(self, session_id: str):
        """
        Get or create a session.

        Args:
            session_id: Session identifier

        Returns:
            Session object
        """
        from src.chatbot_interface import ChatSession

        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession(
                model_name=self.model_name,
                system_prompt_path=self.system_prompt_path,
                data_dir=self.data_dir,
                gemini_api_key=self.gemini_api_key,
                conversation_id=session_id
            )

        return self.sessions[session_id]

    async def _register_with_api_server(self):
        """Register with the API server."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                url = f"http://{self.api_server_host}:{self.api_server_port}/api/mcp/register"

                data = {
                    "host": self.host,
                    "port": self.port,
                    "name": self.name,
                    "description": self.description
                }

                async with session.post(url, data=data) as response:
                    if response.status != 200:
                        logger.error(f"Failed to register with API server: {await response.text()}")
                    else:
                        logger.info("Registered with API server successfully")
        except Exception as e:
            logger.error(f"Error registering with API server: {e}")

    async def handle_connection(self, websocket, path):
        """
        Handle a WebSocket connection.

        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        try:
            # Get request data
            data = await websocket.recv()
            data = json.loads(data)

            request_type = data.get("type")

            if request_type == "query":
                # Handle query
                query = data.get("query")
                session_id = data.get("session_id", "default")

                if not query:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": "Query is required"
                    }))
                    return

                # Get or create session
                session = self._get_session(session_id)

                # Process query
                response_data = session.process_message(query)

                # Send response
                await websocket.send(json.dumps({
                    "type": "response",
                    "response": response_data["response"],
                    "metadata": response_data["metadata"]
                }))
            elif request_type == "ping":
                # Handle ping
                await websocket.send(json.dumps({
                    "type": "pong",
                    "name": self.name,
                    "description": self.description
                }))
            else:
                # Unknown request type
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": f"Unknown request type: {request_type}"
                }))
        except Exception as e:
            logger.error(f"Error handling WebSocket connection: {e}")

            try:
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": str(e)
                }))
            except:
                pass

    async def run_server(self):
        """Run the MCP server."""
        try:
            # Register with API server
            await self._register_with_api_server()

            # Start WebSocket server
            server = await websockets.serve(
                self.handle_connection,
                self.host,
                self.port
            )

            logger.info(
                f"MCP server '{self.name}' running on ws://{self.host}:{self.port}")

            # Keep server running
            await server.wait_closed()
        except Exception as e:
            logger.error(f"Error running MCP server: {e}")
            raise

    def run(self):
        """Run the MCP server."""
        asyncio.run(self.run_server())


# Main function
def main():
    """Main function for the API server."""
    parser = argparse.ArgumentParser(description="Data AI Chatbot API Server")
    parser.add_argument("--model", default="databot",
                        help="Name of the Ollama model to use")
    parser.add_argument("--system-prompt", default="/home/ubuntu/data-ai-chatbot/config/system_prompt.md",
                        help="Path to the system prompt file")
    parser.add_argument(
        "--data-dir", default="/home/ubuntu/data-ai-chatbot/data", help="Directory containing data")
    parser.add_argument("--gemini-api-key", help="Gemini API key for fallback")
    parser.add_argument(
        "--server-type", choices=["api", "mcp"], default="api", help="Server type (api or mcp)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind to")
    parser.add_argument("--mcp-name", default="default",
                        help="MCP server name")
    parser.add_argument("--mcp-description", help="MCP server description")
    parser.add_argument("--api-server-host", default="localhost",
                        help="API server host for MCP registration")
    parser.add_argument("--api-server-port", type=int, default=8000,
                        help="API server port for MCP registration")

    args = parser.parse_args()

    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)

    # Start server
    if args.server_type == "api":
        # Start API server
        api_server = APIServer(
            model_name=args.model,
            system_prompt_path=args.system_prompt,
            data_dir=args.data_dir,
            gemini_api_key=args.gemini_api_key,
            host=args.host,
            port=args.port
        )
        api_server.run()
    else:
        # Start MCP server
        mcp_server = MCPServer(
            model_name=args.model,
            system_prompt_path=args.system_prompt,
            data_dir=args.data_dir,
            gemini_api_key=args.gemini_api_key,
            host=args.host,
            port=args.port,
            name=args.mcp_name,
            description=args.mcp_description,
            api_server_host=args.api_server_host,
            api_server_port=args.api_server_port
        )
        mcp_server.run()


if __name__ == "__main__":
    main()

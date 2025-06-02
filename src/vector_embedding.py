#!/usr/bin/env python3
"""
Vector Embedding Module for Data AI Chatbot

This module provides functionality for generating and managing vector embeddings
with fallback to simulated embeddings when dependencies are not available.
"""

import os
import json
import numpy as np
import hashlib
import pickle
import requests
from typing import Dict, List, Optional, Union, Any, Tuple

# Check for sentence_transformers availability
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Note: sentence_transformers not available. Vector embeddings will be simulated.")
    print("To install in production: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Check for FAISS availability
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("Note: faiss not available. Vector search will be simulated.")
    print("To install in production: pip install faiss-cpu or pip install faiss-gpu")
    FAISS_AVAILABLE = False


class VectorEmbedding:
    """
    Class for generating and managing vector embeddings.
    """
    
    def __init__(self, data_dir: str, model_name: str = "mikepfunk28/databot-embed"):
        """
        Initialize the vector embedding manager.
        
        Args:
            data_dir: Directory for storing embeddings
            model_name: Name of the embedding model (default: mikepfunk28/databot-embed)
        """
        self.data_dir = data_dir
        self.model_name = model_name
        self.embeddings_dir = os.path.join(data_dir, "embeddings")
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Initialize embedding model if available
        self.model = None
        self.use_ollama = False
        
        # Check if this is an Ollama model (contains slash like mikepfunk28/databot-embed)
        if "/" in model_name and not model_name.startswith("sentence-transformers/"):
            self.use_ollama = True
            print(f"Using Ollama embedding model: {model_name}")
            # Test Ollama connection
            try:
                response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json()
                    available_models = [m['name'] for m in models.get('models', [])]
                    if model_name in available_models:
                        print(f"Successfully connected to Ollama model: {model_name}")
                    else:
                        print(f"Warning: {model_name} not found in Ollama. Available models: {available_models}")
                else:
                    print("Warning: Ollama server not responding properly")
                    self.use_ollama = False
            except Exception as e:
                print(f"Warning: Cannot connect to Ollama server: {e}")
                self.use_ollama = False
        
        # Fallback to sentence transformers if not using Ollama
        if not self.use_ollama and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Try to load the nomic embed model first
                if "nomic-embed" in model_name:
                    print(f"Loading Nomic Embed model: {model_name}")
                    self.model = SentenceTransformer(model_name, trust_remote_code=True)
                else:
                    self.model = SentenceTransformer(model_name)
                print(f"Successfully initialized embedding model: {model_name}")
            except Exception as e:
                print(f"Error initializing embedding model {model_name}: {e}")
                print("Trying fallback model: all-MiniLM-L6-v2")
                try:
                    self.model = SentenceTransformer("all-MiniLM-L6-v2")
                    self.model_name = "all-MiniLM-L6-v2"
                    print("Successfully loaded fallback model")
                except Exception as e2:
                    print(f"Error with fallback model: {e2}")
                    print("Falling back to simulated embeddings")
    
    def generate_embeddings(self, dataset_id: str) -> bool:
        """
        Generate embeddings for a dataset.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Success status
        """
        # Get dataset path
        dataset_path = os.path.join(self.data_dir, "datasets", f"{dataset_id}.json")
        
        if not os.path.exists(dataset_path):
            print(f"Dataset not found: {dataset_id}")
            return False
        
        try:
            # Load dataset
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
            
            # Extract text content
            texts = []
            ids = []
            
            if dataset.get("source_type") == "csv" or dataset.get("source_type") == "json":
                # For tabular data, use row content
                for i, row in enumerate(dataset.get("data", [])):
                    text = " ".join([f"{k}: {v}" for k, v in row.items()])
                    texts.append(text)
                    ids.append(f"row_{i}")
            
            elif dataset.get("source_type") == "text":
                # For text data, use chunks
                for i, chunk in enumerate(dataset.get("chunks", [])):
                    texts.append(chunk)
                    ids.append(f"chunk_{i}")
            
            elif dataset.get("source_type") == "url":
                # For URL data, use content
                for i, content in enumerate(dataset.get("content", [])):
                    texts.append(content)
                    ids.append(f"content_{i}")
            
            else:
                print(f"Unsupported dataset type: {dataset.get('source_type')}")
                return False
            
            # Generate embeddings
            embeddings = []
            
            if SENTENCE_TRANSFORMERS_AVAILABLE and self.model:
                # Use real embeddings
                embeddings = self.model.encode(texts)
            else:
                # Simulate embeddings with random vectors
                print("Using simulated embeddings (random vectors)")
                # Use deterministic random based on text hash for consistency
                for text in texts:
                    text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16)
                    np.random.seed(text_hash)
                    # Generate a 384-dimensional vector (common embedding size)
                    embedding = np.random.randn(384).astype(np.float32)
                    # Normalize to unit length
                    embedding = embedding / np.linalg.norm(embedding)
                    embeddings.append(embedding)
            
            # Save embeddings
            embeddings_path = os.path.join(self.embeddings_dir, f"{dataset_id}.pkl")
            
            with open(embeddings_path, 'wb') as f:
                pickle.dump({
                    "embeddings": embeddings,
                    "ids": ids,
                    "texts": texts
                }, f)
            
            # Update dataset info
            dataset["has_embeddings"] = True
            
            with open(dataset_path, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            print(f"Generated embeddings for {len(texts)} items in dataset {dataset_id}")
            return True
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return False
    
    def similarity_search(
        self,
        dataset_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar items in a dataset.
        
        Args:
            dataset_id: Dataset identifier
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of similar items with scores
        """
        # Check if embeddings exist
        embeddings_path = os.path.join(self.embeddings_dir, f"{dataset_id}.pkl")
        
        if not os.path.exists(embeddings_path):
            print(f"Embeddings not found for dataset: {dataset_id}")
            return []
        
        try:
            # Load embeddings
            with open(embeddings_path, 'rb') as f:
                data = pickle.load(f)
            
            embeddings = data["embeddings"]
            ids = data["ids"]
            texts = data["texts"]
            
            # Generate query embedding
            if SENTENCE_TRANSFORMERS_AVAILABLE and self.model:
                # Use real embedding
                query_embedding = self.model.encode(query)
            else:
                # Simulate query embedding
                print("Using simulated query embedding (random vector)")
                query_hash = int(hashlib.md5(query.encode()).hexdigest(), 16)
                np.random.seed(query_hash)
                query_embedding = np.random.randn(384).astype(np.float32)
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Search for similar items
            if FAISS_AVAILABLE:
                # Use FAISS for efficient search
                embeddings_array = np.array(embeddings).astype(np.float32)
                index = faiss.IndexFlatL2(embeddings_array.shape[1])
                index.add(embeddings_array)
                
                distances, indices = index.search(
                    np.array([query_embedding]).astype(np.float32),
                    min(top_k, len(embeddings))
                )
                
                results = []
                for i, idx in enumerate(indices[0]):
                    results.append({
                        "id": ids[idx],
                        "text": texts[idx],
                        "score": float(1.0 / (1.0 + distances[0][i]))  # Convert distance to similarity score
                    })
                
                return results
            else:
                # Fallback to numpy for search
                print("Using numpy for similarity search (FAISS not available)")
                
                # Calculate cosine similarity
                similarities = []
                for i, embedding in enumerate(embeddings):
                    similarity = np.dot(query_embedding, embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                    )
                    similarities.append((i, float(similarity)))
                
                # Sort by similarity (descending)
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Return top k results
                results = []
                for i, score in similarities[:top_k]:
                    results.append({
                        "id": ids[i],
                        "text": texts[i],
                        "score": score
                    })
                
                return results
                
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        if self.use_ollama:
            # Use Ollama embedding model
            try:
                response = requests.post(
                    "http://127.0.0.1:11434/api/embeddings",
                    json={
                        "model": self.model_name,
                        "prompt": text
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    result = response.json()
                    embedding = np.array(result["embedding"], dtype=np.float32)
                    return embedding / np.linalg.norm(embedding)  # Normalize
                else:
                    print(f"Ollama embedding error: {response.status_code}")
                    return self._fallback_embedding(text)
            except Exception as e:
                print(f"Error getting Ollama embedding: {e}")
                return self._fallback_embedding(text)
        elif SENTENCE_TRANSFORMERS_AVAILABLE and self.model:
            # Use real embedding
            return self.model.encode(text)
        else:
            # Simulate embedding
            return self._fallback_embedding(text)
    
    def _fallback_embedding(self, text: str) -> np.ndarray:
        """Generate fallback embedding when models are unavailable"""
        print("Using simulated embedding (random vector)")
        text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(text_hash)
        embedding = np.random.randn(384).astype(np.float32)
        return embedding / np.linalg.norm(embedding)
    
    def batch_get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Array of embedding vectors
        """
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.model:
            # Use real embeddings
            return self.model.encode(texts)
        else:
            # Simulate embeddings
            print("Using simulated embeddings (random vectors)")
            embeddings = []
            for text in texts:
                text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16)
                np.random.seed(text_hash)
                embedding = np.random.randn(384).astype(np.float32)
                embeddings.append(embedding / np.linalg.norm(embedding))
            return np.array(embeddings)

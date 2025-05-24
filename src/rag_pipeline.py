#!/usr/bin/env python3
"""
RAG Pipeline Module

This module provides functionality for Retrieval-Augmented Generation (RAG),
including external data retrieval, document processing, and integration with
the vector database for context augmentation.
"""

import os
import json
import requests
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import pickle
import hashlib
import re
from urllib.parse import urlparse
import pandas as pd
import numpy as np

# For web scraping
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    print("Note: beautifulsoup4 not available. Web scraping will be limited.")
    print("To install in production: pip install beautifulsoup4")
    BS4_AVAILABLE = False

# For vector embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Note: sentence_transformers not available. Vector embeddings will be simulated.")
    print("To install in production: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Import local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vector_embedding import VectorEmbedding, SENTENCE_TRANSFORMERS_AVAILABLE, FAISS_AVAILABLE


class WebRetriever:
    """
    Class for retrieving and processing web content.
    """
    
    def __init__(
        self, 
        data_dir: str = "/home/ubuntu/data-ai-chatbot/data",
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    ):
        """
        Initialize the web retriever.
        
        Args:
            data_dir: Directory to store retrieved data
            user_agent: User agent string for HTTP requests
        """
        self.data_dir = data_dir
        self.web_data_dir = os.path.join(data_dir, "web")
        self.user_agent = user_agent
        
        os.makedirs(self.web_data_dir, exist_ok=True)
        
        # Initialize data ingestion for storing retrieved content
        from src.data_ingestion import DataIngestion
        self.data_ingestion = DataIngestion(data_dir)
    
    def retrieve_url(self, url: str, timeout: int = 10) -> Optional[str]:
        """
        Retrieve content from a URL.
        
        Args:
            url: URL to retrieve
            timeout: Request timeout in seconds
            
        Returns:
            HTML content or None if retrieval failed
        """
        try:
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Cache-Control": "max-age=0"
            }
            
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            return response.text
            
        except Exception as e:
            print(f"Error retrieving URL {url}: {e}")
            return None
    
    def extract_text_from_html(self, html: str) -> str:
        """
        Extract text content from HTML.
        
        Args:
            html: HTML content
            
        Returns:
            Extracted text
        """
        try:
            if BS4_AVAILABLE:
                soup = BeautifulSoup(html, "html.parser")
                
                # Remove script and style elements
                for script in soup(["script", "style", "header", "footer", "nav"]):
                    script.extract()
                
                # Get text
                text = soup.get_text(separator="\n")
                
                # Remove extra whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = "\n".join(chunk for chunk in chunks if chunk)
                
                return text
            else:
                # Fallback to simple regex-based extraction if BeautifulSoup is not available
                # This is less accurate but provides basic functionality
                # Remove HTML tags
                text = re.sub(r'<[^>]+>', ' ', html)
                # Remove extra whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            
        except Exception as e:
            print(f"Error extracting text from HTML: {e}")
            return ""
    
    def extract_metadata_from_html(self, html: str, url: str) -> Dict[str, str]:
        """
        Extract metadata from HTML.
        
        Args:
            html: HTML content
            url: Source URL
            
        Returns:
            Dictionary of metadata
        """
        try:
            if BS4_AVAILABLE:
                soup = BeautifulSoup(html, "html.parser")
                
                # Extract title
                title = soup.title.string if soup.title else ""
                
                # Extract description
                description = ""
                meta_desc = soup.find("meta", attrs={"name": "description"})
                if meta_desc:
                    description = meta_desc.get("content", "")
                
                # Extract keywords
                keywords = ""
                meta_keywords = soup.find("meta", attrs={"name": "keywords"})
                if meta_keywords:
                    keywords = meta_keywords.get("content", "")
                
                # Extract author
                author = ""
                meta_author = soup.find("meta", attrs={"name": "author"})
                if meta_author:
                    author = meta_author.get("content", "")
                
                # Extract publication date
                pub_date = ""
                meta_date = soup.find("meta", attrs={"name": "date"})
                if meta_date:
                    pub_date = meta_date.get("content", "")
                else:
                    meta_date = soup.find("meta", attrs={"property": "article:published_time"})
                    if meta_date:
                        pub_date = meta_date.get("content", "")
            else:
                # Fallback to basic metadata if BeautifulSoup is not available
                title = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
                title = title.group(1) if title else ""
                description = keywords = author = pub_date = ""
            
            return {
                "title": title,
                "description": description,
                "keywords": keywords,
                "author": author,
                "publication_date": pub_date,
                "url": url,
                "retrieved_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error extracting metadata from HTML: {e}")
            return {
                "title": "",
                "description": "",
                "keywords": "",
                "author": "",
                "publication_date": "",
                "url": url,
                "retrieved_date": datetime.now().isoformat()
            }
    
    def process_url(self, url: str, chunk_size: int = 1000) -> Optional[str]:
        """
        Process a URL by retrieving content, extracting text, and storing it.
        
        Args:
            url: URL to process
            chunk_size: Size of text chunks in characters
            
        Returns:
            Dataset ID or None if processing failed
        """
        try:
            # Retrieve content
            html = self.retrieve_url(url)
            
            if not html:
                return None
            
            # Extract text
            text = self.extract_text_from_html(html)
            
            if not text:
                return None
            
            # Extract metadata
            metadata = self.extract_metadata_from_html(html, url)
            
            # Generate a unique filename
            domain = urlparse(url).netloc
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{domain}_{timestamp}.txt"
            filepath = os.path.join(self.web_data_dir, filename)
            
            # Save text to file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
            
            # Save metadata
            metadata_path = os.path.join(self.web_data_dir, f"{filename}.meta.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            
            # Ingest text file
            dataset_id = self.data_ingestion.ingest_text(filepath, chunk_size=chunk_size)
            
            # Update dataset info with metadata
            with open(self.data_ingestion.registry_path, "r") as f:
                registry = json.load(f)
            
            if dataset_id in registry["datasets"]:
                registry["datasets"][dataset_id]["metadata"] = metadata
                registry["datasets"][dataset_id]["source_url"] = url
                
                with open(self.data_ingestion.registry_path, "w") as f:
                    json.dump(registry, f, indent=2)
            
            return dataset_id
            
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
            return None
    
    def search_and_retrieve(
        self, 
        query: str, 
        num_results: int = 3,
        search_api_key: Optional[str] = None,
        search_engine_id: Optional[str] = None
    ) -> List[str]:
        """
        Search for information and retrieve relevant content.
        
        Args:
            query: Search query
            num_results: Number of results to retrieve
            search_api_key: Google Custom Search API key
            search_engine_id: Google Custom Search engine ID
            
        Returns:
            List of dataset IDs for retrieved content
        """
        try:
            # Use Google Custom Search API if credentials are provided
            if search_api_key and search_engine_id:
                search_results = self._google_search(query, num_results, search_api_key, search_engine_id)
            else:
                # Fallback to a simple search (this is a placeholder)
                # In a real implementation, you might use a different search API
                print("Warning: No search API credentials provided. Using fallback search.")
                search_results = self._fallback_search(query, num_results)
            
            # Process each result
            dataset_ids = []
            for result in search_results:
                dataset_id = self.process_url(result["link"])
                if dataset_id:
                    dataset_ids.append(dataset_id)
            
            return dataset_ids
            
        except Exception as e:
            print(f"Error in search and retrieve: {e}")
            return []
    
    def _google_search(
        self, 
        query: str, 
        num_results: int,
        api_key: str,
        engine_id: str
    ) -> List[Dict[str, str]]:
        """
        Perform a search using Google Custom Search API.
        
        Args:
            query: Search query
            num_results: Number of results to retrieve
            api_key: Google Custom Search API key
            engine_id: Google Custom Search engine ID
            
        Returns:
            List of search results
        """
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": api_key,
                "cx": engine_id,
                "q": query,
                "num": min(num_results, 10)  # API limit is 10 per request
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if "items" not in data:
                return []
            
            results = []
            for item in data["items"]:
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                })
            
            return results
            
        except Exception as e:
            print(f"Error in Google search: {e}")
            return []
    
    def _fallback_search(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """
        Fallback search method when no API credentials are provided.
        This is a placeholder and should be replaced with a real implementation.
        
        Args:
            query: Search query
            num_results: Number of results to retrieve
            
        Returns:
            List of search results
        """
        # This is a placeholder. In a real implementation, you might use
        # a different search API or web scraping (with appropriate permissions).
        # For now, we'll return some dummy results for demonstration.
        return [
            {
                "title": "Wikipedia - " + query,
                "link": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                "snippet": f"Information about {query} from Wikipedia."
            },
            {
                "title": "GitHub - " + query,
                "link": f"https://github.com/search?q={query.replace(' ', '+')}",
                "snippet": f"Code repositories related to {query} on GitHub."
            },
            {
                "title": "Stack Overflow - " + query,
                "link": f"https://stackoverflow.com/search?q={query.replace(' ', '+')}",
                "snippet": f"Questions and answers about {query} on Stack Overflow."
            }
        ][:num_results]


class DocumentProcessor:
    """
    Class for processing documents into chunks suitable for embedding.
    """
    
    def __init__(self, data_dir: str = "/home/ubuntu/data-ai-chatbot/data"):
        """
        Initialize the document processor.
        
        Args:
            data_dir: Directory containing data
        """
        self.data_dir = data_dir
        from src.data_ingestion import DataIngestion
        self.data_ingestion = DataIngestion(data_dir)
    
    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = 1000, 
        overlap: int = 200
    ) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Size of chunks in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        if len(text) <= chunk_size:
            chunks.append(text)
        else:
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                
                # Try to find a sentence boundary for a cleaner cut
                if end < len(text):
                    # Look for sentence boundaries (., !, ?)
                    sentence_end = max(
                        text.rfind(". ", start, end),
                        text.rfind("! ", start, end),
                        text.rfind("? ", start, end)
                    )
                    
                    if sentence_end > start:
                        end = sentence_end + 1
                
                chunks.append(text[start:end])
                start = end - overlap
        
        return chunks


class RAGPipeline:
    """
    Class for Retrieval-Augmented Generation (RAG) pipeline.
    """
    
    def __init__(
        self, 
        data_dir: str = "/home/ubuntu/data-ai-chatbot/data",
        vector_embedding: Optional[VectorEmbedding] = None,
        max_context_length: int = 4000
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            data_dir: Directory containing data
            vector_embedding: Vector embedding instance
            max_context_length: Maximum context length in tokens
        """
        self.data_dir = data_dir
        self.max_context_length = max_context_length
        self.external_data_retrieved = False
        
        # Initialize components
        from src.data_ingestion import DataIngestion
        self.data_ingestion = DataIngestion(data_dir)
        self.vector_embedding = vector_embedding or VectorEmbedding(data_dir)
        self.web_retriever = WebRetriever(data_dir)
        self.document_processor = DocumentProcessor(data_dir)
    
    def retrieve_context(
        self, 
        query: str, 
        dataset_ids: List[str],
        top_k: int = 5,
        retrieve_external: bool = False,
        search_api_key: Optional[str] = None,
        search_engine_id: Optional[str] = None
    ) -> str:
        """
        Retrieve context for a query.
        
        Args:
            query: User query
            dataset_ids: List of dataset IDs to search
            top_k: Number of top results to include
            retrieve_external: Whether to retrieve external data
            search_api_key: Google Custom Search API key
            search_engine_id: Google Custom Search engine ID
            
        Returns:
            Retrieved context as a string
        """
        try:
            # Reset external data flag
            self.external_data_retrieved = False
            
            # Retrieve from existing datasets
            context_parts = []
            
            for dataset_id in dataset_ids:
                results = self.vector_embedding.similarity_search(
                    dataset_id=dataset_id,
                    query=query,
                    top_k=top_k
                )
                
                if results:
                    # Add dataset info
                    dataset_info = self.data_ingestion.get_dataset_info(dataset_id)
                    source_info = f"From dataset: {dataset_id}"
                    
                    if dataset_info.get("source_type"):
                        source_info += f" (Type: {dataset_info['source_type']})"
                    
                    if dataset_info.get("source_url"):
                        source_info += f" (URL: {dataset_info['source_url']})"
                    
                    context_parts.append(f"--- {source_info} ---")
                    
                    # Add results
                    for i, result in enumerate(results):
                        context_parts.append(f"[{i+1}] {result['text']} (Score: {result['score']:.4f})")
            
            # Retrieve external data if enabled and no sufficient context found
            if retrieve_external and (not context_parts or len("\n".join(context_parts)) < 500):
                print(f"Retrieving external data for query: {query}")
                
                # Search and retrieve
                external_dataset_ids = self.web_retriever.search_and_retrieve(
                    query=query,
                    num_results=3,
                    search_api_key=search_api_key,
                    search_engine_id=search_engine_id
                )
                
                if external_dataset_ids:
                    self.external_data_retrieved = True
                    
                    # Add separator
                    if context_parts:
                        context_parts.append("\n--- External Data ---")
                    
                    # Generate embeddings for new datasets
                    for dataset_id in external_dataset_ids:
                        if SENTENCE_TRANSFORMERS_AVAILABLE:
                            self.vector_embedding.generate_embeddings(dataset_id)
                        
                        # Retrieve from new dataset
                        results = self.vector_embedding.similarity_search(
                            dataset_id=dataset_id,
                            query=query,
                            top_k=top_k
                        )
                        
                        if results:
                            # Add dataset info
                            dataset_info = self.data_ingestion.get_dataset_info(dataset_id)
                            source_info = f"From external source: {dataset_id}"
                            
                            if dataset_info.get("source_url"):
                                source_info += f" (URL: {dataset_info['source_url']})"
                            
                            context_parts.append(f"--- {source_info} ---")
                            
                            # Add results
                            for i, result in enumerate(results):
                                context_parts.append(f"[{i+1}] {result['text']} (Score: {result['score']:.4f})")
            
            # Combine context parts
            context = "\n".join(context_parts)
            
            # Truncate if too long
            if len(context) > self.max_context_length:
                context = context[:self.max_context_length] + "...[truncated]"
            
            return context
            
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return f"Error retrieving context: {str(e)}"

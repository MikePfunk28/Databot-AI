#!/usr/bin/env python3
"""
Data Ingestion and Vector Embedding Module

This module provides functionality for ingesting data from various sources
and generating vector embeddings for efficient retrieval.
"""

import os
import json
import csv
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import pickle
import hashlib
from datetime import datetime

# Import local vector embedding module
from src.vector_embedding import VectorEmbedding, SENTENCE_TRANSFORMERS_AVAILABLE, FAISS_AVAILABLE


class DataIngestion:
    """
    Class for ingesting data from various sources.
    """

    def __init__(self, data_dir: str = "/home/ubuntu/data-ai-chatbot/data"):
        """
        Initialize the data ingestion module.

        Args:
            data_dir: Directory to store processed data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Create subdirectories
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.embeddings_dir = os.path.join(data_dir, "embeddings")

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)

        # Initialize data registry
        self.registry_path = os.path.join(data_dir, "data_registry.json")
        self._init_registry()

    def _init_registry(self) -> None:
        """Initialize the data registry if it doesn't exist."""
        if not os.path.exists(self.registry_path):
            with open(self.registry_path, 'w') as f:
                json.dump({
                    "datasets": {},
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)

    def _update_registry(self, dataset_id: str, metadata: Dict[str, Any]) -> None:
        """
        Update the data registry with new dataset information.

        Args:
            dataset_id: Unique identifier for the dataset
            metadata: Metadata about the dataset
        """
        try:
            with open(self.registry_path, 'r') as f:
                registry = json.load(f)

            registry["datasets"][dataset_id] = {
                **metadata,
                "last_updated": datetime.now().isoformat()
            }
            registry["last_updated"] = datetime.now().isoformat()

            with open(self.registry_path, 'w') as f:
                json.dump(registry, f, indent=2)

        except Exception as e:
            print(f"Error updating registry: {e}")

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all available datasets.

        Returns:
            List of dataset metadata
        """
        try:
            with open(self.registry_path, 'r') as f:
                registry = json.load(f)

            return [
                {"id": dataset_id, **metadata}
                for dataset_id, metadata in registry["datasets"].items()
            ]
        except Exception as e:
            print(f"Error listing datasets: {e}")
            return []

    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Dataset metadata or None if not found
        """
        try:
            with open(self.registry_path, 'r') as f:
                registry = json.load(f)

            return registry["datasets"].get(dataset_id)
        except Exception as e:
            print(f"Error getting dataset info: {e}")
            return None

    def _generate_dataset_id(self, file_path: str, source_type: str) -> str:
        """
        Generate a unique dataset ID based on file path and type.

        Args:
            file_path: Path to the data file
            source_type: Type of data source

        Returns:
            Unique dataset ID
        """
        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Create a hash of the file path for uniqueness
        path_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]

        return f"{name_without_ext}_{source_type}_{path_hash}_{timestamp}"

    def ingest_csv(self, file_path: str, delimiter: str = ',') -> str:
        """
        Ingest data from a CSV file.

        Args:
            file_path: Path to the CSV file
            delimiter: CSV delimiter

        Returns:
            Dataset ID
        """
        try:
            # Generate dataset ID
            dataset_id = self._generate_dataset_id(file_path, "csv")

            # Read CSV file
            df = pd.read_csv(file_path, delimiter=delimiter)

            # Save raw data
            raw_path = os.path.join(self.raw_dir, f"{dataset_id}.csv")
            df.to_csv(raw_path, index=False)

            # Save processed data
            processed_path = os.path.join(
                self.processed_dir, f"{dataset_id}.parquet")
            df.to_parquet(processed_path, index=False)

            # Extract metadata
            metadata = {
                "source_type": "csv",
                "original_path": file_path,
                "raw_path": raw_path,
                "processed_path": processed_path,
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "has_embeddings": False,
                "sample_data": df.head(5).to_dict(orient="records")
            }

            # Update registry
            self._update_registry(dataset_id, metadata)

            return dataset_id

        except Exception as e:
            print(f"Error ingesting CSV: {e}")
            raise

    def ingest_json(self, file_path: str) -> str:
        """
        Ingest data from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Dataset ID
        """
        try:
            # Generate dataset ID
            dataset_id = self._generate_dataset_id(file_path, "json")

            # Read JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Convert to DataFrame if possible
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                df = pd.DataFrame(data)
            else:
                # Handle nested JSON by flattening
                df = pd.json_normalize(data)

            # Save raw data
            raw_path = os.path.join(self.raw_dir, f"{dataset_id}.json")
            with open(raw_path, 'w') as f:
                json.dump(data, f, indent=2)

            # Save processed data
            processed_path = os.path.join(
                self.processed_dir, f"{dataset_id}.parquet")
            df.to_parquet(processed_path, index=False)

            # Extract metadata
            metadata = {
                "source_type": "json",
                "original_path": file_path,
                "raw_path": raw_path,
                "processed_path": processed_path,
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "has_embeddings": False,
                "sample_data": df.head(5).to_dict(orient="records") if not df.empty else {}
            }

            # Update registry
            self._update_registry(dataset_id, metadata)

            return dataset_id

        except Exception as e:
            print(f"Error ingesting JSON: {e}")
            raise

    def ingest_excel(self, file_path: str, sheet_name: Optional[str] = None) -> str:
        """
        Ingest data from an Excel file.

        Args:
            file_path: Path to the Excel file
            sheet_name: Name of the sheet to read (None for all sheets)

        Returns:
            Dataset ID
        """
        try:
            # Generate dataset ID
            dataset_id = self._generate_dataset_id(file_path, "excel")

            # Read Excel file
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheets = [sheet_name]
            else:
                # Read all sheets
                excel_data = pd.read_excel(file_path, sheet_name=None)
                sheets = list(excel_data.keys())

                # Combine all sheets or use the first one
                if len(sheets) == 1:
                    df = excel_data[sheets[0]]
                else:
                    # Create a dictionary of DataFrames
                    df_dict = excel_data

                    # Use the first sheet for metadata
                    df = excel_data[sheets[0]]

            # Save raw data (copy the original file)
            raw_path = os.path.join(self.raw_dir, f"{dataset_id}.xlsx")
            import shutil
            shutil.copy2(file_path, raw_path)

            # Save processed data
            processed_path = os.path.join(
                self.processed_dir, f"{dataset_id}.parquet")

            if sheet_name or len(sheets) == 1:
                df.to_parquet(processed_path, index=False)
            else:
                # Save each sheet separately
                os.makedirs(processed_path.replace(
                    ".parquet", ""), exist_ok=True)

                for sheet, sheet_df in df_dict.items():
                    sheet_path = processed_path.replace(
                        ".parquet", f"/{sheet}.parquet")
                    sheet_df.to_parquet(sheet_path, index=False)

            # Extract metadata
            metadata = {
                "source_type": "excel",
                "original_path": file_path,
                "raw_path": raw_path,
                "processed_path": processed_path,
                "sheets": sheets,
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "has_embeddings": False,
                "sample_data": df.head(5).to_dict(orient="records")
            }

            # Update registry
            self._update_registry(dataset_id, metadata)

            return dataset_id

        except Exception as e:
            print(f"Error ingesting Excel: {e}")
            raise

    def ingest_sqlite(self, file_path: str, table_name: Optional[str] = None) -> str:
        """
        Ingest data from a SQLite database.

        Args:
            file_path: Path to the SQLite database
            table_name: Name of the table to read (None for all tables)

        Returns:
            Dataset ID
        """
        try:
            # Generate dataset ID
            dataset_id = self._generate_dataset_id(file_path, "sqlite")

            # Connect to the database
            conn = sqlite3.connect(file_path)

            # Get list of tables
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]

            if not tables:
                raise ValueError("No tables found in the database")

            # Determine which tables to read
            if table_name:
                if table_name not in tables:
                    raise ValueError(
                        f"Table '{table_name}' not found in the database")
                tables_to_read = [table_name]
            else:
                tables_to_read = tables

            # Read tables
            table_data = {}
            for table in tables_to_read:
                table_data[table] = pd.read_sql_query(
                    f"SELECT * FROM {table}", conn)

            # Close connection
            conn.close()

            # Save raw data (copy the original file)
            raw_path = os.path.join(self.raw_dir, f"{dataset_id}.db")
            import shutil
            shutil.copy2(file_path, raw_path)

            # Save processed data
            processed_dir = os.path.join(self.processed_dir, dataset_id)
            os.makedirs(processed_dir, exist_ok=True)

            for table, df in table_data.items():
                table_path = os.path.join(processed_dir, f"{table}.parquet")
                df.to_parquet(table_path, index=False)

            # Use the first table for metadata
            first_table = tables_to_read[0]
            df = table_data[first_table]

            # Extract metadata
            metadata = {
                "source_type": "sqlite",
                "original_path": file_path,
                "raw_path": raw_path,
                "processed_path": processed_dir,
                "tables": tables,
                "tables_read": tables_to_read,
                "table_rows": {table: len(df) for table, df in table_data.items()},
                "table_columns": {table: df.columns.tolist() for table, df in table_data.items()},
                "has_embeddings": False,
                "sample_data": {table: df.head(5).to_dict(orient="records") for table, df in table_data.items()}
            }

            # Update registry
            self._update_registry(dataset_id, metadata)

            return dataset_id

        except Exception as e:
            print(f"Error ingesting SQLite: {e}")
            raise

    def ingest_text(self, file_path: str, chunk_size: int = 1000) -> str:
        """
        Ingest data from a text file, chunking it for analysis.

        Args:
            file_path: Path to the text file
            chunk_size: Size of text chunks in characters

        Returns:
            Dataset ID
        """
        try:
            # Generate dataset ID
            dataset_id = self._generate_dataset_id(file_path, "text")

            # Read text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Chunk the text
            chunks = []
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                if chunk.strip():  # Skip empty chunks
                    chunks.append(chunk)

            # Create DataFrame
            df = pd.DataFrame({
                "chunk_id": range(len(chunks)),
                "text": chunks,
                "start_char": [i * chunk_size for i in range(len(chunks))],
                "end_char": [(i + 1) * chunk_size for i in range(len(chunks))]
            })

            # Save raw data (copy the original file)
            raw_path = os.path.join(self.raw_dir, f"{dataset_id}.txt")
            import shutil
            shutil.copy2(file_path, raw_path)

            # Save processed data
            processed_path = os.path.join(
                self.processed_dir, f"{dataset_id}.parquet")
            df.to_parquet(processed_path, index=False)

            # Extract metadata
            metadata = {
                "source_type": "text",
                "original_path": file_path,
                "raw_path": raw_path,
                "processed_path": processed_path,
                "num_chunks": len(chunks),
                "chunk_size": chunk_size,
                "total_chars": len(text),
                "has_embeddings": False,
                "sample_data": df.head(5).to_dict(orient="records")
            }

            # Update registry
            self._update_registry(dataset_id, metadata)

            return dataset_id

        except Exception as e:
            print(f"Error ingesting text: {e}")
            raise

    def ingest_url(self, url: str) -> str:
        """
        Ingest data from a URL.

        Args:
            url: URL to fetch data from

        Returns:
            Dataset ID
        """
        try:
            import requests
            from bs4 import BeautifulSoup

            # Generate dataset ID
            dataset_id = self._generate_dataset_id(url, "url")

            # Fetch URL content
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Determine content type and process accordingly
            content_type = response.headers.get('content-type', '').lower()

            if 'application/json' in content_type:
                # Handle JSON URLs
                data = response.json()
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    # If it's a single object, make it a list
                    df = pd.DataFrame([data])
                else:
                    # Convert to string and treat as text
                    df = pd.DataFrame({'content': [str(data)]})

            elif 'text/csv' in content_type or url.endswith('.csv'):
                # Handle CSV URLs
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))

            elif 'text/html' in content_type:
                # Handle HTML pages - extract text content
                soup = BeautifulSoup(response.content, 'html.parser')

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Get text content
                text = soup.get_text()
                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip()
                          for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)

                # Create DataFrame with chunked text
                chunk_size = 1000
                chunks = []
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i+chunk_size]
                    if chunk.strip():
                        chunks.append(chunk)

                df = pd.DataFrame({
                    'chunk_id': range(len(chunks)),
                    'text': chunks,
                    'url': [url] * len(chunks)
                })

            else:
                # Default: treat as plain text
                text = response.text
                df = pd.DataFrame({
                    'content': [text],
                    'url': [url]
                })

            # Save raw data
            raw_path = os.path.join(self.raw_dir, f"{dataset_id}.json")
            with open(raw_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'url': url,
                    'content_type': content_type,
                    'status_code': response.status_code,
                    'content': response.text
                }, f, indent=2)

            # Save processed data
            processed_path = os.path.join(
                self.processed_dir, f"{dataset_id}.parquet")
            df.to_parquet(processed_path, index=False)

            # Extract metadata
            metadata = {
                "source_type": "url",
                "original_path": url,
                "raw_path": raw_path,
                "processed_path": processed_path,
                "content_type": content_type,
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "has_embeddings": False,
                "sample_data": df.head(5).to_dict(orient="records")
            }

            # Update registry
            self._update_registry(dataset_id, metadata)

            return dataset_id

        except Exception as e:
            print(f"Error ingesting URL: {e}")
            raise

    def load_dataset(self, dataset_id: str) -> pd.DataFrame:
        """
        Load a dataset by ID.

        Args:
            dataset_id: Dataset identifier

        Returns:
            DataFrame containing the dataset
        """
        try:
            # Get dataset info
            dataset_info = self.get_dataset_info(dataset_id)

            if not dataset_info:
                raise ValueError(f"Dataset '{dataset_id}' not found")

            # Load based on source type
            source_type = dataset_info["source_type"]
            processed_path = dataset_info["processed_path"]

            if source_type in ["csv", "json", "text"]:
                # Simple file types
                return pd.read_parquet(processed_path)

            elif source_type == "excel":
                # Check if multiple sheets
                if os.path.isdir(processed_path.replace(".parquet", "")):
                    # Return dictionary of DataFrames
                    sheets = dataset_info["sheets"]
                    result = {}

                    for sheet in sheets:
                        sheet_path = processed_path.replace(
                            ".parquet", f"/{sheet}.parquet")
                        result[sheet] = pd.read_parquet(sheet_path)

                    return result
                else:
                    # Single sheet
                    return pd.read_parquet(processed_path)

            elif source_type == "sqlite":
                # Return dictionary of DataFrames
                tables = dataset_info["tables_read"]
                result = {}

                for table in tables:
                    table_path = os.path.join(
                        processed_path, f"{table}.parquet")
                    result[table] = pd.read_parquet(table_path)

                return result

            else:
                raise ValueError(f"Unsupported source type: {source_type}")

        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def get_dataset_schema(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get the schema of a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Dictionary containing schema information
        """
        try:
            # Get dataset info
            dataset_info = self.get_dataset_info(dataset_id)

            if not dataset_info:
                raise ValueError(f"Dataset '{dataset_id}' not found")

            # Extract schema based on source type
            source_type = dataset_info["source_type"]

            if source_type in ["csv", "json", "text"]:
                # Simple file types
                return {
                    "columns": dataset_info.get("columns", []),
                    "dtypes": dataset_info.get("dtypes", {}),
                    "num_rows": dataset_info.get("num_rows", 0)
                }

            elif source_type == "excel":
                # Check if multiple sheets
                if isinstance(dataset_info.get("sheets"), list) and len(dataset_info["sheets"]) > 1:
                    # Load each sheet to get schema
                    df_dict = self.load_dataset(dataset_id)

                    return {
                        "sheets": dataset_info["sheets"],
                        "sheet_schemas": {
                            sheet: {
                                "columns": df.columns.tolist(),
                                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                                "num_rows": len(df)
                            }
                            for sheet, df in df_dict.items()
                        }
                    }
                else:
                    # Single sheet
                    return {
                        "columns": dataset_info.get("columns", []),
                        "dtypes": dataset_info.get("dtypes", {}),
                        "num_rows": dataset_info.get("num_rows", 0)
                    }

            elif source_type == "sqlite":
                # Return schema for each table
                return {
                    "tables": dataset_info["tables"],
                    "table_rows": dataset_info.get("table_rows", {}),
                    "table_columns": dataset_info.get("table_columns", {})
                }

            else:
                raise ValueError(f"Unsupported source type: {source_type}")

        except Exception as e:
            print(f"Error getting dataset schema: {e}")
            raise

    def get_dataset_sample(self, dataset_id: str, num_rows: int = 5) -> Dict[str, Any]:
        """
        Get a sample of a dataset.

        Args:
            dataset_id: Dataset identifier
            num_rows: Number of rows to sample

        Returns:
            Dictionary containing sample data
        """
        try:
            # Get dataset info
            dataset_info = self.get_dataset_info(dataset_id)

            if not dataset_info:
                raise ValueError(f"Dataset '{dataset_id}' not found")

            # Load dataset
            data = self.load_dataset(dataset_id)

            # Extract sample based on source type
            source_type = dataset_info["source_type"]

            if source_type in ["csv", "json", "text"]:
                # Simple file types
                return data.head(num_rows).to_dict(orient="records")

            elif source_type == "excel":
                # Check if multiple sheets
                if isinstance(data, dict):
                    # Return sample for each sheet
                    return {
                        sheet: df.head(num_rows).to_dict(orient="records")
                        for sheet, df in data.items()
                    }
                else:
                    # Single sheet
                    return data.head(num_rows).to_dict(orient="records")

            elif source_type == "sqlite":
                # Return sample for each table
                return {
                    table: df.head(num_rows).to_dict(orient="records")
                    for table, df in data.items()
                }

            else:
                raise ValueError(f"Unsupported source type: {source_type}")

        except Exception as e:
            print(f"Error getting dataset sample: {e}")
            raise


class VectorEmbedding:
    """
    Class for generating and managing vector embeddings.
    """

    def __init__(
        self,
        data_dir: str = "/home/ubuntu/data-ai-chatbot/data",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the vector embedding module.

        Args:
            data_dir: Directory containing data
            model_name: Name of the embedding model
        """
        self.data_dir = data_dir
        self.embeddings_dir = os.path.join(data_dir, "embeddings")
        self.registry_path = os.path.join(data_dir, "data_registry.json")
        self.model_name = model_name

        os.makedirs(self.embeddings_dir, exist_ok=True)

        # Initialize embedding model
        try:
            self.model = SentenceTransformer(model_name)
            print(f"Loaded embedding model: {model_name}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            self.model = None

    def _update_registry(self, dataset_id: str, embedding_info: Dict[str, Any]) -> None:
        """
        Update the data registry with embedding information.

        Args:
            dataset_id: Dataset identifier
            embedding_info: Information about the embeddings
        """
        try:
            with open(self.registry_path, 'r') as f:
                registry = json.load(f)

            if dataset_id in registry["datasets"]:
                registry["datasets"][dataset_id]["has_embeddings"] = True
                registry["datasets"][dataset_id]["embedding_info"] = embedding_info
                registry["datasets"][dataset_id]["last_updated"] = datetime.now(
                ).isoformat()
                registry["last_updated"] = datetime.now().isoformat()

                with open(self.registry_path, 'w') as f:
                    json.dump(registry, f, indent=2)
            else:
                print(f"Dataset '{dataset_id}' not found in registry")

        except Exception as e:
            print(f"Error updating registry with embedding info: {e}")

    def generate_embeddings(
        self,
        dataset_id: str,
        text_column: str = "text",
        id_column: Optional[str] = None,
        batch_size: int = 32
    ) -> str:
        """
        Generate embeddings for a dataset.

        Args:
            dataset_id: Dataset identifier
            text_column: Column containing text to embed
            id_column: Column containing unique identifiers
            batch_size: Batch size for embedding generation

        Returns:
            Path to the embeddings file
        """
        try:
            if not self.model:
                raise ValueError("Embedding model not initialized")

            # Load dataset
            data_ingestion = DataIngestion(self.data_dir)
            dataset_info = data_ingestion.get_dataset_info(dataset_id)

            if not dataset_info:
                raise ValueError(f"Dataset '{dataset_id}' not found")

            data = data_ingestion.load_dataset(dataset_id)

            # Handle different dataset types
            source_type = dataset_info["source_type"]

            if source_type in ["csv", "json"]:
                # Check if text column exists
                if text_column not in data.columns:
                    raise ValueError(
                        f"Text column '{text_column}' not found in dataset")

                # Generate embeddings
                texts = data[text_column].tolist()

                # Generate in batches to avoid memory issues
                embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_embeddings = self.model.encode(batch_texts)
                    embeddings.extend(batch_embeddings)

                embeddings = np.array(embeddings)

                # Create ID column if not provided
                if id_column and id_column in data.columns:
                    ids = data[id_column].tolist()
                else:
                    ids = list(range(len(texts)))

                # Save embeddings
                embeddings_path = os.path.join(
                    self.embeddings_dir, f"{dataset_id}_embeddings.pkl")

                with open(embeddings_path, 'wb') as f:
                    pickle.dump({
                        "embeddings": embeddings,
                        "texts": texts,
                        "ids": ids,
                        "model_name": self.model_name
                    }, f)

                # Create FAISS index for fast retrieval
                index_path = os.path.join(
                    self.embeddings_dir, f"{dataset_id}_index.faiss")

                # Normalize embeddings for cosine similarity
                faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
                normalized_embeddings = embeddings.copy()
                faiss.normalize_L2(normalized_embeddings)
                faiss_index.add(normalized_embeddings)

                faiss.write_index(faiss_index, index_path)

                # Update registry
                embedding_info = {
                    "embeddings_path": embeddings_path,
                    "index_path": index_path,
                    "model_name": self.model_name,
                    "num_embeddings": len(embeddings),
                    "embedding_dim": embeddings.shape[1],
                    "text_column": text_column,
                    "id_column": id_column
                }

                self._update_registry(dataset_id, embedding_info)

                return embeddings_path

            elif source_type == "text":
                # Text datasets already have a text column
                if "text" not in data.columns:
                    raise ValueError("Text column not found in text dataset")

                # Generate embeddings
                texts = data["text"].tolist()

                # Generate in batches to avoid memory issues
                embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_embeddings = self.model.encode(batch_texts)
                    embeddings.extend(batch_embeddings)

                embeddings = np.array(embeddings)

                # Use chunk_id as ID
                ids = data["chunk_id"].tolist()

                # Save embeddings
                embeddings_path = os.path.join(
                    self.embeddings_dir, f"{dataset_id}_embeddings.pkl")

                with open(embeddings_path, 'wb') as f:
                    pickle.dump({
                        "embeddings": embeddings,
                        "texts": texts,
                        "ids": ids,
                        "model_name": self.model_name
                    }, f)

                # Create FAISS index for fast retrieval
                index_path = os.path.join(
                    self.embeddings_dir, f"{dataset_id}_index.faiss")

                # Normalize embeddings for cosine similarity
                faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
                normalized_embeddings = embeddings.copy()
                faiss.normalize_L2(normalized_embeddings)
                faiss_index.add(normalized_embeddings)

                faiss.write_index(faiss_index, index_path)

                # Update registry
                embedding_info = {
                    "embeddings_path": embeddings_path,
                    "index_path": index_path,
                    "model_name": self.model_name,
                    "num_embeddings": len(embeddings),
                    "embedding_dim": embeddings.shape[1],
                    "text_column": "text",
                    "id_column": "chunk_id"
                }

                self._update_registry(dataset_id, embedding_info)

                return embeddings_path

            elif source_type in ["excel", "sqlite"]:
                # These types return dictionaries of DataFrames
                if isinstance(data, dict):
                    # Process each sheet/table separately
                    results = {}

                    for name, df in data.items():
                        # Check if text column exists
                        if text_column not in df.columns:
                            print(
                                f"Warning: Text column '{text_column}' not found in {name}, skipping")
                            continue

                        # Generate embeddings
                        texts = df[text_column].tolist()

                        # Generate in batches to avoid memory issues
                        embeddings = []
                        for i in range(0, len(texts), batch_size):
                            batch_texts = texts[i:i+batch_size]
                            batch_embeddings = self.model.encode(batch_texts)
                            embeddings.extend(batch_embeddings)

                        embeddings = np.array(embeddings)

                        # Create ID column if not provided
                        if id_column and id_column in df.columns:
                            ids = df[id_column].tolist()
                        else:
                            ids = list(range(len(texts)))

                        # Save embeddings
                        embeddings_path = os.path.join(
                            self.embeddings_dir, f"{dataset_id}_{name}_embeddings.pkl")

                        with open(embeddings_path, 'wb') as f:
                            pickle.dump({
                                "embeddings": embeddings,
                                "texts": texts,
                                "ids": ids,
                                "model_name": self.model_name
                            }, f)

                        # Create FAISS index for fast retrieval
                        index_path = os.path.join(
                            self.embeddings_dir, f"{dataset_id}_{name}_index.faiss")

                        # Normalize embeddings for cosine similarity
                        faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
                        normalized_embeddings = embeddings.copy()
                        faiss.normalize_L2(normalized_embeddings)
                        faiss_index.add(normalized_embeddings)

                        faiss.write_index(faiss_index, index_path)

                        results[name] = {
                            "embeddings_path": embeddings_path,
                            "index_path": index_path,
                            "num_embeddings": len(embeddings),
                            "embedding_dim": embeddings.shape[1]
                        }

                    # Update registry
                    embedding_info = {
                        "model_name": self.model_name,
                        "text_column": text_column,
                        "id_column": id_column,
                        "components": results
                    }

                    self._update_registry(dataset_id, embedding_info)

                    return json.dumps(results)
                else:
                    # Single sheet
                    # Check if text column exists
                    if text_column not in data.columns:
                        raise ValueError(
                            f"Text column '{text_column}' not found in dataset")

                    # Generate embeddings
                    texts = data[text_column].tolist()

                    # Generate in batches to avoid memory issues
                    embeddings = []
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i:i+batch_size]
                        batch_embeddings = self.model.encode(batch_texts)
                        embeddings.extend(batch_embeddings)

                    embeddings = np.array(embeddings)

                    # Create ID column if not provided
                    if id_column and id_column in data.columns:
                        ids = data[id_column].tolist()
                    else:
                        ids = list(range(len(texts)))

                    # Save embeddings
                    embeddings_path = os.path.join(
                        self.embeddings_dir, f"{dataset_id}_embeddings.pkl")

                    with open(embeddings_path, 'wb') as f:
                        pickle.dump({
                            "embeddings": embeddings,
                            "texts": texts,
                            "ids": ids,
                            "model_name": self.model_name
                        }, f)

                    # Create FAISS index for fast retrieval
                    index_path = os.path.join(
                        self.embeddings_dir, f"{dataset_id}_index.faiss")

                    # Normalize embeddings for cosine similarity
                    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
                    normalized_embeddings = embeddings.copy()
                    faiss.normalize_L2(normalized_embeddings)
                    faiss_index.add(normalized_embeddings)

                    faiss.write_index(faiss_index, index_path)

                    # Update registry
                    embedding_info = {
                        "embeddings_path": embeddings_path,
                        "index_path": index_path,
                        "model_name": self.model_name,
                        "num_embeddings": len(embeddings),
                        "embedding_dim": embeddings.shape[1],
                        "text_column": text_column,
                        "id_column": id_column
                    }

                    self._update_registry(dataset_id, embedding_info)

                    return embeddings_path

            else:
                raise ValueError(f"Unsupported source type: {source_type}")

        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise

    def search(
        self,
        dataset_id: str,
        query: str,
        top_k: int = 5,
        component: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar texts in a dataset.

        Args:
            dataset_id: Dataset identifier
            query: Query text
            top_k: Number of results to return
            component: Component name for multi-component datasets

        Returns:
            List of search results
        """
        try:
            if not self.model:
                raise ValueError("Embedding model not initialized")

            # Get dataset info
            with open(self.registry_path, 'r') as f:
                registry = json.load(f)

            if dataset_id not in registry["datasets"]:
                raise ValueError(f"Dataset '{dataset_id}' not found")

            dataset_info = registry["datasets"][dataset_id]

            if not dataset_info.get("has_embeddings", False):
                raise ValueError(
                    f"Dataset '{dataset_id}' does not have embeddings")

            # Get embedding info
            embedding_info = dataset_info.get("embedding_info", {})

            # Handle multi-component datasets
            if "components" in embedding_info:
                if not component:
                    raise ValueError(
                        "Component name required for multi-component datasets")

                if component not in embedding_info["components"]:
                    raise ValueError(
                        f"Component '{component}' not found in dataset")

                component_info = embedding_info["components"][component]
                embeddings_path = component_info["embeddings_path"]
                index_path = component_info["index_path"]
            else:
                embeddings_path = embedding_info["embeddings_path"]
                index_path = embedding_info["index_path"]

            # Load embeddings
            with open(embeddings_path, 'rb') as f:
                embeddings_data = pickle.load(f)

            # Load index
            faiss_index = faiss.read_index(index_path)

            # Generate query embedding
            query_embedding = self.model.encode([query])[0]

            # Normalize query embedding
            query_embedding_normalized = query_embedding.copy().reshape(1, -1)
            faiss.normalize_L2(query_embedding_normalized)

            # Search
            distances, indices = faiss_index.search(
                query_embedding_normalized, top_k)

            # Format results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(embeddings_data["texts"]):
                    results.append({
                        "id": embeddings_data["ids"][idx],
                        "text": embeddings_data["texts"][idx],
                        "score": float(distances[0][i])
                    })

            return results

        except Exception as e:
            print(f"Error searching embeddings: {e}")
            raise

    def get_embedding_info(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get information about embeddings for a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Dictionary containing embedding information
        """
        try:
            # Get dataset info
            with open(self.registry_path, 'r') as f:
                registry = json.load(f)

            if dataset_id not in registry["datasets"]:
                raise ValueError(f"Dataset '{dataset_id}' not found")

            dataset_info = registry["datasets"][dataset_id]

            if not dataset_info.get("has_embeddings", False):
                return {"has_embeddings": False}

            return dataset_info.get("embedding_info", {})

        except Exception as e:
            print(f"Error getting embedding info: {e}")
            raise


# Example usage
if __name__ == "__main__":
    # Create data ingestion instance
    data_ingestion = DataIngestion()

    # Create sample data
    sample_data = pd.DataFrame({
        "id": range(10),
        "text": [
            "This is a sample text for testing embeddings.",
            "Vector embeddings are useful for semantic search.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models can process natural language.",
            "Data analysis helps extract insights from information.",
            "Python is a popular programming language for data science.",
            "Neural networks are inspired by the human brain.",
            "Transformers have revolutionized natural language processing.",
            "Sentence embeddings capture semantic meaning.",
            "Retrieval augmented generation improves model responses."
        ]
    })

    # Save sample data
    sample_path = "/home/ubuntu/data-ai-chatbot/data/sample_data.csv"
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    sample_data.to_csv(sample_path, index=False)

    # Ingest sample data
    dataset_id = data_ingestion.ingest_csv(sample_path)
    print(f"Ingested dataset: {dataset_id}")

    # Create vector embedding instance
    vector_embedding = VectorEmbedding()

    # Generate embeddings
    embeddings_path = vector_embedding.generate_embeddings(
        dataset_id, text_column="text", id_column="id")
    print(f"Generated embeddings: {embeddings_path}")

    # Search embeddings
    results = vector_embedding.search(dataset_id, "semantic search", top_k=3)
    print("Search results:")
    for result in results:
        print(f"  {result['score']:.4f}: {result['text']}")

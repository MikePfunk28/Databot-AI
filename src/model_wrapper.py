#!/usr/bin/env python3
"""
Model Wrapper for Data AI Chatbot

This module provides a wrapper around the Ollama model to inject system prompts,
manage conversations, and handle different types of data analysis tasks.
"""

import os
import json
import subprocess
import requests
from typing import Dict, List, Optional, Union, Any


class ModelWrapper:
    """
    A wrapper for the Ollama model that injects system prompts and manages
    conversations for data analysis tasks.
    """

    def __init__(
        self,
        model_name: str = "databot",
        system_prompt_path: str = "/home/ubuntu/data-ai-chatbot/config/system_prompt.md",
        conversation_memory: bool = True,
        max_context_length: int = 4096,
        temperature: float = 0.7,
        api_url: str = "http://localhost:11434/api"
    ):
        """
        Initialize the model wrapper.

        Args:
            model_name: Name of the Ollama model to use
            system_prompt_path: Path to the system prompt file
            conversation_memory: Whether to maintain conversation history
            max_context_length: Maximum context length for the model
            temperature: Temperature for model generation
            api_url: URL for the Ollama API
        """
        self.model_name = model_name
        self.system_prompt_path = system_prompt_path
        self.conversation_memory = conversation_memory
        self.max_context_length = max_context_length
        self.temperature = temperature
        self.api_url = api_url

        # Load system prompt
        self.system_prompt = self._load_system_prompt()

        # Initialize conversation history
        self.conversation_history = []

        # Check if model exists, create if not
        self._ensure_model_exists()

    def _load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        try:
            with open(self.system_prompt_path, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading system prompt: {e}")
            return "You are a helpful AI assistant specialized in data analysis."

    def _ensure_model_exists(self) -> None:
        """
        Check if the model with custom system prompt exists in Ollama.
        If not, create it.
        """
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True
            )

            if self.model_name not in result.stdout:
                print(
                    f"Warning: Model '{self.model_name}' not found in Ollama. Available models:")
                print(result.stdout)
                # Try to use the model anyway - Ollama might pull it automatically
            else:
                print(f"Using existing model: {self.model_name}")

        except Exception as e:
            print(f"Error checking model exists: {e}")
            # Fall back to base model
            print(f"Falling back to base model: {self.model_name}")

    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: User prompt
            context: Additional context (e.g., from RAG)

        Returns:
            Model response
        """
        # Prepare the full prompt with context if provided
        full_prompt = prompt
        if context:
            full_prompt = f"Context information:\n{context}\n\nUser query: {prompt}"

        # Add conversation history if enabled
        if self.conversation_memory and self.conversation_history:
            history_text = "\n".join([
                f"User: {item['prompt']}\nAssistant: {item['response']}"
                for item in self.conversation_history
            ])
            full_prompt = f"Previous conversation:\n{history_text}\n\nNew query: {full_prompt}"

        try:
            # Call Ollama API
            response = requests.post(
                f"{self.api_url}/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_ctx": self.max_context_length
                    }
                }
            )

            response.raise_for_status()
            result = response.json()

            # Extract response
            model_response = result.get("response", "")

            # Update conversation history
            if self.conversation_memory:
                self.conversation_history.append({
                    "prompt": prompt,
                    "response": model_response
                })

                # Limit history size
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]

            return model_response

        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

    def analyze_data(self, prompt: str, data_context: str) -> str:
        """
        Analyze data based on the prompt.

        Args:
            prompt: User prompt about the data
            data_context: Context about the data (schema, sample, etc.)

        Returns:
            Analysis result
        """
        # Prepare a specialized prompt for data analysis
        analysis_prompt = f"""
Data Analysis Request:
{prompt}

Data Information:
{data_context}

Please analyze this data and provide insights.
"""
        return self.generate_response(analysis_prompt)

    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []

    def update_system_prompt(self, new_prompt: str) -> None:
        """
        Update the system prompt and recreate the model.

        Args:
            new_prompt: New system prompt
        """
        # Save new prompt to file
        with open(self.system_prompt_path, 'w') as f:
            f.write(new_prompt)

        # Reload prompt
        self.system_prompt = self._load_system_prompt()

        # Recreate model
        model_id = f"data-insight-{self.model_name.split(':')[0]}"

        # Create Modelfile
        modelfile_content = f"""
FROM {self.model_name.split('-')[0]}:{self.model_name.split(':')[1] if ':' in self.model_name else '1.5b'}
SYSTEM {self.system_prompt}
PARAMETER temperature {self.temperature}
PARAMETER num_ctx {self.max_context_length}
"""

        modelfile_path = os.path.join(os.path.dirname(
            self.system_prompt_path), "Modelfile")
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)

        # Create model
        try:
            subprocess.run(
                ["ollama", "delete", model_id],
                capture_output=True
            )

            subprocess.run(
                ["ollama", "create", model_id, "-f", modelfile_path],
                check=True
            )

            print(f"Updated custom model: {model_id}")
            self.model_name = model_id
        except Exception as e:
            print(f"Error updating model: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_name": self.model_name,
            "system_prompt_path": self.system_prompt_path,
            "conversation_memory": self.conversation_memory,
            "max_context_length": self.max_context_length,
            "temperature": self.temperature,
            "conversation_history_length": len(self.conversation_history)
        }


class GeminiWrapper:
    """
    A wrapper for the Gemini API as a fallback option.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-pro",
        temperature: float = 0.7,
        max_output_tokens: int = 2048
    ):
        """
        Initialize the Gemini wrapper.

        Args:
            api_key: Gemini API key
            model_name: Gemini model name
            temperature: Temperature for model generation
            max_output_tokens: Maximum output tokens
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        # Check if API key is available
        if not self.api_key:
            print("Warning: Gemini API key not provided. Fallback will not work.")

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response from Gemini.

        Args:
            prompt: User prompt
            system_prompt: System prompt

        Returns:
            Model response
        """
        if not self.api_key:
            return "Error: Gemini API key not provided."

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)

            # Configure the model
            generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            }

            # Create the model
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config
            )

            # Generate response
            if system_prompt:
                response = model.generate_content(
                    [
                        {"role": "system", "parts": [system_prompt]},
                        {"role": "user", "parts": [prompt]}
                    ]
                )
            else:
                response = model.generate_content(prompt)

            return response.text

        except ImportError:
            return "Error: google-generativeai package not installed. Run 'pip install google-generativeai'."
        except Exception as e:
            return f"Error generating response from Gemini: {str(e)}"


class ModelManager:
    """
    Manager class to handle multiple models and fallback options.
    """

    def __init__(
        self,
        primary_model: ModelWrapper,
        fallback_model: Optional[GeminiWrapper] = None,
        fallback_threshold: int = 3,
        auto_fallback: bool = True
    ):
        """
        Initialize the model manager.

        Args:
            primary_model: Primary model wrapper
            fallback_model: Fallback model wrapper
            fallback_threshold: Number of retries before fallback
            auto_fallback: Whether to automatically fall back
        """
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.fallback_threshold = fallback_threshold
        self.auto_fallback = auto_fallback
        self.failure_count = 0

    def generate_response(self, prompt: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response, with fallback if needed.

        Args:
            prompt: User prompt
            context: Additional context

        Returns:
            Dictionary with response and metadata
        """
        try:
            # Try primary model
            response = self.primary_model.generate_response(prompt, context)

            # Reset failure count on success
            self.failure_count = 0

            return {
                "response": response,
                "model_used": self.primary_model.model_name,
                "fallback_used": False
            }

        except Exception as e:
            print(f"Primary model error: {e}")
            self.failure_count += 1

            # Check if we should fall back
            if self.auto_fallback and self.failure_count >= self.fallback_threshold and self.fallback_model:
                try:
                    # Load system prompt for fallback
                    system_prompt = self.primary_model.system_prompt

                    # Try fallback model
                    response = self.fallback_model.generate_response(
                        prompt, system_prompt)

                    return {
                        "response": response,
                        "model_used": self.fallback_model.model_name,
                        "fallback_used": True
                    }

                except Exception as fallback_error:
                    return {
                        "response": f"Error in both primary and fallback models. Primary: {e}, Fallback: {fallback_error}",
                        "model_used": "none",
                        "fallback_used": True,
                        "error": True
                    }

            return {
                "response": f"Error generating response: {str(e)}",
                "model_used": "none",
                "fallback_used": False,
                "error": True
            }

    def analyze_data(self, prompt: str, data_context: str) -> Dict[str, Any]:
        """
        Analyze data with fallback if needed.

        Args:
            prompt: User prompt
            data_context: Data context

        Returns:
            Dictionary with analysis and metadata
        """
        try:
            # Try primary model
            response = self.primary_model.analyze_data(prompt, data_context)

            # Reset failure count on success
            self.failure_count = 0

            return {
                "response": response,
                "model_used": self.primary_model.model_name,
                "fallback_used": False
            }

        except Exception as e:
            print(f"Primary model error in data analysis: {e}")
            self.failure_count += 1

            # Check if we should fall back
            if self.auto_fallback and self.failure_count >= self.fallback_threshold and self.fallback_model:
                try:
                    # Prepare specialized prompt for fallback
                    fallback_prompt = f"""
Data Analysis Request:
{prompt}

Data Information:
{data_context}

Please analyze this data and provide insights.
"""

                    # Load system prompt for fallback
                    system_prompt = self.primary_model.system_prompt

                    # Try fallback model
                    response = self.fallback_model.generate_response(
                        fallback_prompt, system_prompt)

                    return {
                        "response": response,
                        "model_used": self.fallback_model.model_name,
                        "fallback_used": True
                    }

                except Exception as fallback_error:
                    return {
                        "response": f"Error in both primary and fallback models for data analysis. Primary: {e}, Fallback: {fallback_error}",
                        "model_used": "none",
                        "fallback_used": True,
                        "error": True
                    }

            return {
                "response": f"Error analyzing data: {str(e)}",
                "model_used": "none",
                "fallback_used": False,
                "error": True
            }


# Example usage
if __name__ == "__main__":
    # Create primary model wrapper
    primary_model = ModelWrapper(
        model_name="databot",
        system_prompt_path="/home/ubuntu/data-ai-chatbot/config/system_prompt.md"
    )

    # Create fallback model wrapper (optional)
    fallback_model = None
    if os.environ.get("GEMINI_API_KEY"):
        fallback_model = GeminiWrapper()

    # Create model manager
    model_manager = ModelManager(
        primary_model=primary_model,
        fallback_model=fallback_model
    )

    # Test response generation
    response = model_manager.generate_response(
        "What insights can you extract from customer purchase data?"
    )

    print(f"Model used: {response['model_used']}")
    print(f"Response: {response['response']}")

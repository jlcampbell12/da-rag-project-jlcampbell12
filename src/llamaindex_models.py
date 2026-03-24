"""Model isolation layer for AI Lab Azure OpenAI models.

This module provides controlled access to specific AI Lab models, ensuring that all
model interactions go through authenticated and configured interfaces. No direct
model access is allowed - everything must go through these controlled factories.
"""

from typing import Any, Dict, Optional
from llama_index.llms.azure_openai import AzureOpenAI as LlamaIndexAzureOpenAI  
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from openai import AzureOpenAI

from ailab.utils.azure import get_ailab_bearer_token_provider, get_ailab_endpoint


# Model registry - defines exactly which models are available and their configurations
MODEL_REGISTRY = {
    "chat": {
        "gpt-4o": {
            "deployment_name": "gpt-4o",
            "model_name": "gpt-4o", 
            "api_version": "2024-10-01-preview",
            "description": "GPT-4o chat completion model"
        }
    },
    "embeddings": {
        "text-embedding-3-large": {
            "deployment_name": "text-embedding-3-large",
            "model_name": "text-embedding-3-large",
            "api_version": "2024-10-01-preview", 
            "description": "Text embedding model for semantic search and RAG"
        }
    }
}


class ModelAccessError(Exception):
    """Raised when attempting to access unauthorized or non-existent models."""
    pass


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Get the registry of all available models.
    
    Returns:
        Dictionary containing model categories and their available models.
    """
    return MODEL_REGISTRY.copy()


def get_chat_model(model_name: str = "gpt-4o", **kwargs) -> LlamaIndexAzureOpenAI:
    """Get a LlamaIndex chat model with isolated, controlled configuration.
    
    Args:
        model_name: The name of the chat model to use. Must be in MODEL_REGISTRY.
        **kwargs: Additional arguments to pass to the LlamaIndex model (e.g., temperature).
    
    Returns:
        Configured LlamaIndex AzureOpenAI chat model instance.
        
    Raises:
        ModelAccessError: If the requested model is not available or authorized.
    """
    if model_name not in MODEL_REGISTRY["chat"]:
        available = list(MODEL_REGISTRY["chat"].keys())
        raise ModelAccessError(
            f"Chat model '{model_name}' is not available. "
            f"Available models: {available}"
        )
    
    model_config = MODEL_REGISTRY["chat"][model_name]
    
    # Create the LlamaIndex model with controlled configuration
    llm = LlamaIndexAzureOpenAI(
        model=model_config["model_name"],
        deployment_name=model_config["deployment_name"],
        api_key="dummy",  # Will be overridden by token provider
        azure_ad_token_provider=get_ailab_bearer_token_provider(),
        azure_endpoint=get_ailab_endpoint(),
        api_version=model_config["api_version"],
        **kwargs
    )
    
    return llm


def get_embedding_model(model_name: str = "text-embedding-3-large", **kwargs) -> AzureOpenAIEmbedding:
    """Get a LlamaIndex embedding model with isolated, controlled configuration.
    
    Args:
        model_name: The name of the embedding model to use. Must be in MODEL_REGISTRY.
        **kwargs: Additional arguments to pass to the embedding model.
    
    Returns:
        Configured LlamaIndex AzureOpenAIEmbedding instance.
        
    Raises:
        ModelAccessError: If the requested model is not available or authorized.
    """
    if model_name not in MODEL_REGISTRY["embeddings"]:
        available = list(MODEL_REGISTRY["embeddings"].keys())
        raise ModelAccessError(
            f"Embedding model '{model_name}' is not available. "
            f"Available models: {available}"
        )
    
    model_config = MODEL_REGISTRY["embeddings"][model_name]
    
    # Create the LlamaIndex embedding model with controlled configuration
    embedding = AzureOpenAIEmbedding(
        model=model_config["model_name"],
        deployment_name=model_config["deployment_name"],
        api_key="dummy",  # Will be overridden by token provider
        azure_ad_token_provider=get_ailab_bearer_token_provider(),
        azure_endpoint=get_ailab_endpoint(),
        api_version=model_config["api_version"],
        **kwargs
    )
    
    return embedding


def get_raw_openai_client(**kwargs) -> AzureOpenAI:
    """Get a raw Azure OpenAI client for direct API access.
    
    This provides access to the underlying OpenAI client for cases where
    LlamaIndex wrappers are insufficient. All authentication is still controlled.
    
    Args:
        **kwargs: Additional arguments to pass to AzureOpenAI client.
    
    Returns:
        Configured AzureOpenAI client instance.
    """
    client = AzureOpenAI(
        api_version="2024-10-01-preview",
        azure_ad_token_provider=get_ailab_bearer_token_provider(),
        azure_endpoint=get_ailab_endpoint(),
        **kwargs
    )
    
    return client


def validate_model_access(model_type: str, model_name: str) -> bool:
    """Validate that a specific model is available for access.
    
    Args:
        model_type: Type of model ("chat" or "embeddings").
        model_name: Name of the specific model.
    
    Returns:
        True if model is available, False otherwise.
    """
    if model_type not in MODEL_REGISTRY:
        return False
    return model_name in MODEL_REGISTRY[model_type]


# Convenience functions for the specific models documented in authentication.md
def get_gpt4o(**kwargs) -> LlamaIndexAzureOpenAI:
    """Get the GPT-4o chat model with controlled access."""
    return get_chat_model("gpt-4o", **kwargs)


def get_text_embedding_3_large(**kwargs) -> AzureOpenAIEmbedding:
    """Get the text-embedding-3-large model with controlled access.""" 
    return get_embedding_model("text-embedding-3-large", **kwargs)
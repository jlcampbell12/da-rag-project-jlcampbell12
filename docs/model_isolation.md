# Model Isolation and Controlled Access

This document explains the model isolation layer implemented in the AI Lab authentication module, which ensures that all model access is controlled, authenticated, and auditable.

## Overview

The model isolation system provides a controlled interface for accessing AI Lab models, preventing direct instantiation and ensuring that all access goes through authenticated, pre-configured pathways. This eliminates the risk of configuration drift, unauthorized model usage, and security vulnerabilities.

## Key Principles

1. **No Direct Access**: Third-party libraries cannot directly instantiate models
2. **Registry-Based Control**: Only explicitly registered models are available
3. **Centralized Authentication**: All models use the same controlled authentication
4. **Fail-Safe Design**: Unauthorized access attempts are blocked with clear error messages

## Available Models

Based on `docs/authentication.md`, the following models are available:

### Chat Models
- **gpt-4o**: GPT-4o chat completion model (2024-10-01-preview)

### Embedding Models  
- **text-embedding-3-large**: Text embedding model for semantic search and RAG (2024-10-01-preview)

## Usage Patterns

### ✅ Correct Usage (Controlled Access)

```python
from llamaindex_models import get_gpt4o, get_text_embedding_3_large

# Get models through controlled interfaces
llm = get_gpt4o(temperature=0.7, max_tokens=150)
embedding = get_text_embedding_3_large()

# Use with LlamaIndex as normal
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, 
                                       llm=llm, 
                                       embed_model=embedding)
```

### ❌ Blocked Usage (Direct Access)

```python
# This will NOT work - direct instantiation is blocked
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

# These will fail due to missing authentication or unauthorized models
llm = AzureOpenAI(model="gpt-4o", ...)  # Missing controlled auth
embedding = AzureOpenAIEmbedding(model="unauthorized-model", ...)  # Not in registry
```

## API Reference

### Core Functions

#### `get_chat_model(model_name: str = "gpt-4o", **kwargs) -> LlamaIndexAzureOpenAI`
Get a LlamaIndex chat model with controlled configuration.

**Parameters:**
- `model_name`: The chat model name (must be in registry)
- `**kwargs`: Additional LlamaIndex parameters (temperature, max_tokens, etc.)

**Returns:** Configured LlamaIndex AzureOpenAI instance

**Raises:** `ModelAccessError` if model is not authorized

#### `get_embedding_model(model_name: str = "text-embedding-3-large", **kwargs) -> AzureOpenAIEmbedding`
Get a LlamaIndex embedding model with controlled configuration.

**Parameters:**
- `model_name`: The embedding model name (must be in registry)
- `**kwargs`: Additional embedding model parameters

**Returns:** Configured AzureOpenAIEmbedding instance

**Raises:** `ModelAccessError` if model is not authorized

#### `get_raw_openai_client(**kwargs) -> AzureOpenAI`
Get a raw Azure OpenAI client for direct API access with controlled authentication.

**Returns:** Configured AzureOpenAI client instance

### Convenience Functions

#### `get_gpt4o(**kwargs) -> LlamaIndexAzureOpenAI`
Direct access to the GPT-4o model.

#### `get_text_embedding_3_large(**kwargs) -> AzureOpenAIEmbedding`
Direct access to the text-embedding-3-large model.

### Utility Functions

#### `get_available_models() -> Dict[str, Dict[str, Any]]`
Get the complete registry of available models.

#### `validate_model_access(model_type: str, model_name: str) -> bool`
Validate that a specific model is available for access.

## Model Registry

The system uses a centralized registry that defines exactly which models are available:

```python
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
```

## Security Features

### Authentication Control
- All models use `get_ailab_bearer_token_provider()` for authentication
- No environment variables or direct API keys allowed
- Token provider ensures proper Azure Identity integration

### Access Control  
- Only registry-defined models are accessible
- Unauthorized model access raises `ModelAccessError`
- Clear error messages guide users to valid options

### Configuration Control
- Centralized endpoint configuration via `get_ailab_endpoint()`
- Consistent API version enforcement (2024-10-01-preview)
- No manual configuration allowed

## Benefits

1. **Security**: Prevents unauthorized model access and configuration drift
2. **Auditability**: All model access goes through controlled pathways
3. **Maintainability**: Central registry makes it easy to add/remove models
4. **Consistency**: All models use the same authentication and configuration
5. **Developer Experience**: Simple, clear API with helpful error messages
6. **Integration Ready**: Works seamlessly with LlamaIndex and other frameworks

## Error Handling

When attempting to access unauthorized models, the system provides clear guidance:

```python
from llamaindex_models import get_chat_model, ModelAccessError

try:
    llm = get_chat_model("unauthorized-model")
except ModelAccessError as e:
    print(e)
    # Output: Chat model 'unauthorized-model' is not available. 
    #         Available models: ['gpt-4o']
```

## Adding New Models

To add new models to the system:

1. Update the `MODEL_REGISTRY` in `src/llamaindex_models.py`
2. Ensure the model is documented in `docs/authentication.md`
3. Add tests for the new model 
4. Update this documentation

This ensures that model additions go through a controlled process with proper testing and documentation.
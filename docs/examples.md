# Examples Guide

This guide describes the example scripts included in the project that demonstrate various aspects of the AI Lab authentication and model isolation system.

## Overview

The project includes four example scripts in `docs/llamaindex_examples/` that progressively demonstrate the system's capabilities:

1. **example_usage.py** - Basic authentication module usage (for LlamaIndex, use llamaindex_models.py instead)
2. **example_model_isolation.py** - Model isolation and security controls
3. **example_chat_usage.py** - LlamaIndex chat completions
4. **example_vector_search.py** - Vector embeddings and semantic search

## Running Examples

All examples can be run using `uv`:

```bash
uv run python docs/llamaindex_examples/example_usage.py
uv run python docs/llamaindex_examples/example_model_isolation.py
uv run python docs/llamaindex_examples/example_chat_usage.py
uv run python docs/llamaindex_examples/example_vector_search.py
```

## Example Scripts

### 1. example_usage.py - Authentication Basics

**Purpose**: Demonstrates the core authentication module functionality without model isolation.

**What it shows**:
- Getting bearer token provider from `ailab.utils.azure`
- Getting endpoint configuration
- Testing token generation
- Basic OpenAI client setup

**Key concepts**:
- Direct use of `get_ailab_bearer_token_provider()` and `get_ailab_endpoint()`
- How to configure an AzureOpenAI client with controlled authentication
- Token validation and preview

**When to use this pattern**:
- When you need direct OpenAI client access without LlamaIndex
- For understanding the underlying authentication mechanism
- For debugging authentication issues

**Sample output**:
```
🔑 Azure OpenAI Authentication Module Demo
==================================================

1. Getting authentication components...
   ✓ Token provider: function
   ✓ Endpoint: https://ct-enterprisechat-api.azure-api.net/

2. Testing token generation...
   ✓ Token obtained successfully (length: 2272)
   ✓ Token preview: eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6InJ0c0...
```

### 2. example_model_isolation.py - Security Controls

**Purpose**: Demonstrates the model isolation layer and security controls that prevent unauthorized model access.

**What it shows**:
- Available models in the registry
- Controlled model access through factory functions
- Security blocking of unauthorized models
- Proper integration patterns with LlamaIndex

**Key concepts**:
- Model registry and authorization
- `ModelAccessError` exceptions for unauthorized access
- Difference between controlled vs. direct access patterns
- Benefits of centralized model management

**When to use this pattern**:
- Understanding security boundaries
- Learning proper model access patterns
- Verifying model authorization before development

**Sample output**:
```
🔒 AI Lab Model Isolation Demo
==================================================

1. Available Models (Controlled Registry)
   📁 Chat Models:
      - gpt-4o (GPT-4o chat completion model)
   📁 Embeddings Models:
      - text-embedding-3-large (Text embedding model for semantic search and RAG)

2. Controlled Model Access
   🤖 Getting GPT-4o chat model...
      ✓ Chat model: AzureOpenAI
   📊 Getting text embedding model...
      ✓ Embedding model: AzureOpenAIEmbedding

3. Unauthorized Access Prevention
   ✓ Correctly blocked unauthorized model: gpt-3.5-turbo
   ✓ Correctly blocked unauthorized model: claude-3
```

### 3. example_chat_usage.py - LlamaIndex Chat

**Purpose**: Demonstrates how to use the GPT-4o model with LlamaIndex for chat completions and conversational AI.

**What it shows**:
- Getting controlled GPT-4o model with `get_gpt4o()`
- Simple text completion
- Multi-turn conversations with chat history
- Streaming responses
- Model configuration (temperature, max_tokens, system prompts)

**Key concepts**:
- LlamaIndex `complete()` method for simple completions
- LlamaIndex `chat()` method for conversations
- LlamaIndex `stream_chat()` for streaming responses
- Controlled model configuration via factory function parameters

**When to use this pattern**:
- Building chatbots or conversational interfaces
- Question-answering systems
- Text generation tasks
- Any application requiring LLM completions

**Sample output**:
```
🤖 LlamaIndex Chat Model Demo
==================================================

1. Getting GPT-4o model through controlled interface...
   ✓ Model obtained: AzureOpenAI

2. Simple completion example...
   📝 Response: Clean code is a concept popularized by Robert C. Martin...

3. Chat conversation example...
   💬 User: What are the key principles of clean code?
   🤖 Response: [detailed response about clean code principles]
```

### 4. example_vector_search.py - Semantic Search

**Purpose**: Demonstrates vector embeddings and semantic search using LlamaIndex with controlled model access.

**What it shows**:
- Getting controlled embedding model with `get_text_embedding_3_large()`
- Creating a vector index from documents
- Semantic similarity search with relevance scoring
- Query engine configuration
- Direct embedding computation
- Similarity evaluation

**Key concepts**:
- Document indexing with vector embeddings
- `VectorStoreIndex` creation and usage
- Query engines for semantic search
- Embedding dimensions and similarity metrics
- Integration of both LLM and embedding models in RAG systems

**When to use this pattern**:
- Building RAG (Retrieval-Augmented Generation) systems
- Document search and retrieval
- Semantic similarity comparisons
- Knowledge base systems
- Question answering over documents

**Sample output**:
```
🔍 LlamaIndex Vector Similarity Search Demo
============================================================

1. Getting models through controlled interfaces...
   ✓ Embedding model: AzureOpenAIEmbedding
   ✓ Chat model: AzureOpenAI

2. Creating sample documents...
   📄 Created 5 sample documents

3. Building vector index with controlled embedding model...
   ✅ Vector index created with 5 nodes

5. Running 4 semantic search queries...
   Query 1: 'What is a programming language good for beginners?'
   🎯 Answer: A programming language that is good for beginners...
   📚 Sources:
      1. Topic: programming, Similarity: 0.443
```

## Progressive Learning Path

Follow these examples in order for the best learning experience:

1. **Start with `docs/llamaindex_examples/example_usage.py`** to understand authentication fundamentals
2. **Move to `docs/llamaindex_examples/example_model_isolation.py`** to understand security controls and proper access patterns
3. **Try `docs/llamaindex_examples/example_chat_usage.py`** to learn basic LLM interactions
4. **Finish with `docs/llamaindex_examples/example_vector_search.py`** to see complete RAG system integration

## Common Patterns

### Getting Models

```python
from llamaindex_models import get_gpt4o, get_text_embedding_3_large

# Get chat model with configuration
llm = get_gpt4o(
    temperature=0.7,
    max_tokens=150,
    system_prompt="You are a helpful assistant."
)

# Get embedding model
embedding = get_text_embedding_3_large()
```

### Using with LlamaIndex

```python
from llama_index.core import VectorStoreIndex, Document

# Create documents
documents = [Document(text="Your content here")]

# Build index with controlled models
index = VectorStoreIndex.from_documents(
    documents,
    llm=llm,
    embed_model=embedding
)

# Query the index
response = index.as_query_engine().query("Your question")
```

### Error Handling

```python
from llamaindex_models import get_chat_model, ModelAccessError

try:
    llm = get_chat_model("unauthorized-model")
except ModelAccessError as e:
    print(f"Model not authorized: {e}")
    # Handle gracefully - check available models
```

## Authentication Requirements

All examples require prior authentication:

```bash
azd auth login --scope api://ailab/Model.Access
```

If examples fail with authentication errors, verify your login status and permissions.

## Modifying Examples

Feel free to modify these examples to experiment with:

- Different model parameters (temperature, max_tokens)
- Your own documents and queries
- Different system prompts
- Additional LlamaIndex features (retrievers, post-processors, etc.)

The controlled model access ensures you stay within authorized boundaries while experimenting.

## Next Steps

After working through the examples:

1. Review the [Model Isolation Guide](model_isolation.md) for detailed API reference
2. Check [Authentication Guide](authentication.md) for troubleshooting
3. Review the test suite in `tests/` for additional usage patterns
4. Start building your own RAG applications with the controlled models

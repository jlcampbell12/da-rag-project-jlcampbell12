# Setup
This is the README for the RAG system you are building. For instructions to the consultant leading the project, see the [readme_setup](./readme_setup.md).

# Wikipedia RAG System

A Retrieval-Augmented Generation (RAG) system using LlamaIndex with Wikipedia articles, built with FastAPI and Azure OpenAI models.

## Overview

This project implements a complete RAG pipeline that:
- Loads Wikipedia passages from HuggingFace datasets
- Generates embeddings using Azure OpenAI's `text-embedding-3-large`
- Stores vectors in a local LlamaIndex vector database
- Retrieves relevant documents based on semantic similarity
- Generates answers using Azure OpenAI's `gpt-4o` model

## Features

- 🔒 **Secure Model Access**: All Azure OpenAI models accessed through controlled authentication
- 🚀 **FastAPI Backend**: RESTful API for all RAG operations
- 📊 **Full Observability**: Jupyter notebooks for inspecting every stage of the pipeline
- 🎯 **Explainable**: Clear separation of retrieval and generation steps
- 🧪 **Evaluation Ready**: Built-in support for test questions and metrics

## Architecture

```
User Query → Embedding → Vector Search → Retrieve Top-K Docs → Augment Prompt → GPT-4o → Answer
```

See `docs/architecture.md` for detailed architecture diagrams.

## Prerequisites

1. **Python 3.13+**
2. **Azure Access**: Access to Azure OpenAI services with AI Lab models
3. **Azure CLI**: For authentication
4. **UV Package Manager**: Recommended for dependency management

## Installation

### 1. Clone and Setup

```bash
cd /path/to/your_project

# Set up virtual environment and install dependencies using uv
uv venv
uv sync
```

### 2. Authenticate with Azure

```bash
# Login with AI Lab scope
azd auth login --scope api://ailab/Model.Access
```

This authentication is required for accessing the Azure OpenAI models (GPT-4o and text-embedding-3-large).

See `docs/authentication.md` for more details on authentication setup.

## Usage

### Starting the API Server


### API Documentation

### Key Endpoints

#### Health Check

#### Ingest Data




#### Query Documents

#### Generate Answer (RAG)

## Observability with Jupyter Notebooks

The system includes three Jupyter notebooks for inspecting and understanding the RAG pipeline.



### Running Notebooks


## Project Structure



## Data Sources

The system uses the `rag-mini-wikipedia` dataset from HuggingFace:
- **Passages**: `hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet`
- **Test Questions**: `hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet`

## Model Configuration

This project uses controlled access to Azure OpenAI models:

- **Embedding Model**: `text-embedding-3-large` (2024-10-01-preview)
- **Chat Model**: `gpt-4o` (2024-10-01-preview)

All model access goes through the `llamaindex_models.py` isolation layer. See `docs/model_isolation.md` for details.

## Development Workflow

### 1. First Time Setup
```bash
# Authenticate
azd auth login --scope api://ailab/Model.Access

# Install dependencies
uv venv
uv sync

# Start API server

```

### 2. Ingest Data


### 3. Explore with Notebooks


### 4. Iterate and Observe


## Troubleshooting

### Authentication Issues
```bash
# Verify Azure login
azd auth login --scope api://ailab/Model.Access

# Check token
azd auth token --output json | jq -r '.expiresOn'
azd auth token --output json | jq -r '.token'

# remove token
azd auth logout
```

### Import Errors
```bash
# Reinstall dependencies
uv sync --reinstall
```

### Index Not Found
If you get "No index available" errors:
```bash
# Run ingestion first
```

### Dataset Loading Issues

## Testing

The system includes comprehensive integration tests that verify both internal behavior and API consistency.

### Quick Start

See `docs/testing.md` for detailed testing guide.

### Test Organization

- **Internal Tests** 
- **Integration Tests**  
- **Logs**:  

## Troubleshooting

This project follows two core principles:

1. **Simple (a la Rich Hickey)**: Independent, unentangled components in a clear data pipeline
2. **Explainable (a la "Rewilding Software Engineering")**: Observable, inspectable steps from query to answer

See `instructions.md` for the complete project brief.

## Examples

See the `docs/llamaindex_examples/example_*.py` files for standalone demonstrations:
- `example_model_isolation.py` - Model access patterns
- `example_chat_usage.py` - LLM completions
- `example_vector_search.py` - Vector similarity search

## Documentation

Comprehensive documentation is available in the `docs/` directory:
- **Architecture**: System design and data flow
- **Authentication**: Azure setup and model access
- **Model Isolation**: Security and controlled access patterns
- **Examples**: Usage patterns and best practices



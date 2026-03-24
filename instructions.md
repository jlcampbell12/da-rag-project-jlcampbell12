# Project Brief: Client Story Query System

We are designing and building a simple, mostly-local, LLM-based backend system. The system's purpose is to load a small dataset of wikipedia articles and query them. 

# Personas

*   **You** are an expert LLM assistant helping with system design and implementation.
*   **I** am an experienced software engineer with experience in LLM-specific backend systems.

# Core Design Principles

Our two guiding principles are:

1.  **Simple (a la Rich Hickey):** Systems should be composed of unentangled, independent components. We prefer pipelines of data transformation over complex, interconnected objects. See https://github.com/matthiasn/talk-transcripts/blob/master/Hickey_Rich/SimpleMadeEasy.md. 
2.  **Explainable (a la "Rewilding Software Engineering"):** The system's behavior must be exlainable, observable, and understandable. We must be able to trace how an input (a query) leads to an output (an answer) through a series of clear, inspectable steps. Black boxes are to be avoided. See https://moldabledevelopment.com/.


# Observability and Explainability Strategy
* **You** will observe the system through the results of pytest tests and curl commands against relevant endpoints
* **I** will observe the system by using jupyter notebooks to hit API endpoints

# Technology Stack

## Confirmed Decisions
*   **Language:** Python
*   **Web Framework:** FastAPI
*   **Developer Observability:** Jupyter Notebooks
*   **LLM Access:** Internal company Azure models, authenticated via Microsoft Entra ID.
*   **Authentication:** Azure Identity with DefaultAzureCredential for secure access to Azure OpenAI services.

## Implementation Approach
We have chosen LlamaIndex as our core LLM framework to implement the system components. There is a LlamaIndex component to route queries to a local LlamaIndex vector database. 

## Authentication and Model Isolation
LLM models available to Llama Index should be accessed via llamaindex_models.py. See example_model_isolation.py, example_usage.py, example_chat_usage.py, and example_vector_search.py as to how LlamaIndex should use the models available to it. LlamaIndex should NOT use its default model access method via an environment variable. 

# Implementation Plan
See STATUS.md for the checklist for implementing this system. 

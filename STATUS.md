# Prerequisites
* For each task create the appropriate set of FastAPI endpoints to support the given functionality
* These endpoints will be used to create observability Jupyter notebooks to demonstrate the functionality of the system and to observe the internal state and behavior of the system 
* Proceed in a test-driven way -- that is, for each of these tasks, create the code for the task, create the tests for the task, and run the tests; only proceed when the tests for the task pass successfully

# Ingestion
* Get source data from hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet
* Create an embedding for the source data using the given embedding LLM
* Load embeddings into a local-only vector database provided by Llama Index 
* Create an observability notebook showing the status of the data load, the embeddings, and the vector database

# User Query and Content Retrieval
* Create functions to generate embeddings for a provided query 
* Create functions to retrieve the top K matches for the query from the vector database
* Create an observability notebook to demonstrate this behavior

# Augment LLM Prompt and Generate Response
* Construct a new prompt using the provided query and the retrieved document matches
* Send the new augmented prompt to the LLM and synthesize an answer to the original query
* Create an observability notebook demonstrating this functionality
* Use the questions and answers at hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet to generate tests and examples, and to evaluate the performance of this RAG system. 


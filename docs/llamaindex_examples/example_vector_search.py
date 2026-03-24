#!/usr/bin/env python3
"""Example demonstrating LlamaIndex vector similarity search with controlled embedding model access.

This example shows how to use the text-embedding-3-large model through the controlled
interface for document indexing and semantic search.
"""

from llamaindex_models import get_text_embedding_3_large, get_gpt4o
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SimpleNodeParser


def create_sample_documents():
    """Create sample documents for demonstration."""
    documents = [
        Document(
            text="Python is a high-level programming language known for its simplicity and readability. "
                 "It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
            metadata={"topic": "programming", "language": "python"}
        ),
        Document(
            text="Machine learning is a subset of artificial intelligence that enables computers to learn "
                 "and make decisions from data without being explicitly programmed for every task.",
            metadata={"topic": "ai", "category": "machine-learning"}
        ),
        Document(
            text="Docker is a containerization platform that allows developers to package applications "
                 "and their dependencies into lightweight, portable containers that can run anywhere.",
            metadata={"topic": "devops", "tool": "docker"}
        ),
        Document(
            text="REST APIs provide a standardized way for different software applications to communicate "
                 "over HTTP using standard methods like GET, POST, PUT, and DELETE.",
            metadata={"topic": "web-development", "type": "api"}
        ),
        Document(
            text="Version control systems like Git help developers track changes in code, collaborate "
                 "with team members, and manage different versions of software projects.",
            metadata={"topic": "development", "tool": "git"}
        )
    ]
    return documents


def main():
    """Demonstrate LlamaIndex vector search with controlled embedding model access."""
    print("🔍 LlamaIndex Vector Similarity Search Demo")
    print("=" * 60)
    
    # Get controlled models
    print("\n1. Getting models through controlled interfaces...")
    try:
        embedding_model = get_text_embedding_3_large()
        llm = get_gpt4o(temperature=0.3, max_tokens=150)
        
        print(f"   ✓ Embedding model: {type(embedding_model).__name__}")
        print(f"   ✓ Chat model: {type(llm).__name__}")
    except Exception as e:
        print(f"   ❌ Failed to get models: {e}")
        return
    
    # Create sample documents
    print("\n2. Creating sample documents...")
    documents = create_sample_documents()
    print(f"   📄 Created {len(documents)} sample documents")
    for i, doc in enumerate(documents, 1):
        print(f"      {i}. {doc.metadata.get('topic', 'N/A')}: {doc.text[:50]}...")
    
    # Create vector index
    print("\n3. Building vector index with controlled embedding model...")
    try:
        # Parse documents into nodes
        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(documents)
        
        # Create index with controlled models
        index = VectorStoreIndex(
            nodes=nodes,
            embed_model=embedding_model,
            llm=llm
        )
        print(f"   ✅ Vector index created with {len(nodes)} nodes")
    except Exception as e:
        print(f"   ❌ Index creation failed: {e}")
        return
    
    # Create query engine with explicit model configuration
    print("\n4. Creating query engine...")
    try:
        # Configure the index to use our controlled models
        from llama_index.core import Settings
        Settings.llm = llm
        Settings.embed_model = embedding_model
        
        query_engine = index.as_query_engine(
            similarity_top_k=3,
            response_mode="compact"
        )
        print("   ✅ Query engine created with controlled models")
    except Exception as e:
        print(f"   ❌ Query engine creation failed: {e}")
        return
    
    # Example queries
    queries = [
        "What is a programming language good for beginners?",
        "How do containers help with software deployment?",
        "What tools help developers work together on code?",
        "How do applications communicate over the internet?"
    ]
    
    print(f"\n5. Running {len(queries)} semantic search queries...")
    
    for i, query in enumerate(queries, 1):
        print(f"\n   Query {i}: '{query}'")
        try:
            response = query_engine.query(query)
            print(f"   🎯 Answer: {str(response)}")
            
            # Show source information
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print("   📚 Sources:")
                for j, node in enumerate(response.source_nodes[:2], 1):  # Show top 2 sources
                    score = getattr(node, 'score', 'N/A')
                    topic = node.metadata.get('topic', 'N/A')
                    if isinstance(score, float):
                        print(f"      {j}. Topic: {topic}, Similarity: {score:.3f}")
                    else:
                        print(f"      {j}. Topic: {topic}, Similarity: {score}")
            
        except Exception as e:
            print(f"   ❌ Query failed: {e}")
    
    # Demonstrate direct embedding functionality and similarity
    print("\n6. Direct embedding functionality and similarity test...")
    try:
        test_texts = [
            "Python programming language",
            "Machine learning algorithms", 
            "Docker containers"
        ]
        
        print("   🧮 Computing embeddings for sample texts...")
        for text in test_texts:
            embedding = embedding_model.get_text_embedding(text)
            print(f"      '{text}': embedding dimension = {len(embedding)}")
        
        # Use LlamaIndex's built-in semantic similarity evaluator
        from llama_index.core.evaluation import SemanticSimilarityEvaluator
        
        print("   📊 Testing semantic similarity with LlamaIndex evaluator...")
        evaluator = SemanticSimilarityEvaluator(embed_model=embedding_model)
        
        # Compare similar texts
        result1 = evaluator.evaluate(
            response="Python programming",
            reference="Programming with Python"
        )
        print(f"      Similar texts similarity: {result1.score:.4f} (passing: {result1.passing})")
        
        # Compare different texts  
        result2 = evaluator.evaluate(
            response="Python programming", 
            reference="Docker containers"
        )
        print(f"      Different texts similarity: {result2.score:.4f} (passing: {result2.passing})")
        
    except Exception as e:
        print(f"   ❌ Embedding and similarity test failed: {e}")
    
    print("\n7. System verification...")
    print(f"   🔧 Embedding model class: {type(embedding_model).__name__}")
    print(f"   🔧 Has 'get_text_embedding' method: {hasattr(embedding_model, 'get_text_embedding')}")
    print(f"   🔧 Vector index type: {type(index).__name__}")
    print(f"   🔧 Query engine type: {type(query_engine).__name__}")
    
    print("\n✅ Vector similarity search integration successful!")
    print("\nKey benefits:")
    print("  - ✅ Controlled embedding model access")
    print("  - ✅ Automatic authentication handling") 
    print("  - ✅ Full LlamaIndex embedding interface support")
    print("  - ✅ Semantic search with relevance scoring")
    print("  - ✅ No manual configuration required")
    print("  - ✅ Ready for production RAG systems")


if __name__ == "__main__":
    main()
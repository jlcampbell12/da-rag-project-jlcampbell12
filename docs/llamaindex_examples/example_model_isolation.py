#!/usr/bin/env python3
"""Example script demonstrating isolated model access through AI Lab module.

This script shows how third-party libraries like LlamaIndex must access models
through controlled, authenticated interfaces with no direct access allowed.
"""

from llamaindex_models import (
    get_available_models,
    get_gpt4o, 
    get_text_embedding_3_large,
    get_chat_model,
    get_embedding_model,
    get_raw_openai_client,
    ModelAccessError
)

def main():
    """Demonstrate isolated model access patterns."""
    print("🔒 AI Lab Model Isolation Demo")
    print("=" * 50)
    
    # Show available models
    print("\n1. Available Models (Controlled Registry)")
    registry = get_available_models()
    for model_type, models in registry.items():
        print(f"   📁 {model_type.title()} Models:")
        for name, config in models.items():
            print(f"      - {name} ({config['description']})")
    
    # Demonstrate controlled model access
    print("\n2. Controlled Model Access")
    
    try:
        # Get models through controlled interfaces
        print("   🤖 Getting GPT-4o chat model...")
        gpt4o = get_gpt4o(temperature=0.7, max_tokens=150)
        print(f"      ✓ Chat model: {type(gpt4o).__name__}")
        
        print("   📊 Getting text embedding model...")
        embedding_model = get_text_embedding_3_large()
        print(f"      ✓ Embedding model: {type(embedding_model).__name__}")
        
        print("   🔧 Getting raw OpenAI client...")
        client = get_raw_openai_client()
        print(f"      ✓ Raw client: {type(client).__name__}")
        
    except Exception as e:
        print(f"   ❌ Model access failed: {e}")
        return
    
    # Demonstrate that unauthorized access is blocked
    print("\n3. Unauthorized Access Prevention")
    
    unauthorized_models = ["gpt-3.5-turbo", "claude-3", "custom-model"]
    
    for model_name in unauthorized_models:
        try:
            get_chat_model(model_name)
            print(f"   ❌ ERROR: {model_name} should have been blocked!")
        except ModelAccessError:
            print(f"   ✓ Correctly blocked unauthorized model: {model_name}")
        except Exception as e:
            print(f"   ⚠️  Unexpected error for {model_name}: {e}")
    
    # Show how this integrates with LlamaIndex
    print("\n4. LlamaIndex Integration Pattern")
    print("   This is how developers should access models:")
    print("""
   # ✅ CORRECT: Controlled access through AI Lab module
   from llamaindex_models import get_gpt4o, get_text_embedding_3_large
   
   llm = get_gpt4o(temperature=0.7)
   embedding = get_text_embedding_3_large()
   
   # Now use with LlamaIndex as normal
   from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
   
   documents = SimpleDirectoryReader("data").load_data()
   index = VectorStoreIndex.from_documents(documents, 
                                          llm=llm, 
                                          embed_model=embedding)
   
   # ❌ BLOCKED: Direct model instantiation bypasses controls
   # from llama_index.llms.azure_openai import AzureOpenAI
   # llm = AzureOpenAI(model="unauthorized-model", ...)  # This won't work
   """)
    
    print("\n5. Benefits of Model Isolation")
    print("   ✓ All model access is authenticated and controlled")
    print("   ✓ No direct instantiation of unauthorized models") 
    print("   ✓ Centralized configuration and versioning")
    print("   ✓ Audit trail of all model usage")
    print("   ✓ Easy to add/remove models from registry")
    print("   ✓ Prevents configuration drift and security issues")
    
    print("\n🎯 Model isolation is active and protecting your AI Lab resources!")

if __name__ == "__main__":
    main()
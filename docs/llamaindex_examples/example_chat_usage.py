#!/usr/bin/env python3
"""Example demonstrating LlamaIndex chat functionality with controlled AI Lab model access.

This example shows how to use the GPT-4o model through the controlled interface
for basic chat completions and conversational AI.
"""

from llamaindex_models import get_gpt4o


def main():
    """Demonstrate LlamaIndex chat model usage with controlled access."""
    print("🤖 LlamaIndex Chat Model Demo")
    print("=" * 50)
    
    # Get the controlled GPT-4o model
    print("\n1. Getting GPT-4o model through controlled interface...")
    try:
        llm = get_gpt4o(
            temperature=0.7,
            max_tokens=200,
            system_prompt="You are a helpful AI assistant with expertise in software engineering."
        )
        print(f"   ✓ Model obtained: {type(llm).__name__}")
    except Exception as e:
        print(f"   ❌ Failed to get model: {e}")
        return
    
    # Example 1: Simple completion
    print("\n2. Simple completion example...")
    try:
        response = llm.complete("What are the key principles of clean code?")
        print(f"   📝 Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Completion failed: {e}")
    
    # Example 2: Chat conversation
    print("\n3. Chat conversation example...")
    try:
        from llama_index.core.llms import ChatMessage
        
        messages = [
            ChatMessage(role="user", content="What is dependency injection?"),
        ]
        
        chat_response = llm.chat(messages)
        print(f"   💬 Chat response: {chat_response.message.content}")
        
        # Follow-up message
        messages.append(chat_response.message)
        messages.append(ChatMessage(role="user", content="Can you give me a Python example?"))
        
        follow_up = llm.chat(messages)
        print(f"   🔄 Follow-up: {follow_up.message.content}")
        
    except Exception as e:
        print(f"   ❌ Chat failed: {e}")
    
    # Example 3: Streaming response
    print("\n4. Streaming response example...")
    try:
        print("   🌊 Streaming response for 'Explain microservices architecture':")
        stream = llm.stream_complete("Explain microservices architecture in 3 bullet points.")
        
        response_text = ""
        for chunk in stream:
            if chunk.delta:
                print(chunk.delta, end="", flush=True)
                response_text += chunk.delta
        
        print(f"\n   ✓ Complete streamed response received ({len(response_text)} chars)")
        
    except Exception as e:
        print(f"   ❌ Streaming failed: {e}")
    
    print("\n5. Model configuration verification...")
    print(f"   🔧 Model class: {type(llm).__name__}")
    print(f"   🔧 Model has 'complete' method: {hasattr(llm, 'complete')}")
    print(f"   🔧 Model has 'chat' method: {hasattr(llm, 'chat')}")
    print(f"   🔧 Model has 'stream_complete' method: {hasattr(llm, 'stream_complete')}")
    
    print("\n✅ LlamaIndex chat model integration successful!")
    print("\nKey benefits:")
    print("  - ✅ Controlled access through ailab.models")
    print("  - ✅ Automatic authentication handling")
    print("  - ✅ Full LlamaIndex LLM interface support")
    print("  - ✅ No manual configuration required")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# Tim Chase
# 2025-11-23
# This is an example of using the CapTech azure util to access AI Lab models. However, 
# see models.py for the preferred way to access models in a controlled manner.

"""Example script demonstrating Azure OpenAI authentication module usage.

This script shows how to use the authentication module to access AI Lab models
as described in the project instructions.
"""

from ailab.utils.azure import get_ailab_bearer_token_provider, get_ailab_endpoint

def main():
    """Demonstrate the authentication module functionality."""
    print("🔑 Azure OpenAI Authentication Module Demo")
    print("=" * 50)
    
    # Get the token provider and endpoint
    print("\n1. Getting authentication components...")
    token_provider = get_ailab_bearer_token_provider()
    endpoint = get_ailab_endpoint()
    
    print(f"   ✓ Token provider: {type(token_provider).__name__}")
    print(f"   ✓ Endpoint: {endpoint}")
    
    # Test token generation
    print("\n2. Testing token generation...")
    try:
        token = token_provider()
        print(f"   ✓ Token obtained successfully (length: {len(token)})")
        print(f"   ✓ Token preview: {token[:50]}...")
    except Exception as e:
        print(f"   ❌ Token generation failed: {e}")
        return
    
    print("\n3. Ready for OpenAI client usage!")
    print("\nExample usage:")
    print("""
from ailab.utils.azure import get_ailab_bearer_token_provider, get_ailab_endpoint
from openai import AzureOpenAI

client = AzureOpenAI(
    api_version='2024-10-01-preview',
    azure_ad_token_provider=get_ailab_bearer_token_provider(),
    azure_endpoint=get_ailab_endpoint()
)

# Now you can use the client to access AI Lab models
# result = client.chat.completions.create(model="gpt-4o", ...)
""")
    
    print("\n🎉 Authentication module is working correctly!")

if __name__ == "__main__":
    main()
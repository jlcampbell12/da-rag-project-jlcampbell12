# Authentication Guide

This guide explains how to set up and configure authentication for the Client Story Query System, which uses Azure OpenAI services with Microsoft Entra ID (formerly Azure Active Directory) authentication.

## Authentication Architecture

The system uses Azure's authentication mechanisms to securely access Azure OpenAI services. Here's how it works:

1. **Azure Identity**: The system uses `DefaultAzureCredential` from the `azure-identity` package, which provides a simple way to authenticate to Azure services.
   
2. **Token Provider**: A bearer token provider is created to access the AI Lab model using the `get_ailab_bearer_token_provider()` function from `ailab.utils.azure`.

3. **Microsoft Entra ID Integration**: Authentication is done using Microsoft Entra ID (formerly Azure Active Directory) with the scope `api://ailab/Model.Access`.

# Models  to access
The models available to access via Azure are listed below. For all models, the "deployment_name" is the same as the model name:
* Azure OpenAI, gpt-4o (2024-10-01-preview version)
* Azure OpenAI, text-embedding-3-large (2024-10-01-preview version)

## Setting Up Authentication

### Prerequisites

- Azure account with access to Azure OpenAI services
- Microsoft Entra ID configured for your account
- Appropriate permissions assigned to your account for AI Lab model access

### Local Development Setup

1. **Install Azure CLI**:
   ```bash
   # For macOS
   brew install azure-cli
   
   # For other platforms, see:
   # https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
   ```

2. **Login to Azure**:
   ```bash
   # This will open a browser window for authentication
   azd auth login --scope api://ailab/Model.Access
   ```




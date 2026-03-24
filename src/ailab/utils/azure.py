def get_ailab_endpoint():
    import os
    if 'AILAB_ENDPOINT' in os.environ:
        return os.environ['AILAB_ENDPOINT']    
    return 'https://ct-enterprisechat-api.azure-api.net/'

def get_ailab_bearer_token_provider():
    """
    Retrieves a bearer token provider for AI Lab model access using Azure credentials.
    
    The Azure credentials are loaded using the [DefaultAzureCredential](https://docs.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) class from the [azure-identity](https://pypi.org/project/azure-identity/) package. See the documentation for the sequence of authentication methods used to try and obtain credentials.
    
    For local use, the recommended path is via SSO using the `azd auth login --scope api://ailab/Model.Access` command. This will store the credentials in the local cache and allow the DefaultAzureCredential to retrieve them.

    Returns:
        Callable: A token provider function that can be used to obtain bearer tokens.

    Example:
        token_provider = get_ailab_bearer_token_provider()
        token = token_provider()
    """
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider as _get_bearer_token_provider 
    token_provider = _get_bearer_token_provider(
        DefaultAzureCredential(), "api://ailab/Model.Access"
    )    
    return token_provider
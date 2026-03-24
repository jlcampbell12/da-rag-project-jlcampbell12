from azure.identity import DefaultAzureCredential
credential = DefaultAzureCredential()
token = credential.get_token("api://ailab/Model.Access")
print(token.token)
import json

import requests

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'


#TODO:
# ---
# https://dialx.ai/dial_api#operation/sendEmbeddingsRequest
# ---
# Implement DialEmbeddingsClient:
# - constructor should apply deployment name and api key
# - create method `get_embeddings` that will generate embeddings for input list (don't forget about dimensions)
#   with Embedding model and return back a dict with indexed embeddings (key is index from input list and value vector list)

class DialEmbeddingsClient:
    def __init__(self, deployment_name: str, api_key: str):
        self.deployment_name = deployment_name
        self.api_key = api_key

    def get_embeddings(self, input_list: list, dimensions: int) -> dict:
        headers = {
            'Content-Type': 'application/json',
            'Api-Key': self.api_key
        }
        url = DIAL_EMBEDDINGS.format(model=self.deployment_name)
        payload = {
            "input": input_list,
            "dimensions": dimensions
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        data = response.json().get('data', [])
        return {item['index']: item['embedding'] for item in data}


# Hint:
#  Response JSON:
#  {
#     "data": [
#         {
#             "embedding": [
#                 0.19686688482761383,
#                 ...
#             ],
#             "index": 0,
#             "object": "embedding"
#         }
#     ],
#     ...
#  }

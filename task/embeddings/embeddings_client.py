import json

import requests

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'


class DialEmbeddingsClient:
    _api_key: str
    _endpoint: str

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be null or empty")

        self._api_key = api_key
        self._endpoint = DIAL_EMBEDDINGS.format(model=model)

    def get_embeddings(
            self, inputs: str | list[str],
            dimensions: int
    ) -> dict[int, list[float]]:

        headers = {
            "api-key": self._api_key,
            "Content-Type": "application/json"
        }
        request_data = {
            "input": inputs,
            "dimensions": dimensions,
        }

        response = requests.post(url=self._endpoint, headers=headers, json=request_data, timeout=60)

        if response.status_code == 200:
            response_json = response.json()
            data = response_json.get("data", [])
            return self._from_data(data)
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    def _from_data(self, data: list[dict]) -> dict[int, list[float]]:
        return {embedding_obj['index']: embedding_obj['embedding'] for embedding_obj in data}
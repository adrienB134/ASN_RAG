"""Modified class for protected endpoint, see https://github.com/stanfordnlp/dspy/issues/287"""

import requests
from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory
from dsp.modules.hf import HFModel, openai_to_hf


class HFClientTGI(HFModel):
    def __init__(self, model, port=None, url=None, token=None, http_request_kwargs=None, **kwargs):
        super().__init__(model=model, is_client=True)

        self.url = url
        self.ports = port if isinstance(port, list) else [port]
        self.http_request_kwargs = http_request_kwargs or {}

        self.headers = {"Accept": "application/json", "Content-Type": "application/json"}

        if token:
            self.headers["Authorization"] = f"Bearer {token}"

        self.kwargs = {
            "temperature": 0.01,
            "max_tokens": 75,
            "top_p": 0.97,
            "n": 1,
            "stop": ["\n", "\n\n"],
            **kwargs,
        }

        # print(self.kwargs)

    def _generate(self, prompt, **kwargs):
        kwargs = {**self.kwargs, **kwargs}

        payload = {
            "inputs": prompt,
            "parameters": {
                "do_sample": kwargs["n"] > 1,
                "best_of": kwargs["n"],
                "details": kwargs["n"] > 1,
                # "max_new_tokens": kwargs.get('max_tokens', kwargs.get('max_new_tokens', 75)),
                # "stop": ["\n", "\n\n"],
                **kwargs,
            },
        }

        payload["parameters"] = openai_to_hf(**payload["parameters"])

        payload["parameters"]["temperature"] = max(0.1, payload["parameters"]["temperature"])

        # print(payload['parameters'])

        response = send_hftgi_request_v01_wrapped(
            f"{self.url}/generate",
            json=payload,
            headers=self.headers,
            **self.http_request_kwargs,
        )

        try:
            json_response = response.json()

            # completions = json_response["generated_text"]

            completions = [json_response["generated_text"]]

            if "details" in json_response and "best_of_sequences" in json_response["details"]:
                completions += [x["generated_text"] for x in json_response["details"]["best_of_sequences"]]

            response = {"prompt": prompt, "choices": [{"text": c} for c in completions]}
            return response
        except Exception as e:
            print("Failed to parse JSON response:", response.text)
            raise Exception("Received invalid JSON response from server")


@CacheMemory.cache(ignore=["arg"])
def send_hftgi_request_v01(arg, **kwargs):
    return requests.post(arg, **kwargs)


# @functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache(ignore=["arg"])
def send_hftgi_request_v01_wrapped(arg, **kwargs):
    return send_hftgi_request_v01(arg, **kwargs)

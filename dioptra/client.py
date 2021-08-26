import logging
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class Client:

    def __init__(self):

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["POST"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)

        self.http_session = requests.Session()
        self.http_session.mount("https://", adapter)
        self.timeout=10

    def call(self, payload, endpoint_url, headers):

        response = self.http_session.post(
            timeout=self.timeout,
            url=endpoint_url,
            headers=headers,
            json=payload)

        if response.status_code != requests.codes.ok:
            logging.error('There was an error calling the API')
            response.raise_for_status()

        return response.json()


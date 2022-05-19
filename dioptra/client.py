import logging
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from threading import Semaphore
from concurrent.futures import ThreadPoolExecutor
from requests_futures.sessions import FuturesSession

class Client:

    def __init__(self, max_workers, queue_size, synchronous_mode):

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=['POST']
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)

        if synchronous_mode:
            self.http_session = requests.Session()
        else:
            self.http_session = FuturesSession(
                executor=TaskManager(max_workers=max_workers, queue_size=queue_size))
        self.http_session.mount('https://', adapter)

        self.timeout=10
        self.synchronous_mode = synchronous_mode

    def response_hook(self, response, *args, **kwargs):
        if response.status_code != requests.codes.ok:
            logging.error('There was an error calling Dioptra API')
            response.raise_for_status()

    def call(self, payload, endpoint_url, headers):

        if self.synchronous_mode:

            self.http_session.post(
                timeout=self.timeout,
                url=endpoint_url,
                headers=headers,
                json=payload)
        else:
            self.http_session.post(
                timeout=self.timeout,
                url=endpoint_url,
                headers=headers,
                json=payload,
                hooks={
                    'response': self.response_hook
                })

class TaskManager():
    def __init__(self, max_workers, queue_size):
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.workers = Semaphore(max_workers + queue_size)

    def submit(self, my_method, *args, **kwargs):
        """Start a new task, blocks if queue is full."""
        self.workers.acquire()
        try:
            future = self.pool.submit(my_method, *args, **kwargs)
            future.add_done_callback(lambda x: self.workers.release())
            return future
        except:
            logging.error('Error in the task manager')
            self.workers.release()

    def shutdown(self):
        self.pool.shutdown()

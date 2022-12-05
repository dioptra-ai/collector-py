import os
import requests


class BaseMiner():

    def __init__(self):

        api_key = os.environ.get('DIOPTRA_API_KEY', None)
        if api_key is None:
            raise RuntimeError('DIOPTRA_API_KEY env var is not set')

        self.api_key = api_key
        self.app_endpoint = os.environ.get('DIOPTRA_APP_ENDPOINT', 'https://app.dioptra.ai')
        self.miner_id = None

    def get_status(self):

        try:
            r = requests.get(f'{self.app_endpoint}/api/tasks/miners/inspect/{self.miner_id}', headers={
                'content-type': 'application/json',
                'x-api-key': self.api_key
            })
            r.raise_for_status()
        except requests.exceptions.RequestException as err:
            print('There was an error getting miner status...')
            raise err
        return r.json().get('task', {}).get('status', None)

    def get_config(self):

        try:
            r = requests.get(f'{self.app_endpoint}/api/tasks/miners/{self.miner_id}', headers={
                'content-type': 'application/json',
                'x-api-key': self.api_key
            })
            r.raise_for_status()
        except requests.exceptions.RequestException as err:
            print('There was an error getting miner config...')
            raise err
        return r.json()

    def get_results(self):

        try:
            r = requests.get(f'{self.app_endpoint}/api/tasks/miners/inspect/{self.miner_id}', headers={
                'content-type': 'application/json',
                'x-api-key': self.api_key
            })
            r.raise_for_status()
        except requests.exceptions.RequestException as err:
            print('There was an error getting miner results...')
            raise err
        return r.json().get('task', {}).get('result', None)

    def run(self):

        try:
            r = requests.post(f'{self.app_endpoint}/api/tasks/miner/run', headers={
                'content-type': 'application/json',
                'x-api-key': self.api_key
            },
            json={'miner_id': self.miner_id})
            r.raise_for_status()
        except requests.exceptions.RequestException as err:
            print('There was an error running the miner...')
            raise err

    def delete(self):

        try:
            r = requests.post(f'{self.app_endpoint}/api/tasks/miners/delete', headers={
                'content-type': 'application/json',
                'x-api-key': self.api_key
            },
            json={'miner_id': self.miner_id})
            r.raise_for_status()
        except requests.exceptions.RequestException as err:
            print('There was an error deleting the miner...')
            raise err

    def reset(self):

        try:
            r = requests.post(f'{self.app_endpoint}/api/tasks/miner/reset', headers={
                'content-type': 'application/json',
                'x-api-key': self.api_key
            },
            json={'miner_id': self.miner_id})
            r.raise_for_status()
        except requests.exceptions.RequestException as err:
            print('There was an error deleting the miner...')
            raise err

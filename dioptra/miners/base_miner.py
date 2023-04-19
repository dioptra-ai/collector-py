import os
import requests
import time

DIOPTRA_APP_ENDPOINT = os.environ.get('DIOPTRA_APP_ENDPOINT', 'https://app.dioptra.ai')
# We ask for a definitive signal to disable but using the positive form is easier in code.
DIOPTRA_SSL_VERIFY = not os.environ.get('DIOPTRA_SSL_NOVERIFY', 'False') == 'True'

class BaseMiner():

    def __init__(self):

        api_key = os.environ.get('DIOPTRA_API_KEY', None)
        if api_key is None:
            raise RuntimeError('DIOPTRA_API_KEY env var is not set')

        self.api_key = api_key
        self.miner_id = None
        self.miner_name = None

    def get_status(self):

        try:
            r = requests.get(f'{DIOPTRA_APP_ENDPOINT}/api/tasks/miners/inspect/{self.miner_id}', verify=DIOPTRA_SSL_VERIFY, headers={
                'content-type': 'application/json',
                'x-api-key': self.api_key
            })
            r.raise_for_status()
        except requests.exceptions.RequestException as err:
            print('There was an error getting miner status...')
            raise err
        return r.json().get('task', {}).get('status', None)

    def get_config(self):

        r = requests.get(f'{DIOPTRA_APP_ENDPOINT}/api/tasks/miners/{self.miner_id}', verify=DIOPTRA_SSL_VERIFY, headers={
            'content-type': 'application/json',
            'x-api-key': self.api_key
        })
        r.raise_for_status()

        return r.json()

    def get_results(self, retry_on_empty_success=True):
        sleepTimeSecs = 1
        totalSleepTimeSecs = 0

        while True:
            if totalSleepTimeSecs > 3600:
                raise RuntimeError('Timed out waiting for miner results')
                
            r = requests.get(f'{DIOPTRA_APP_ENDPOINT}/api/tasks/miners/inspect/{self.miner_id}', verify=DIOPTRA_SSL_VERIFY, headers={
                'content-type': 'application/json',
                'x-api-key': self.api_key
            })
            r.raise_for_status()
            response_json = r.json()
            task = response_json['task']
            status = task['status']
            if status == 'SUCCESS':

                results =  task['result']
                # TODO: remove this.
                if results is None:
                    results = []
                if len(results) == 0 and retry_on_empty_success:
                    time.sleep(10)
                    return self.get_results(retry_on_empty_success=False)
                else:
                    return results

            elif status == 'FAILURE' or status == 'REVOKED':
                raise RuntimeError('Miner failed')
            else:
                time.sleep(sleepTimeSecs)
                totalSleepTimeSecs += sleepTimeSecs
                sleepTimeSecs = min(sleepTimeSecs * 2, 60)

    def run(self):

        try:
            r = requests.post(f'{DIOPTRA_APP_ENDPOINT}/api/tasks/miner/run', verify=DIOPTRA_SSL_VERIFY, headers={
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
            r = requests.post(f'{DIOPTRA_APP_ENDPOINT}/api/tasks/miners/delete', verify=DIOPTRA_SSL_VERIFY, headers={
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
            r = requests.post(f'{DIOPTRA_APP_ENDPOINT}/api/tasks/miner/reset', verify=DIOPTRA_SSL_VERIFY, headers={
                'content-type': 'application/json',
                'x-api-key': self.api_key
            },
            json={'miner_id': self.miner_id})
            r.raise_for_status()
        except requests.exceptions.RequestException as err:
            print('There was an error deleting the miner...')
            raise err

    def __str__(self):
        return f'Miner with uuid {self.miner_id} and name {self.miner_name}'

    def __repr__(self):
        return self.__str__()

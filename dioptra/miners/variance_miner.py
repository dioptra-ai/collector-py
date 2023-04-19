import requests
from .base_miner import BaseMiner

import os
DIOPTRA_APP_ENDPOINT = os.environ.get('DIOPTRA_APP_ENDPOINT', 'https://app.dioptra.ai')
# We ask for a definitive signal to disable but using the positive form is easier in code.
DIOPTRA_SSL_VERIFY = not os.environ.get('DIOPTRA_SSL_NOVERIFY', 'False') == 'True'

class VarianceMiner(BaseMiner):
    def __init__(
            self, display_name, size, select_filters,
            select_limit=None, select_order_by=None, select_desc=None, model_name=None):
        """
        Variance miner
        Will perform a AL query based on Variance
        Parameters:
            display_name: name to be displayed in Dioptra
            size: number of datapoints to query
            select_filters: dioptra style filters to select the data to be queried from
            select_limit: limit to selected the data
            select_order_by: field to use to sort the data to control how limit is performed
            select_desc: whether to order by dec or not
        """

        super().__init__()
        try:
            r = requests.post(f'{DIOPTRA_APP_ENDPOINT}/api/tasks/miners', verify=DIOPTRA_SSL_VERIFY, headers={
                'content-type': 'application/json',
                'x-api-key': self.api_key
            },
            json={
                'display_name': display_name,
                'strategy': 'VARIANCE',
                'size': size,
                'model_name': model_name,
                'select': {
                    'filters': select_filters,
                    **({'limit': select_limit} if select_limit is not None else {}),
                    **({'order_by': select_order_by} if select_order_by is not None else {}),
                    **({'desc': select_desc} if select_desc is not None else {}),
                }
            })
            r.raise_for_status()
            self.miner_id = r.json()['miner_id']
            self.miner_name = display_name
        except requests.exceptions.RequestException as err:
            print('There was an error getting miner status...')
            raise err
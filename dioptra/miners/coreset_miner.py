import requests
from .base_miner import BaseMiner

class ActivationMiner(BaseMiner):
    def __init__(
            self, display_name, size, embeddings_field, select_filters,
            select_limit=None, select_order_by=None, select_desc=None,
            select_reference_filters=None, select_reference_limit=None,
            select_reference_order_by=None, select_reference_desc=None):
        super().__init__()
        try:
            r = requests.post(f'{self.app_endpoint}/api/tasks/miners', headers={
                'content-type': 'application/json',
                'x-api-key': self.api_key
            },
            json={
                'display_name': display_name,
                'strategy': 'CORESET',
                'size': size,
                'embeddings_field': embeddings_field,
                'select': {
                    'filters': select_filters,
                    **({'limit': select_limit if select_limit is not None else {}}),
                    **({'order_by': select_order_by if select_order_by is not None else {}}),
                    **({'desc': select_desc if select_desc is not None else {}}),
                },
                **({'select_reference':
                    {
                        'filters': select_reference_filters,
                        **({'limit': select_reference_limit if select_reference_limit is not None else {}}),
                        **({'order_by': select_reference_order_by if select_reference_order_by is not None else {}}),
                        **({'desc': select_reference_desc if select_reference_desc is not None else {}}),
                    }
                } if select_reference_filters is not None else {})
            })
            r.raise_for_status()
            self.miner_id = r.json()['miner_id']
        except requests.exceptions.RequestException as err:
            print('There was an error getting miner status...')
            raise err

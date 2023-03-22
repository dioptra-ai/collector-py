import requests
from .base_miner import BaseMiner

class ActivationMiner(BaseMiner):
    def __init__(
            self, display_name, size, embeddings_field, select_filters,
            select_limit=None, select_order_by=None, select_desc=None, 
            select_reference_filters=None, select_reference_limit=None,select_reference_order_by=None, select_reference_desc=None,
            skip_caching=False):
        """
        Activation miner
        Will perform a AL query based on activation

        Parameters:
            display_name: name to be displayed in Dioptra
            size: number of datapoints to query
            embeddings_field: embedding fields to run the analysis on. Could be 'embeddings' 'prediction.embeddings'
            select_filters: dioptra style filters to select the data to be queried from
            select_limit: limit to selected the data
            select_order_by: field to use to sort the data to control how limit is performed
            select_desc: whether to order by dec or not
            skip_caching: whether to skip vector caching or not

        """

        super().__init__()
        try:
            r = requests.post(f'{self.app_endpoint}/api/tasks/miners', headers={
                'content-type': 'application/json',
                'x-api-key': self.api_key
            },
            json={
                'display_name': display_name,
                'strategy': 'ACTIVATION',
                'size': size,
                'embeddings_field': embeddings_field,
                'select': {
                    'filters': select_filters,
                    **({'limit': select_limit} if select_limit is not None else {}),
                    **({'order_by': select_order_by} if select_order_by is not None else {}),
                    **({'desc': select_desc} if select_desc is not None else {}),
                },
                **({'select_reference':
                    {
                        'filters': select_reference_filters,
                        **({'limit': select_reference_limit} if select_reference_limit is not None else {}),
                        **({'order_by': select_reference_order_by} if select_reference_order_by is not None else {}),
                        **({'desc': select_reference_desc} if select_reference_desc is not None else {}),
                    }
                } if select_reference_filters is not None else {}),
                'skip_caching': skip_caching
            })
            r.raise_for_status()
            self.miner_id = r.json()['miner_id']
            self.miner_name = display_name
        except requests.exceptions.RequestException as err:
            print('There was an error getting miner status...')
            raise err

import os
import argparse
import uuid
import json
import random
import datetime

from dioptra.api import Logger
from dioptra.supported_types import SupportedTypes

API_KEY = os.environ.get('DIOPTRA_API_KEY')

NUMBER_OF_EVENTS = 1000000

def load_config():
    """
    Loading the config from file

    """
    with open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                'semantic_similarity_config.json')
            , 'r') as conf_file:
        conf_json = conf_file.read()
        return json.loads(conf_json)

def get_datapoint(config):
    index = int(random.random() * (len(config)))
    return config[index]

def main():

    config = load_config()

    dioptra_logger = Logger(api_key=API_KEY)

    model_id = 'semantic_similarity'
    model_version = 'v1.1'

    for _ in range(NUMBER_OF_EVENTS):

        request_timestamp = datetime.datetime.utcnow()
        request_uuid = str(uuid.uuid4())

        datapoint = get_datapoint(config)

        # After a prediction we log the prediction, features and tags
        print('Sending a datapoint')
        dioptra_logger.commit(
            model_id=model_id,
            model_version=model_version,
            model_type=SupportedTypes.SEMANTIC_SIMILARITY,
            timestamp=request_timestamp,
            request_id=request_uuid,
            prediction=datapoint['prediction'],
            input_data=datapoint['input_data'],
            groundtruth=datapoint['groundtruth'])

if __name__ == '__main__':
    main()

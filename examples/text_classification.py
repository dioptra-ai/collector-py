import json
import os
import datetime
import uuid
import random

from dioptra.api import Logger
from dioptra.supported_types import SupportedTypes

API_KEY = os.environ.get('DIOPTRA_API_KEY')
NUMBER_OF_EVENTS = 10000


def load_config():
    """
    Loading the config from file

    """
    with open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                'text_classification_config.json')
            , 'r') as conf_file:
        conf_json = conf_file.read()
        return json.loads(conf_json)

def get_datapoint(config):
    index = int(random.random() * (len(config)))
    return config[index]


def main():

    data = load_config()

    model_id = 'imdb_genre_classifier'
    model_version = 'v1.1'

    dioptra_logger = Logger(api_key=API_KEY)

    for _ in range(NUMBER_OF_EVENTS):

        datapoint = get_datapoint(data)

        request_timestamp = datetime.datetime.utcnow()
        request_uuid = str(uuid.uuid4())

        print('Sending a datapoint')
        dioptra_logger.commit(
            model_id=model_id,
            model_version=model_version,
            model_type=SupportedTypes.IMAGE_CLASSIFIER,
            timestamp=request_timestamp,
            request_id=request_uuid,
            text=datapoint['text'],
            embeddings=datapoint['embeddings'],
            text_metadata=datapoint['text_metadata'],
            tags=datapoint['tags'],
            prediction=datapoint['prediction'],
            groundtruth=datapoint['groundtruth'])

if __name__ == '__main__':
    main()

import os
import argparse
import uuid
import json
import random
import datetime

from dioptra.api import Logger

API_KEY = os.environ.get('DIOPTRA_API_KEY')
NUMBER_OF_EVENTS = 10000


def load_config():
    """
    Loading the config from file

    """
    with open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                'image_classifier_config.json')
            , 'r') as conf_file:
        conf_json = conf_file.read()
        return json.loads(conf_json)

def get_datapoint(config):

    image_index = int(random.random() * (len(config['images'])))

    return {
        'request_tags': config['images'][image_index]['tags'],
        'model_prediction': config['images'][image_index]['prediction'],
        'model_confidence': config['images'][image_index]['confidence'],
        'image_url': config['images'][image_index]['url'],
        'image_embeddings': config['images'][image_index]['embeddings'],
        'image_features': config['images'][image_index]['features'],
        'groundtruth': config['images'][image_index]['class']
    }

def main():

    config = load_config()

    dioptra_logger = Logger(api_key=API_KEY)

    model_id = 'document_classification'
    model_version = 'v1.1'

    for _ in range(NUMBER_OF_EVENTS):

        request_timestamp = datetime.datetime.utcnow().isoformat()
        request_uuid = str(uuid.uuid4())

        datapoint = get_datapoint(config)

        # After a prediction we log the prediction, features and tags
        print('Sending a datapoint')
        dioptra_logger.log(
            model_id=model_id,
            model_version=model_version,
            timestamp=request_timestamp,
            request_id=request_uuid,
            prediction=datapoint['model_prediction'],
            confidence=datapoint['model_confidence'],
            features=datapoint['image_features'],
            embeddings=datapoint['image_embeddings'],
            image_url=datapoint['image_url'],
            tags=datapoint['request_tags'])

        # We can log the groundtruth asynchronously
        print('Sending a groundtruth')
        dioptra_logger.log(
            request_id=request_uuid,
            groundtruth=datapoint['groundtruth'])

if __name__ == '__main__':
    main()

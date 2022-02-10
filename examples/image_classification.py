import os
import argparse
import uuid
import json
import random
import datetime

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
                'image_classifier_config.json')
            , 'r') as conf_file:
        conf_json = conf_file.read()
        return json.loads(conf_json)

def get_datapoint(config):

    image_index = int(random.random() * (len(config['images'])))

    return {
        'tags': config['images'][image_index]['tags'],
        'prediction': config['images'][image_index]['prediction'],
        'confidence': config['images'][image_index]['confidence'],
        'embeddings': config['images'][image_index]['embeddings'],
        'image_metadata': {
            'uri': config['images'][image_index]['url'],
            'rotation': config['images'][image_index]['features']['rotation']
        },
        'groundtruth': config['images'][image_index]['class']
    }

def main(args):

    config = load_config()

    dioptra_logger = Logger(api_key=API_KEY)

    model_id = 'document_classification'
    model_version = 'v1.1'
    dataset_id = None
    benchmark_id = None

    if args.benchmark:
        dataset_id = 'doc_classification_benchmark'
        benchmark_id = str(uuid.uuid4())

    for _ in range(NUMBER_OF_EVENTS):

        request_timestamp = datetime.datetime.utcnow()
        request_uuid = str(uuid.uuid4())

        datapoint = get_datapoint(config)

        # After a prediction we log the prediction, features and tags
        print('Sending a datapoint')
        dioptra_logger.log(
            model_id=model_id,
            model_version=model_version,
            model_type=SupportedTypes.IMAGE_CLASSIFIER,
            dataset_id=dataset_id,
            benchmark_id=benchmark_id,
            timestamp=request_timestamp,
            request_id=request_uuid,
            prediction=datapoint['prediction'],
            confidence=datapoint['confidence'],
            embeddings=datapoint['embeddings'],
            image_metadata=datapoint['image_metadata'],
            tags=datapoint['tags'])

        # We can log the groundtruth asynchronously
        print('Sending a groundtruth')
        dioptra_logger.commit(
            request_id=request_uuid,
            timestamp=request_timestamp,
            model_type=SupportedTypes.IMAGE_CLASSIFIER,
            groundtruth=datapoint['groundtruth'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    my_args = parser.parse_args()
    main(my_args)

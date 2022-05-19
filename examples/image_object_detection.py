import os
from zipfile import ZipFile
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
                'image_object_detection_config.json')
            , 'r') as conf_file:
        conf_json = conf_file.read()
        return json.loads(conf_json)


def get_datapoint(config):

    image_index = int(random.random() * (len(config)))
    found = True

    pred_bbox = []
    gt_bbox = []

    for prediction in config[image_index]['prediction']:
        pred_bbox.append({
            'left': prediction['bbox'][0],
            'top': prediction['bbox'][1],
            'width': prediction['bbox'][2],
            'height': prediction['bbox'][3],
            'class_name': prediction['class_name'],
            'confidence': prediction['confidence']
            })

    for groundtruth in config[image_index]['groundtruth']:
        gt_bbox.append({
            'left': groundtruth['bbox'][0],
            'top': groundtruth['bbox'][1],
            'width': groundtruth['bbox'][2],
            'height': groundtruth['bbox'][3],
            'class_name': groundtruth['class_name']
            })

    return {
        'tags': config[image_index]['tags'],
        'prediction': pred_bbox,
        'embeddings': config[image_index]['embeddings'],
        'image_metadata': {
            'uri': config[image_index]['url'],
            'rotation': config[image_index]['features']['rotation']
        },
        'groundtruth': gt_bbox
    }

def main():

    config = load_config()

    dioptra_logger = Logger(api_key=API_KEY)

    model_id = 'document_extraction'
    model_version = 'v1.1'

    for _ in range(NUMBER_OF_EVENTS):

        request_timestamp = datetime.datetime.utcnow().isoformat()
        request_uuid = str(uuid.uuid4())

        datapoint = get_datapoint(config)

        # After a prediction we log the prediction, features and tags
        print('Sending a datapoint')
        dioptra_logger.commit(
            request_id=request_uuid,
            model_id=model_id,
            model_version=model_version,
            model_type=SupportedTypes.OBJECT_DETECTION,
            timestamp=request_timestamp,
            tags=datapoint['tags'],
            prediction=datapoint['prediction'],
            embeddings=datapoint['embeddings'],
            image_metadata=datapoint['image_metadata'],
            groundtruth=datapoint['groundtruth'])

if __name__ == '__main__':
    main()

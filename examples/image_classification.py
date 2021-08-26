import argparse
import uuid
import datetime

from image_classifier_generator import (
    get_random_image_index,
    generate_groundtruth,
    generate_datapoint
)

from dioptra.api import Logger

API_KEY = '4gHEEZD5pA9yHXHiSZi5w1pMr8u8bGn53VBYkza6'
NUMBER_OF_EVENTS = 10000

def main(args):

    dioptra_logger = Logger(api_key=API_KEY)

    model_id = 'document_classification'
    model_version = 'v1.1'

    for _ in range(NUMBER_OF_EVENTS):

        request_timestamp = datetime.datetime.now().isoformat()
        request_uuid = str(uuid.uuid4())

        image_index = get_random_image_index(args.cheques)
        generated_datapoint = generate_datapoint(image_index, args.cheques)
        generated_groundtruth = generate_groundtruth(image_index)

        # After a prediction we log the prediction, features and tags
        dioptra_logger.log(
            model_id=model_id,
            model_version=model_version,
            timestamp=request_timestamp,
            request_id=request_uuid,
            prediction=generated_datapoint['model_prediction'],
            confidence=generated_datapoint['model_confidence'],
            features=generated_datapoint['image_features'],
            embeddings=generated_datapoint['image_embeddings'],
            image_url=generated_datapoint['image_url'],
            tags=generated_datapoint['request_tags'])

        # We can log the groundtruth asynchronously
        dioptra_logger.log(
            request_id=request_uuid,
            groundtruth=generated_groundtruth)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cheques',
        help='Uses the cheque scenario that causes a data drift',
        action='store_true')
    args = parser.parse_args()
    main(args)

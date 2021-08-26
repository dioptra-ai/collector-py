import time
import json

from dioptra.client import Client
from dioptra.utils import (
    validate_tags,
    validate_embeddings,
    validate_features,
    validate_groundtruth,
    validate_prediction,
    validate_confidence,
    validate_image_url,
    validate_timestamp,
    add_prefix_to_keys
)

class Logger:
    """
    Dioptra logger client

    """

    def __init__(
        self,
        api_key,
        endpoint_url='https://jchf4wca8i.execute-api.us-east-2.amazonaws.com/demo/test'
    ):
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        self.event_url = endpoint_url + '/event'
        self._headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json"
        }

    def log(
        self,
        request_id,
        model_id=None,
        model_version=None,
        timestamp=None,
        groundtruth=None,
        prediction=None,
        confidence=None,
        features=None,
        embeddings=None,
        image_url=None,
        tags=None
    ):

        payload = self.package_payload(
            model_id,
            model_version,
            timestamp,
            request_id,
            groundtruth,
            prediction,
            confidence,
            features,
            embeddings,
            image_url,
            tags
        )

        client = Client()
        my_response = client.call(
            payload=payload,
            endpoint_url=self.endpoint_url,
            headers=self._headers)

        return my_response

    def package_payload(
        self,
        model_id,
        model_version,
        timestamp,
        request_id,
        groundtruth,
        prediction,
        confidence,
        features,
        embeddings,
        image_url,
        tags
    ):

        payload = {
            'request_id': request_id
        }

        if model_id:
            payload['model_id'] = model_id

        if model_version:
            payload['model_version'] = model_version

        if timestamp and validate_timestamp(groundtruth):
            payload['timestamp'] = timestamp

        if groundtruth and validate_groundtruth(groundtruth):
            payload['groundtruth'] = groundtruth

        if prediction and validate_prediction(prediction):
            payload['prediction'] = prediction

        if confidence and validate_confidence(confidence):
            payload['confidence'] = confidence

        if features and validate_features(features):
            prefixed_features = add_prefix_to_keys(features, 'feature')
            payload.update(prefixed_features)

        if image_url and validate_image_url(image_url):
            payload['feature.image_url'] = image_url

        if embeddings and validate_embeddings(embeddings):
            payload['feature.embeddings'] = embeddings

        if tags and validate_tags(tags):
            prefixed_tags = add_prefix_to_keys(tags, 'tag')
            payload.update(prefixed_tags)

        return payload

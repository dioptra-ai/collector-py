import logging
import datetime

import dioptra
from dioptra.client import Client
from dioptra.utils import (
    validate_tags,
    validate_embeddings,
    validate_features,
    validate_annotations,
    validate_confidence,
    validate_image_metadata,
    validate_timestamp,
    validate_text_metadata,
    validate_text,
    validate_audio_metadata,
    validate_model_type,
    validate_input_data
)

class Logger:
    """
    Dioptra logger client

    """

    def __init__(
        self,
        api_key,
        endpoint_url='https://api.dioptra.ai'
    ):
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        self.event_url = endpoint_url + '/events'
        self._headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json'
        }

    def log(
        self,
        request_id,
        timestamp=None,
        dataset_id=None,
        benchmark_id=None,
        model_id=None,
        model_version=None,
        model_type=None,
        groundtruth=None,
        prediction=None,
        confidence=None,
        features=None,
        embeddings=None,
        input_data=None,
        image_metadata=None,
        text=None,
        text_metadata=None,
        tags=None,
        audio_metadata=None
    ):
        """
        The log interface sends a set of events to the dioptra api endpoint.
        Yet, the events are not commited yet.
        As long as a event is not commited, it won't be available for analysis.
        Once a committed event with the same `request_id` is sent,
        it will be aggregated with previous events in a session that share the same `request_id`
        A session is 1 day long.

        """

        payload = self.package_payload(
            model_id,
            dataset_id,
            benchmark_id,
            model_version,
            model_type,
            timestamp,
            request_id,
            groundtruth,
            prediction,
            confidence,
            features,
            embeddings,
            input_data,
            image_metadata,
            text,
            text_metadata,
            tags,
            audio_metadata,
            False
        )

        client = Client()
        my_response = client.call(
            payload=payload,
            endpoint_url=self.event_url,
            headers=self._headers)

        return my_response

    def commit(
        self,
        request_id,
        timestamp=None,
        model_id=None,
        dataset_id=None,
        benchmark_id=None,
        model_version=None,
        model_type=None,
        groundtruth=None,
        prediction=None,
        confidence=None,
        features=None,
        embeddings=None,
        input_data=None,
        image_metadata=None,
        text=None,
        text_metadata=None,
        tags=None,
        audio_metadata=None
    ):
        """
        The commit interface sends a set of events to the dioptra api endpoint and commits them.
        A committed event triggers the aggregation of all previous events with the same `request_id`
        in the same session. A seesion is 1 day long.
        Once a committed all events are available for analysis in the dioptra UI.

        """

        payload = self.package_payload(
            model_id,
            dataset_id,
            benchmark_id,
            model_version,
            model_type,
            timestamp,
            request_id,
            groundtruth,
            prediction,
            confidence,
            features,
            embeddings,
            input_data,
            image_metadata,
            text,
            text_metadata,
            tags,
            audio_metadata,
            True
        )

        client = Client()
        my_response = client.call(
            payload=payload,
            endpoint_url=self.event_url,
            headers=self._headers)

        return my_response

    def package_payload(
        self,
        model_id,
        dataset_id,
        benchmark_id,
        model_version,
        model_type,
        timestamp,
        request_id,
        groundtruth,
        prediction,
        confidence,
        features,
        embeddings,
        input_data,
        image_metadata,
        text,
        text_metadata,
        tags,
        audio_metadata,
        committed
    ):

        payload = {
            'request_id': request_id,
            'committed': committed,
            'api_version': dioptra.__version__
        }

        if model_id:
            payload['model_id'] = model_id

        if dataset_id:
            payload['dataset_id'] = dataset_id

        if benchmark_id:
            payload['benchmark_id'] = benchmark_id

        if model_version:
            payload['model_version'] = model_version

        if model_type:
            if validate_model_type(model_type):
                payload['model_type'] = model_type.model_type.value
                payload['input_type'] = model_type.input_type.value
            else:
                logging.warning('model_type didn\'t validate. Ignoring...')

        if timestamp and validate_timestamp(timestamp):
            payload['timestamp'] = timestamp.isoformat()
        else:
            logging.warning('timestamp didn\'t validate. Replacing with current timestamp...')
            payload['timestamp'] = datetime.datetime.utcnow().isoformat()

        if groundtruth:
            if model_type:
                if validate_model_type(model_type):
                    if validate_annotations(groundtruth, model_type):
                        payload['groundtruth'] = groundtruth
                    else:
                        logging.warning('groundtruth didn\'t validate. Ignoring...')
                else:
                    logging.warning('model_type didn\'t validate. Ignoring groundtruth...')
            else:
                logging.warning('no model model_type defined. Ignoring groundtruth...')

        if prediction:
            if model_type:
                if validate_model_type(model_type):
                    if validate_annotations(prediction, model_type):
                        payload['prediction'] = prediction
                    else:
                        logging.warning('prediction didn\'t validate. Ignoring...')
                else:
                    logging.warning('model_type didn\'t validate. Ignoring prediction...')
            else:
                logging.warning('no model model_type defined. Ignoring prediction...')

        if confidence:
            if validate_confidence(confidence):
                payload['confidence'] = confidence
            else:
                logging.warning('confidence didn\'t validate. Ignoring...')

        if features:
            if validate_features(features):
                payload['features'] = features
            else:
                logging.warning('features didn\'t validate. Ignoring...')

        if image_metadata:
            if validate_image_metadata(image_metadata):
                payload['image_metadata'] = image_metadata
            else:
                logging.warning('image_metadata didn\'t validate. Ignoring...')

        if text:
            if validate_text(text):
                payload['text'] = text
            else:
                logging.warning('text didn\'t validate. Ignoring...')

        if text_metadata:
            if validate_text_metadata(text_metadata):
                payload['text_metadata'] = text_metadata
            else:
                logging.warning('text_metadata didn\'t validate. Ignoring...')

        if embeddings:
            if validate_embeddings(embeddings):
                payload['embeddings'] = embeddings
            else:
                logging.warning('embeddings didn\'t validate. Ignoring...')

        if tags:
            if validate_tags(tags):
                payload['tags'] = tags
            else:
                logging.warning('tags didn\'t validate. Ignoring...')

        if audio_metadata:
            if validate_audio_metadata(audio_metadata):
                payload['audio_metadata'] = audio_metadata
            else:
                logging.warning('audio_metadata didn\'t validate. Ignoring...')

        if input_data:
            if model_type:
                if validate_model_type(model_type):
                    if validate_input_data(input_data, model_type):
                        payload['input_data'] = input_data
                    else:
                        logging.warning('input_data didn\'t validate. Ignoring...')
                else:
                    logging.warning('model_type didn\'t validate. Ignoring input_data...')
            else:
                logging.warning('no model model_type defined. Ignoring input_data...')

        return [payload]

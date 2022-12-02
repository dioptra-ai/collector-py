import logging
import datetime
import dateutil.parser
import orjson
import os
import gzip
import threading
import random
import traceback
from smart_open import open as smart_open

from tqdm import tqdm
import boto3
from boto3.s3.transfer import TransferConfig
from botocore import UNSIGNED
from botocore.client import Config
import uuid

import dioptra
from dioptra.client import Client
from dioptra.supported_types import SupportedTypes
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
    validate_video_metadata,
    validate_model_type,
    validate_input_data
)

s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

class FileProgress(object):
    def __init__(self, filenames):
        self._size = 0

        for filename in filenames:
            self._size += float(os.path.getsize(filename))

        self._lock = threading.Lock()
        self._pbar = tqdm(
            total=self._size,
            unit='B',
            desc='Uploading to Dioptra'
        )

    def __call__(self, bytes_amount):
        with self._lock:
            self._pbar.update(bytes_amount)

BATCH_INGEST_MAX_SIZE = 5000

class Logger:
    def __init__(
        self,
        api_key,
        endpoint_url='https://api.dioptra.ai',
        synchronous_mode=False,
        max_workers=5,
        batch_size=20,
        queue_size=100
    ):
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        self.event_url = endpoint_url + '/events'
        self._headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json'
        }
        # TODO: reenable setting synchronous_mode
        self.client = Client(max_workers, queue_size, synchronous_mode=True)
        self.batch = []
        self.batch_size = batch_size

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
        audio_metadata=None,
        validate_sample=True
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
            False,
            validate_sample=1 if validate_sample else 0
        )

        self.client.call(
            payload=payload,
            endpoint_url=self.event_url,
            headers=self._headers)

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
        audio_metadata=None,
        validate_sample=True
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
            True,
            validate_sample=1 if validate_sample else 0
        )

        self.client.call(
            payload=payload,
            endpoint_url=self.event_url,
            headers=self._headers)

    def add_to_batch_log(
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
        audio_metadata=None,
        validate_sample=True
    ):

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
            False,
            validate_sample=1 if validate_sample else 0
        )

        self.batch.extend(payload)


    def add_to_batch_commit(
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
        audio_metadata=None,
        validate_sample=True
    ):

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
            True,
            validate_sample=1 if validate_sample else 0
        )

        self.batch.extend(payload)

    def submit_batch(self):
        batches = [
            self.batch[i:i + self.batch_size] for i in range(0, len(self.batch), self.batch_size)]
        for batch in tqdm(batches, desc='Batch upload progress'):
            self.client.call(
                payload=batch,
                endpoint_url=self.event_url,
                headers=self._headers)


    def commit_ndjson_file(self, ndjson_filename, validate_sample=0.01, batch_bucket='dioptra-batch-input-prod'):
        """
        Loading the config from file
        """
        events_batch = []
        progress = tqdm(
            unit='Event',
            desc='Uploading to Dioptra'
        )
        def flush_batch(events_batch, progress):
            batch_key = str(uuid.uuid4()) + '.ndjson.gz'
            batch_tmp_filepath = f'/tmp/dioptra/{batch_key}'

            try:
                events_payload = b'\n'.join([orjson.dumps(e) for e in events_batch])
                os.makedirs(os.path.dirname(batch_tmp_filepath), exist_ok=True)
                with open(batch_tmp_filepath, 'wb') as tmp_file:
                    tmp_file.write(gzip.compress(events_payload))

                GB = 1024 ** 3
                transfer = boto3.s3.transfer.S3Transfer(s3_client, config=TransferConfig(multipart_threshold=5*GB))
                transfer.upload_file(
                    batch_tmp_filepath, batch_bucket, batch_key,
                    extra_args={'Metadata': {'x-api-key': self.api_key}}
                )
                progress.update(len(events_batch))
            except Exception as e:
                print(traceback.format_exc())
            finally:
                os.remove(batch_tmp_filepath)

        for _ in range(1):
            for line in smart_open(ndjson_filename):
                try:
                    parsed_line = orjson.loads(line)
                except:
                    raise Exception('Failed to parse json datapoint: ' + str(line))

                events_batch.extend(self.package_payload(**parsed_line, validate_sample=validate_sample))

                if len(events_batch) >= BATCH_INGEST_MAX_SIZE:
                    flush_batch(events_batch, progress)
                    events_batch = []

        if events_batch:
            flush_batch(events_batch, progress)

    def package_payload(
        self,
        model_id=None,
        dataset_id=None,
        benchmark_id=None,
        model_version=None,
        model_type=None,
        timestamp=None,
        request_id=None,
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
        audio_metadata=None,
        committed=None,
        video_metadata=None,
        validate_sample=1
    ):
        skip_validation = random.random() > validate_sample

        payload = {
            'committed': committed,
            'api_version': dioptra.__version__
        }

        if request_id:
            payload['request_id'] = request_id

        if model_id:
            payload['model_id'] = model_id

        if dataset_id:
            payload['dataset_id'] = dataset_id

        if benchmark_id:
            payload['benchmark_id'] = benchmark_id

        if model_version:
            payload['model_version'] = model_version

        if model_type:
            if isinstance(model_type, str):
                model_type = SupportedTypes[model_type]

            if validate_model_type(model_type):
                payload['model_type'] = model_type.model_type.value
                payload['input_type'] = model_type.input_type.value
            else:
                logging.warning('Invalid model_type. Skipping datapoint...')
                return []

        if timestamp:
            if isinstance(timestamp, str):
                try:
                    timestamp = dateutil.parser.isoparse(timestamp)
                except:
                    logging.warning('Could not parse string timestamp from ISO format...')

            if validate_timestamp(timestamp):
                payload['timestamp'] = timestamp.isoformat()
            else:
                logging.warning('Invalid timestamp. Replacing with current time...')
                payload['timestamp'] = datetime.datetime.utcnow().isoformat()

        if groundtruth:
            if model_type:
                if skip_validation or validate_model_type(model_type):
                    if skip_validation or validate_annotations(groundtruth, model_type):
                        payload['groundtruth'] = groundtruth
                    else:
                        logging.warning('Invalid groundtruth. Skipping datapoint...')
                        return []
                else:
                    logging.warning('Invalid model_type. Skipping datapoint...')
                    return []
            else:
                logging.warning('No model_type. Skipping datapoint...')
                return []

        if prediction:
            if model_type:
                if skip_validation or validate_model_type(model_type):
                    if skip_validation or validate_annotations(prediction, model_type):
                        payload['prediction'] = prediction
                    else:
                        logging.warning('Invalid prediction. Skipping datapoint...')
                        return []
                else:
                    logging.warning('Invalid model_type. Skipping datapoint...')
                    return []
            else:
                logging.warning('No model_type defined. Skipping datapoint...')
                return []

        if confidence:
            if skip_validation or validate_confidence(confidence):
                payload['confidence'] = confidence
            else:
                logging.warning('Invalid confidence. Skipping datapoint...')
                return []

        if features:
            if skip_validation or validate_features(features):
                payload['features'] = features
            else:
                logging.warning('Invalid features. Skipping datapoint...')
                return []

        if image_metadata:
            if skip_validation or validate_image_metadata(image_metadata):
                payload['image_metadata'] = image_metadata
            else:
                logging.warning('Invalid image_metadata. Skipping datapoint...')
                return []

        if text:
            if skip_validation or validate_text(text):
                payload['text'] = text
            else:
                logging.warning('Invalid text. Skipping datapoint...')
                return []

        if text_metadata:
            if skip_validation or validate_text_metadata(text_metadata):
                payload['text_metadata'] = text_metadata
            else:
                logging.warning('Invalid text_metadata. Skipping datapoint...')
                return []

        if embeddings:
            if skip_validation or validate_embeddings(embeddings):
                payload['embeddings'] = embeddings
            else:
                logging.warning('Invalid embeddings. Skipping datapoint...')
                return []

        if tags:
            if skip_validation or validate_tags(tags):
                payload['tags'] = tags
            else:
                logging.warning('Invalid tags. Skipping datapoint...')
                return []

        if audio_metadata:
            if skip_validation or validate_audio_metadata(audio_metadata):
                payload['audio_metadata'] = audio_metadata
            else:
                logging.warning('Invalid audio_metadata. Skipping datapoint...')
                return []

        if video_metadata:
            if skip_validation or validate_video_metadata(video_metadata):
                payload['video_metadata'] = video_metadata
            else:
                logging.warning('Invalid video_metadata. Skipping datapoint...')
                return []


        if input_data:
            if model_type:
                if skip_validation or validate_model_type(model_type):
                    if skip_validation or validate_input_data(input_data, model_type):
                        payload['input_data'] = input_data
                    else:
                        logging.warning('Invalid input_data. Skipping datapoint...')
                        return []
                else:
                    logging.warning('Invalid model_type. Skipping datapoint...')
                    return []
            else:
                logging.warning('No model model_type defined. Skipping datapoint...')
                return []

        return [payload]

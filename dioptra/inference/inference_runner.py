import os
import tempfile
import json
import gzip
from json import JSONEncoder
import copy

import numpy as np
import boto3
from boto3.s3.transfer import TransferConfig
from tqdm import tqdm

from dioptra.lake.utils import upload_to_lake_from_s3


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class InferenceRunner():
    def __init__(self):

        api_key = os.environ.get('DIOPTRA_API_KEY', None)
        if api_key is None:
            raise RuntimeError('DIOPTRA_API_KEY env var is not set')

        s3_bucket = os.environ.get('DIOPTRA_UPLOAD_BUCKET', None)

        if s3_bucket is None:
            raise RuntimeError('DIOPTRA_UPLOAD_BUCKET env var is not set')

        self.api_key = api_key
        self.s3_bucket = s3_bucket
        self.max_batch_size = 1000
        self.result_files = []

    def ingest_results(self):
        s3_client = boto3.client('s3')
        ingestion_responses = []
        for result_file in tqdm(copy.copy(self.result_files), desc='uploading results...'):
            GB = 1024 ** 3
            transfer = boto3.s3.transfer.S3Transfer(s3_client, config=TransferConfig(multipart_threshold=5*GB))
            transfer.upload_file(
                result_file.name, self.s3_bucket, os.path.basename(result_file.name),
                extra_args={'Metadata': {'x-api-key': self.api_key}}
            )
            ingestion_responses.append(upload_to_lake_from_s3(self.s3_bucket, os.path.basename(result_file.name)))
            result_file.close()
            self.result_files.remove(result_file)

        return ingestion_responses


    def _dump_data(self, records):
        print('dumping to disk ...')
        temp = tempfile.NamedTemporaryFile(prefix='dioptra_results_', suffix='.ndjson.gz')
        payload = b'\n'.join([json.dumps(record, cls=NumpyArrayEncoder).encode('utf-8') for record in records])
        temp.write(gzip.compress(payload))
        self.result_files.append(temp)

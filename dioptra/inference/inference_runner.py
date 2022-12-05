import os
import tempfile
import orjson
import mgzip
import uuid
from datetime import datetime

import numpy as np
import boto3
from boto3.s3.transfer import TransferConfig
from tqdm import tqdm

from dioptra.lake.utils import upload_to_lake_from_s3


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
        self.ingestion_responses = {}

    def _ingest_data(self, records):
        print('ingesting data ...')

        s3_client = boto3.client('s3')
        transfer = boto3.s3.transfer.S3Transfer(s3_client, config=TransferConfig(multipart_threshold=5*1024**3))
        file_name = f'{str(uuid.uuid4())}_{datetime.utcnow().isoformat()}.ndjson.gz'
        
        payload = b'\n'.join([orjson.dumps(record, option=orjson.OPT_SERIALIZE_NUMPY)for record in records])
        compressed_payload = mgzip.compress(payload, compresslevel=2)
        
        s3_client.put_object(
            Body=compressed_payload,
            Bucket=self.s3_bucket,
            Key=file_name,
        )
        
        self.ingestion_responses[file_name] = upload_to_lake_from_s3(self.s3_bucket, file_name)

import os
import orjson
import mgzip
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import smart_open

from dioptra.lake.utils import wait_for_upload, upload_to_lake_via_object_store


class InferenceRunner():
    def __init__(self):
        self.max_batch_size = 1000
        self.uploads = []

    def _ingest_data(self, records):
        print('ingesting data ...')
        self.uploads.append(upload_to_lake_via_object_store(records))

    def wait_for_uploads(self):
        with ThreadPoolExecutor() as executor:
            upload_ids = list(map(lambda u: u['id'], self.uploads))
            return list(executor.map(wait_for_upload, upload_ids, timeout=900, chunksize=10))

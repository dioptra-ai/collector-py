import os
import orjson
import mgzip
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import smart_open

from dioptra.lake.utils import (
    wait_for_upload,
    upload_to_lake_via_object_store,
    _resolve_mc_drop_out_predictions,
    store_to_local_cache
)

class InferenceRunner():
    def __init__(self):
        self.max_batch_size = 1000
        self.uploads = []
        self.use_local_storage = False

    def _ingest_data(self, records):
        print('ingesting data ...')
        if self.use_local_storage:
            self.uploads.append(store_to_local_cache(records))
        else:
            self.uploads.append(upload_to_lake_via_object_store(records))

    def wait_for_uploads(self):
        with ThreadPoolExecutor() as executor:
            upload_ids = list(map(lambda u: u['id'], self.uploads))
            return list(executor.map(wait_for_upload, upload_ids, timeout=900, chunksize=10))

    def _resolve_records(self, samples_records):
        if len(samples_records) == 1:
            return samples_records[0]
        resolved_records = []
        for record_idx in range(len(samples_records[0])):
            resolved_record = {}
            predictions = []
            for sample_idx in range(len(samples_records)):
                resolved_record.update(samples_records[sample_idx][record_idx])
                predictions.append(samples_records[sample_idx][record_idx]['prediction'])
            resolved_record['prediction'] = _resolve_mc_drop_out_predictions(predictions)
            resolved_records.append(resolved_record)
        return resolved_records

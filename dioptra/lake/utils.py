import os
import io
import requests
import base64
import time
from multiprocessing import Pool
import io
import uuid
from datetime import datetime, timedelta

import smart_open
import PIL
import pandas as pd
import numpy as np
import lz4.frame
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from google.cloud import storage
import json
import tqdm
import orjson
import mgzip

from . import ingestion

# TODO: figure out how to return status codes form lambda functions.
# See the wip_return_error_code branch of the infrastructure repo.
def _raise_for_apigateway_errormessage(response):
    if response is not None and 'errorMessage' in response:
        raise RuntimeError(response['errorMessage'])

DIOPTRA_API_ENDPOINT = os.environ.get('DIOPTRA_API_ENDPOINT', 'https://api.dioptra.ai/events')
DIOPTRA_APP_ENDPOINT = os.environ.get('DIOPTRA_APP_ENDPOINT', 'https://app.dioptra.ai')
# We ask for a definitive signal to disable but using the positive form is easier in code.
DIOPTRA_SSL_VERIFY = not os.environ.get('DIOPTRA_SSL_NOVERIFY', 'False') == 'True'
DIOPTRA_API_KEY = os.environ.get('DIOPTRA_API_KEY', None)

def query_dioptra_app(method, path, json=None, files=None):
    api_key = os.environ.get('DIOPTRA_API_KEY', None)
    if api_key is None:
        raise RuntimeError('DIOPTRA_API_KEY env var is not set')

    r = None
    try:
        r = getattr(requests, method.lower())(
            url=f'{DIOPTRA_APP_ENDPOINT}{path}',
            verify=DIOPTRA_SSL_VERIFY,
            headers={'x-api-key': api_key},
            json=json,
            files=files
        )
        r.raise_for_status()
    except requests.exceptions.RequestException as err:
        if r is None:
            raise err
        else:
            error_body = r.json()
            error_message = error_body.get('error', error_body).get('message', error_body)
            raise Exception(f'Api Error: {error_message}')
    else:
        return r.json()

def select_datapoints(filters, limit=None, order_by=None, desc=None, fields=['*'], offset=0):
    """
    Select metadata from the data lake

    Parameters:
        filters: dioptra style filters to select the data to be queried from
        limit: limit to selected the data
        order_by: field to use to sort the data to control how limit is performed
        desc: whether to order by dec or not
        fields: array of fields to be queried. By default all fields are queried
    """

    return pd.DataFrame(query_dioptra_app(
        'POST',
        '/api/datapoints/select',
        {
            'selectColumns': fields,
            'filters': filters,
            **({'limit': limit} if limit is not None else {}),
            **({'orderBy': order_by} if order_by is not None else {}),
            **({'desc': desc} if desc is not None else {}),
            'offset': offset
        }
    ))

def select_predictions(filters, limit=None, order_by=None, desc=None, fields=['*'], offset=0):
    """
    Select metadata from the data lake

    Parameters:
        filters: dioptra style filters to select the data to be queried from
        limit: limit to selected the data
        order_by: field to use to sort the data to control how limit is performed
        desc: whether to order by dec or not
        fields: array of fields to be queried. By default all fields are queried
    """

    return pd.DataFrame(query_dioptra_app(
        'POST',
        '/api/predictions/select',
        {
            'selectColumns': fields,
            'filters': filters,
            **({'limit': limit} if limit is not None else {}),
            **({'orderBy': order_by} if order_by is not None else {}),
            **({'desc': desc} if desc is not None else {}),
            'offset': offset
        }
    ))

def select_groundtruths(filters, limit=None, order_by=None, desc=None, fields=['*'], offset=0):
    """
    Select metadata from the data lake

    Parameters:
        filters: dioptra style filters to select the data to be queried from
        limit: limit to selected the data
        order_by: field to use to sort the data to control how limit is performed
        desc: whether to order by dec or not
        fields: array of fields to be queried. By default all fields are queried
    """

    return pd.DataFrame(query_dioptra_app(
        'POST',
        '/api/groundtruths/select',
        {
            'selectColumns': fields,
            'filters': filters,
            **({'limit': limit} if limit is not None else {}),
            **({'orderBy': order_by} if order_by is not None else {}),
            **({'desc': desc} if desc is not None else {}),
            'offset': offset
        }
    ))

def select_bboxes(filters, limit=None, order_by=None, desc=None, fields=['*'], offset=0):

    return pd.DataFrame(query_dioptra_app(
        'POST',
        '/api/bboxes/select',
        {
            'selectColumns': fields,
            'filters': filters,
            **({'limit': limit} if limit is not None else {}),
            **({'orderBy': order_by} if order_by is not None else {}),
            **({'desc': desc} if desc is not None else {}),
            'offset': offset
        }
    ))

def delete_predictions(prediction_ids):
    """
    Delete predictions from the data lake

    Parameters:
        prediction_ids: list of prediction ids to delete
    """

    return query_dioptra_app(
        'POST',
        '/api/predictions/delete',
        {
            'predictionIds': prediction_ids
        }
    )

def delete_groundtruths(groundtruth_ids):
    """
    Delete groundtruths from the data lake

    Parameters:
        groundtruth_ids: list of groundtruth ids to delete
    """

    return query_dioptra_app(
        'POST',
        '/api/groundtruths/delete',
        {
            'groundtruthIds': groundtruth_ids
        }
    )

def delete_datapoints(filters, limit=None, order_by=None, desc=None):
    """
    Delete metadata from the data lake

    Parameters:
        filters: dioptra style filters to select the data to be deleted
        limit: limit to selected the data
        order_by: field to use to sort the data to control how limit is performed
        desc: whether to order by dec or not
    """

    return query_dioptra_app(
        'POST',
        '/api/datapoints/delete',
        {
            'filters': filters
        }
    )

def stream_to_lake(records):
    """
    Uploading metadata to the data lake. Maximum payload ~1MB.

    Parameters:
        records: array of dipotra style records. See teh doc for accepted formats
    """

    api_key = os.environ.get('DIOPTRA_API_KEY', None)
    if api_key is None:
        raise RuntimeError('DIOPTRA_API_KEY env var is not set')

    try:
        r = requests.post(DIOPTRA_API_ENDPOINT, verify=DIOPTRA_SSL_VERIFY, headers={
            'content-type': 'application/json',
            'x-api-key': api_key,
            'host': 'api.dioptra.ai'
        }, json={
            'records': ingestion.process_records(records)
        })
        r.raise_for_status()
        response = r.json()
        _raise_for_apigateway_errormessage(response)
    except requests.exceptions.RequestException as err:
        print('There was an error uploading to the lake ...')
        raise err

    return response

def upload_to_lake(records, disable_batching=False):
    # Send records as ndjson in a multipart POST query to /api/ingestion/upload
    # on the dioptra app endpoint.
    file_upload = query_dioptra_app('POST', '/api/ingestion/upload', files={
        'file': ('records.ndjson', '\r'.join([json.dumps(r) for r in records]))
    })

    data_upload = query_dioptra_app('POST', '/api/ingestion/ingest',
        {'url': file_upload['url']} if disable_batching else {'urls': [file_upload['url']]}
    )

    if 'id' in data_upload:
        print(f'Uploaded {len(records)} records to the lake. View the upload status at {DIOPTRA_APP_ENDPOINT}/settings/uploads/{data_upload["id"]}')
    else:
        print(f'Uploaded {len(records)} records to the lake. Upload status: {data_upload}')

    return data_upload

def upload_to_lake_via_object_store(records, custom_path='', disable_batching=False, offset=None, limit=None):
    """
    Uploading metadata to the data lake via an s3 or gcs bucket.
    Should be the prefered upload method for large records
    Object Store credentials should be configured

    Parameters:
        records: array of dipotra style records. See the doc for accepted formats
    """

    api_key = os.environ.get('DIOPTRA_API_KEY', None)
    if api_key is None:
        raise RuntimeError('DIOPTRA_API_KEY env var is not set')

    object_store_bucket = os.environ.get('DIOPTRA_UPLOAD_BUCKET', None)
    if object_store_bucket is None:
        raise RuntimeError('DIOPTRA_UPLOAD_BUCKET env var is not set')

    prefix = os.environ.get('DIOPTRA_UPLOAD_PREFIX', '')
    storage_type = os.environ.get('DIOPTRA_UPLOAD_STORAGE_TYPE', 's3')
    # s3 url is of form: s3://bucket_name/path/to/file
    # gcs url is of form: gs://bucket_name/path/to/file
    file_name = os.path.join(
        prefix, custom_path, f'{str(uuid.uuid4())}_{datetime.utcnow().isoformat()}.ndjson.gz').strip('/')
    store_url = f'{storage_type}://{os.path.join(object_store_bucket, file_name)}'

    records = ingestion.process_records(records)
    compressed_payload = _compress_to_ndjson(records)

    _upload_to_bucket((compressed_payload, store_url), no_compression=True)

    return upload_to_lake_from_bucket(object_store_bucket, file_name, storage_type, disable_batching, offset, limit)

def upload_to_lake_from_bucket(bucket_name, object_name, storage_type='s3', disable_batching=False, offset=None, limit=None):
    """
    Uploading metadata to the data lake from an s3 bucket
    The data should be a new line delimited JSON and can be compressed with Gzip
    Will generate a signed URL and pass it to the dioptra ingestion
    boto3 credentials should be configured

    Parameters:
        bucket_name: name of the bucket where the records are
        object_name: name of the key in the bucket where the records are
    """

    api_key = os.environ.get('DIOPTRA_API_KEY', None)
    if api_key is None:
        raise RuntimeError('DIOPTRA_API_KEY env var is not set')

    if storage_type == 's3':
        signed_url = _generate_s3_signed_url(bucket_name, object_name)
    elif storage_type == 'gs':
        signed_url = _generate_gs_signed_url(bucket_name, object_name)
    try:
        r = requests.post(DIOPTRA_API_ENDPOINT, verify=DIOPTRA_SSL_VERIFY, headers={
            'content-type': 'application/json',
            'x-api-key': api_key,
            'host': 'api.dioptra.ai'
        }, json={
            'url': signed_url,
            'offset': offset,
            'limit': limit
        } if disable_batching else {
            'urls': [signed_url]
        })
        r.raise_for_status()
        response = r.json()
        _raise_for_apigateway_errormessage(response)
    except requests.exceptions.RequestException as err:
        print('There was an error uploading to the lake ...')
        raise err

    return response

def store_to_local_cache(records):
    """
    Store records to a local cache directory

    Parameters:
        records: array of dipotra style records. See the doc for accepted formats
    """

    cache_dir = os.environ.get(
        'DIOPTRA_LOCAL_CACHE_DIR',
        os.path.join(os.path.expanduser('~'), '.dioptra'))

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    file_name = os.path.join(cache_dir, f'{str(uuid.uuid4())}_{datetime.utcnow().isoformat()}.ndjson.gz')

    compressed_payload = _compress_to_ndjson(records)

    with open(file_name, 'wb') as f:
        f.write(compressed_payload)

    return file_name

def wait_for_upload(upload_id):
    """
    Wait for the upload status to be SUCCEEDED | FAILED | TIMED_OUT | ABORTED
    """
    if upload_id is None:
        raise RuntimeError('upload_id must not be None')

    if isinstance(upload_id, dict) and 'id' in upload_id:
        upload_id = upload_id['id']

    api_key = os.environ.get('DIOPTRA_API_KEY', None)
    if api_key is None:
        raise RuntimeError('DIOPTRA_API_KEY env var is not set')

    sleepTimeSecs = 1
    totalSleepTimeSecs = 0
    try:
        while True:
            if totalSleepTimeSecs > 900:
                raise RuntimeError('Timed out waiting for the upload to finish.')

            r = requests.get(f'{DIOPTRA_APP_ENDPOINT}/api/ingestion/executions/{upload_id}', verify=DIOPTRA_SSL_VERIFY, headers={
                'content-type': 'application/json',
                'x-api-key': api_key
            })
            r.raise_for_status()
            upload = r.json()
            if upload['status'] in ['SUCCEEDED']:
                return upload
            elif upload['status'] in ['FAILED', 'TIMED_OUT', 'ABORTED']:
                raise RuntimeError(
                    f'Upload failed with status {upload["status"]}. See more information in the Dioptra UI: {DIOPTRA_APP_ENDPOINT}/settings/uploads/{upload_id}')
            else:
                time.sleep(sleepTimeSecs)
                totalSleepTimeSecs += sleepTimeSecs
                sleepTimeSecs = min(sleepTimeSecs * 2, 60)
    except requests.exceptions.RequestException as err:
        print('There was an error waiting for the upload to finish ...')
        raise err

def _generate_s3_signed_url(bucket_name, object_name):

    s3_client = boto3.client('s3', config=Config(
        signature_version='s3v4',
        region_name=os.environ.get('AWS_REGION', 'us-east-2')
    ))
    try:
        response = s3_client.generate_presigned_url(
            'get_object', Params={'Bucket': bucket_name, 'Key': object_name}, ExpiresIn=3600)
    except ClientError as err:
        print('There was an error getting a signed URL ...')
        raise err

    return response

def _generate_gs_signed_url(bucket_name, object_name):

    cred_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', None)
    if cred_file is None:
        raise RuntimeError('GOOGLE_APPLICATION_CREDENTIALS env var is not set. See the documentation for more information: https://dioptra.gitbook.io/dioptra-doc/EIKhoPaxsbOt062jkPon/overview/lake-ml/configuring-object-stores')

    with open(cred_file, 'r') as f:
        credentials = json.load(f)
    gcs_client = storage.Client.from_service_account_info(credentials)
    bucket = gcs_client.get_bucket(bucket_name)
    blob = bucket.get_blob(object_name)
    try:
        response = blob.generate_signed_url(
            version='v4',
            expiration= timedelta(hours=1),
            method='GET'
        )
    except ClientError as err:
        print('There was an error getting a signed URL ...')
        raise err

    return response

def _list_dataset_metadata():
    """
    List all the dataset metadata

    """

    return query_dioptra_app('GET', '/api/dataset')

def _list_miner_metadata():
    """
    List all the miner metadata

    """

    return query_dioptra_app('GET', '/api/tasks/miners')


def _compress_to_ndjson(records):
    payload = b'\n'.join(
        [orjson.dumps(record, option=orjson.OPT_SERIALIZE_NUMPY)for record in records])
    return mgzip.compress(payload, compresslevel=9)

def join_on_datapoints(datapoints, groundtruths=None, predictions=None):
    """
    Join datapoints with predictions or groundtruth.
    Returns an object store dataset ready dataframe

    Parameters:
        datapoints: a dataframe containing the datapoints
        predictions: a dataframe containing the predictions
        groundtruths: a dataframe containing the groundtruths
    """

    if predictions is not None:
        join_formatted = predictions.set_index('datapoint')\
                .apply(lambda x: x.to_dict(), axis=1)\
                .to_frame('predictions')\
                .groupby('datapoint')\
                .agg(list)
        datapoints = datapoints.join(join_formatted, on='id')

    if groundtruths is not None:
        join_formatted = groundtruths.set_index('datapoint')\
            .apply(lambda x: x.to_dict(), axis=1)\
            .to_frame('groundtruths')\
            .groupby('datapoint')\
            .agg(list)
        datapoints = datapoints.join(join_formatted, on='id')

    return datapoints

def group_by_uri(datapoints, uri_grouped_transform, num_workers=1):
    """
    Group datapoints by uri

    Parameters:
        datapoints: a dataframe containing the datapoints typically out of the `join_on_datapoints` method
        uri_grouped_transform: the transform to be applied to the grouped row
            the transform should take a tuple for each row with
                the first index being the id,
                the second index being the metadata,
                the third index being the groundtruths if present in the datapoints,
                the fourth index being the predictions if present in the datapoints
            the transform should return a dictionary with the columns to be added/modified to the dataframe
        num_workers: number of parallel workers

    """

    datapoints['uri'] = [row['uri'] for row in datapoints['metadata']]

    aggregation_dict = {}

    for key in datapoints.keys():
        if key not in ['id', 'metadata', 'groundtruths', 'predictions']:
            aggregation_dict[key] = 'first'
        elif key == 'id':
            aggregation_dict[key] = list
        elif key == 'metadata':
            aggregation_dict[key] = list
        elif key == 'groundtruths':
            aggregation_dict[key] = sum
        elif key == 'predictions':
            aggregation_dict[key] = sum

    grouped_df = datapoints.groupby('uri').agg(aggregation_dict)

    zipped_columns = []
    for key in ['id', 'metadata', 'groundtruths', 'predictions']:
        if key in grouped_df.keys():
            zipped_columns.append(grouped_df[key])

    zipped_column = list(zip(*zipped_columns))

    with Pool(num_workers) as my_pool:
        results = list(tqdm.tqdm(
            my_pool.imap(uri_grouped_transform, zipped_column),
            total=len(zipped_column),
            desc='Processing your rows ...',
            ncols=100
        ))

    for key in results[0].keys():
        grouped_df[key] = [result[key] for result in results]

    return grouped_df

def _encode_np_array(np_array, light_compression=False):
    """
    Encode and compress a np array

    Parameters:
        np_array: the np array to be encoded

    """
    if not isinstance(np_array, np.ndarray):
        raise RuntimeError('Can only encode numpy arrays')

    bytes_buffer = io.BytesIO()
    np.save(bytes_buffer, np_array)

    compression_level=lz4.frame.COMPRESSIONLEVEL_MAX
    if light_compression:
        compression_level=lz4.frame.COMPRESSIONLEVEL_MIN

    return base64.b64encode(
        lz4.frame.compress(
            bytes_buffer.getvalue(),
            compression_level=compression_level
        )).decode('ascii')

def _decode_to_np_array(value):
    """
    Decode a compress a np array

    Parameters:
        value: the string containing the np array

    """
    decoded_bytes = lz4.frame.decompress(base64.b64decode(value))
    return np.load(io.BytesIO(decoded_bytes), allow_pickle=True)


def _format_groundtruth(groundtruth, task_type, class_names=None, groundtruth_id=None):
    """
    Utility formatting the groundtruth field according to the model type

    Parameters:
        groundtruth: the groundtruth field
        task_type: the type of gt. Supported types CLASSIFICATION, SEGMENTATION
        class_names: a list of class names. If the groundtruth contains indexes, it will be used to convert them to names

    """
    if getattr(groundtruth, 'numpy', None) is not None: # dealing with Tensorflow Tensors
        groundtruth = groundtruth.numpy()
    if task_type == 'CLASSIFICATION':
        if (isinstance(groundtruth, int) or isinstance(groundtruth, np.integer)) and class_names is not None:
            class_name = class_names[groundtruth]
        else:
            class_name = groundtruth
        return {
            'task_type': task_type,
            'class_name': class_name
        }
    if task_type == 'SEGMENTATION':
        if isinstance(groundtruth, PIL.Image.Image):
            gt_array = np.array(groundtruth).tolist()
        if isinstance(groundtruth, np.ndarray):
            gt_array = groundtruth.tolist()
        if isinstance(groundtruth, list):
            gt_array = groundtruth
        return {
            'task_type': task_type,
            'segmentation_class_mask': gt_array,
            **({'class_names': class_names} if class_names is not None else {}),
            **({'id': groundtruth_id} if groundtruth_id is not None else {})
        }

def _format_prediction(
        logits, embeddings, task_type, model_name, transformed_logits=None, grad_embeddings=None,
        class_names=None, prediction_id=None, channel_last=False
        ):
    """
    Utility formatting the prediction field.

    Parameters:
        logits: the prediction logits (before softmax)
        embeddings: a dictionary containing the embeddings by layer names, or a single embeddings vector.
        task_type: the type of gt. Supported types CLASSIFICATION, SEGMENTATION
        class_names: a list of class names.
            If the groundtruth contains indexes, it will be used to convert them to names
        prediction_id: the id of the prediction to be updated
        channel_last: if the logits and embeddings are in channel last format
    """
    if getattr(logits, 'cpu', None) is not None: # dealing with Torch Tensors
        logits = logits.cpu()
    if getattr(logits, 'numpy', None) is not None: # dealing with Torch & Tensorflow Tensors
        logits = logits.numpy()

    if logits is not None:
        if not isinstance(logits, np.ndarray):
            logits = np.array(logits)
        logits = logits.astype(np.float16)

        if channel_last:
            logits = np.moveaxis(logits, -1, 0)

        logits = logits.tolist()

    my_embeddings = {}
    if embeddings is not None and len(embeddings) > 0:
        for k, v in embeddings.items():
            if getattr(v, 'cpu', None) is not None: # dealing with Torch Tensors
                v = v.cpu()
            if getattr(v, 'numpy', None) is not None: # dealing with Torch & Tensorflow Tensors
                v = v.numpy()
            if not isinstance(v, np.ndarray):
                v = np.array(v)
            my_embeddings[k] = v.astype(np.float16).tolist() if not channel_last \
                else np.moveaxis(v.astype(np.float16), -1, 0).tolist()

    if grad_embeddings is not None and len(grad_embeddings) > 0:
        my_grad_embeddings = {}
        for k, v in grad_embeddings.items():
            if getattr(v, 'cpu', None) is not None:
                v = v.cpu()
            if getattr(v, 'numpy', None) is not None:
                v = v.numpy()
            if not isinstance(v, np.ndarray):
                v = np.array(v)
            my_grad_embeddings[k] = v.astype(np.float16).tolist()

    if task_type in ['CLASSIFICATION', 'SEGMENTATION']:
        return {
            'task_type': task_type,
            'model_name': model_name,
            **({'embeddings': my_embeddings} if my_embeddings is not None and len(my_embeddings) > 0 else {}),
            **({'logits': logits} if logits is not None else {}),
            **({'class_names': class_names} if class_names is not None else {}),
            **({'id': prediction_id} if prediction_id is not None else {})
        }
    if task_type == 'LANE_DETECTION':
        return {
            'task_type': task_type,
            'model_name': model_name,
            **({'lanes': transformed_logits['lanes']} if transformed_logits['lanes'] is not None and len(transformed_logits['lanes']) > 0 else {}),
            **({'embeddings': my_embeddings} if my_embeddings is not None and len(my_embeddings) > 0 else {}),
            **({'grad_embeddings': my_grad_embeddings} if my_grad_embeddings is not None and len(my_grad_embeddings) > 0 else {}),
            **({'id': prediction_id} if prediction_id is not None else {})
        }

def _resolve_mc_drop_out_predictions(predictions):
    return {
        **({'encoded_logits': [p['encoded_logits'] for p in predictions]} if 'encoded_logits' in predictions[0] else {}),
        **({'embeddings': predictions[0]['embeddings']} if 'embeddings' in predictions[0] else {}),
        **({'grad_embeddings': predictions[0]['grad_embeddings']} if 'grad_embeddings' in predictions[0] else {}),
        **({'logits': [p['logits'] for p in predictions]} if 'logits' in predictions[0] else {}),
        **({'class_names': predictions[0]['class_names']} if 'class_names' in predictions[0] else {}),
        **({'task_type': predictions[0]['task_type']} if 'task_type' in predictions[0] else {}),
        **({'model_name': predictions[0]['model_name']} if 'model_name' in predictions[0] else {}),
        **({'id': predictions[0]['id']} if 'id' in predictions[0] else {}),
    }

def _upload_to_bucket(payload, no_compression=False):
    """
    Small utility to upload bytes of objects to a url
    """
    content, url = payload[0], payload[1]
    write_type = 'wb' if isinstance(content, bytes) else 'w'
    compression = smart_open.compression.NO_COMPRESSION if no_compression else smart_open.compression.INFER_FROM_EXTENSION
    with smart_open.open(url, write_type, compression=compression) as file:
        file.write(content)


def upload_image_dataset(
    dataset, dataset_type,
    image_field=None,
    groundtruth_field=None,
    datapoints_metadata=None,
    image_ids=None,
    class_names=None, dataset_metadata=None,
    max_batch_size=200, num_workers=20):
    """
    Upload an image dataset to Dioptra.
    The images will be uploaded to a bucket specified with DIOPTRA_UPLOAD_BUCKET.
    The metadata will be uploaded to Dioptra and point to this bucket

    Parameters:
        dataset: the dataset to upload. Should be iteratable.
        image_field: the name of the field containing the image. Should be in PIL format
        dataset_type: the type of data, Supported CLASSIFICATION, SEMANTIC_SEGMENTATION
        groundtruth_field: the name of the field containing the groundtruth
        datapoints_metadata:
            a list of metadata to be added to each datapoint.
            should already be formatted.
        dataset_metadata:
            metadata to be added to each datapoint.
            should already be formatted.
        image_ids:
            a list of image ids. if present, this will be used to name teh images on S3
        class_names: a list of class names to convert indexes in the groundtruth to class names
        class_names: a dict of tags to be added to the entire dataset
        max_batch_size: the maximum batch uplodd size
        num_workers: number of parallel workers to upload the images to the bucket

    """

    def _upload_data(dataset_metadata, img_payload, num_workers):
        with Pool(num_workers) as my_pool:
            my_pool.map(_upload_to_bucket, img_payload)

        return upload_to_lake_via_object_store(dataset_metadata, 'logs')

    def _build_img_url(storage_type, bucket, prefix, image_id, pil_image):
        img_format = pil_image.format
        root, ext = os.path.splitext(image_id)
        if not ext:
            image_id += f'.{img_format.lower()}'
        return f'{storage_type}://{os.path.join(bucket, prefix, "images", image_id)}'

    api_key = os.environ.get('DIOPTRA_API_KEY', None)
    if api_key is None:
        raise RuntimeError('DIOPTRA_API_KEY env var is not set')

    s3_bucket = os.environ.get('DIOPTRA_UPLOAD_BUCKET', None)

    if s3_bucket is None:
        raise RuntimeError('DIOPTRA_UPLOAD_BUCKET env var is not set')

    s3_prefix_bucket = os.environ.get('DIOPTRA_UPLOAD_PREFIX', '')
    storage_type = os.environ.get('DIOPTRA_UPLOAD_STORAGE_TYPE', 's3')

    my_dataset_metadata = []
    img_payload = []
    upload_ids = []

    for index, row in tqdm.tqdm(enumerate(dataset), desc='uploading dataset...'):

        if image_field is not None and image_field not in row:
            raise RuntimeError(f'{image_field} is not in the dataset')
        if groundtruth_field is not None and groundtruth_field not in row:
            raise RuntimeError(f'{groundtruth_field} is not in the dataset')

        my_metadata = None
        image_id = None
        if datapoints_metadata is not None:
            my_metadata = datapoints_metadata[index]
        if image_ids is not None:
            image_id = image_ids[index]

        if image_field is not None:
            pil_img = row[image_field]
            if not isinstance(pil_img, PIL.Image.Image):
                if getattr(pil_img, 'numpy', None) is not None: # dealing with Tensorflow Tensors
                    pil_img = PIL.Image.fromarray(pil_img.numpy())
                    pil_img.format = 'JPEG'
                    if image_id is not None:
                        _, ext = os.path.splitext(image_id)
                        if ext:
                            ext = ext.replace('.', '')
                            if ext.lower() == 'jpg':
                                pil_img.format = 'JPEG'
                            else:
                                pil_img.format = ext.upper()
                else:
                    raise Exception('the image provided was not a PIL image')
            img_width, img_height = pil_img.size

            if image_id is None:
                image_id = str(index) + '_' + str(uuid.uuid4())

        img_url = _build_img_url(storage_type, s3_bucket, s3_prefix_bucket, image_id, pil_img)

        in_mem_file = io.BytesIO()
        pil_img.save(in_mem_file, format=pil_img.format)

        datapoint_tags = {}
        image_metadata = {}

        if  my_metadata is not None and 'tags' in my_metadata:
            datapoint_tags.update(my_metadata['tags'])
        if  my_metadata is not None and 'image_metadata' in my_metadata:
            image_metadata.update(my_metadata['image_metadata'])

        if  dataset_metadata is not None and 'tags' in dataset_metadata:
            datapoint_tags.update(dataset_metadata['tags'])
        if  dataset_metadata is not None and 'image_metadata' in dataset_metadata:
            image_metadata.update(dataset_metadata['image_metadata'])

        image_metadata.update({
            'uri': img_url,
            'width': img_width,
            'height': img_height
        })

        img_payload.append((in_mem_file.getvalue(), img_url))
        my_dataset_metadata.append({
            **(my_metadata if my_metadata is not None else {}),
            **(dataset_metadata if dataset_metadata is not None else {}),
            'image_metadata': image_metadata,
            **({'tags': datapoint_tags} if len(datapoint_tags) > 0 else {}),
            **({
                'groundtruth': _format_groundtruth(row[groundtruth_field], dataset_type, class_names)
            } if groundtruth_field is not None else {}),
        })

        if len(img_payload) > max_batch_size:
            upload_ids.append(_upload_data(
                my_dataset_metadata, img_payload, num_workers))
            my_dataset_metadata = []
            img_payload = []

    if len(img_payload) > 0:
        upload_ids.append(_upload_data(
            my_dataset_metadata, img_payload, num_workers))

    return upload_ids

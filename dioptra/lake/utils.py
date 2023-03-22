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
import lz4
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from google.cloud import storage
import json
import tqdm
import orjson
import mgzip

# TODO: figure out how to return status codes form lambda functions.
# See the wip_return_error_code branch of the infrastructure repo.
def _raise_for_apigateway_errormessage(response):
    if response is not None and 'errorMessage' in response:
        raise RuntimeError(response['errorMessage'])

def query_dioptra_app(method, path, body=None):
    api_key = os.environ.get('DIOPTRA_API_KEY', None)
    if api_key is None:
        raise RuntimeError('DIOPTRA_API_KEY env var is not set')

    app_endpoint = os.environ.get('DIOPTRA_APP_ENDPOINT', 'https://app.dioptra.ai')

    r = getattr(requests, method.lower())(f'{app_endpoint}{path}', headers={
        'content-type': 'application/json',
        'x-api-key': api_key
    }, json=body)
    r.raise_for_status()

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

def get_predictions(datapoint_ids, filters=[]):
    """
    Get predictions for a set of datapoints

    Parameters:
        datapoint_ids: list of datapoint ids to get predictions for
    """

    fields = [
        'datapoints.id',
        'predictions.id',
        'predictions.datapoint',
        'predictions.task_type',
        'predictions.created_at',
        'predictions.class_name',
        'predictions.class_names',
        'predictions.confidence',
        'predictions.confidences',
        'predictions.model_name',
        'predictions.metrics',
        'predictions.encoded_segmentation_class_mask',
        'predictions.top',
        'predictions.left',
        'predictions.height',
        'predictions.width'
    ]
    datapoints = select_datapoints(
        filters=[{'left': 'datapoints.id', 'op': 'in', 'right': datapoint_ids}] + filters,
        fields=fields)

    return pd.DataFrame([p for predictions in datapoints['predictions'] for p in predictions])

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

def get_groundtruths(datapoint_ids, filters=[]):
    """
    Get groundtruths for a set of datapoints

    Parameters:
        datapoint_ids: list of datapoint ids to get groundtruths for
    """

    fields = [
        'datapoints.id',
        'groundtruths.id',
        'groundtruths.datapoint',
        'groundtruths.task_type',
        'groundtruths.created_at',
        'groundtruths.class_name',
        'groundtruths.class_names',
        'groundtruths.encoded_segmentation_class_mask',
        'groundtruths.top',
        'groundtruths.left',
        'groundtruths.height',
        'groundtruths.width'
    ]
    datapoints = select_datapoints(
        filters=[{'left': 'datapoints.id', 'op': 'in', 'right': datapoint_ids}] + filters,
        fields=fields)

    return pd.DataFrame([g for groundtruths in datapoints['groundtruths'] for g in groundtruths])

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

def upload_to_lake(records):
    """
    Uploading metadata to the data lake

    Parameters:
        records: array of dipotra style records. See teh doc for accepted formats
    """

    api_key = os.environ.get('DIOPTRA_API_KEY', None)
    if api_key is None:
        raise RuntimeError('DIOPTRA_API_KEY env var is not set')

    api_endpoint = os.environ.get('DIOPTRA_API_ENDPOINT', 'https://api.dioptra.ai/events')

    try:
        r = requests.post(api_endpoint, headers={
            'content-type': 'application/json',
            'x-api-key': api_key
        }, json={
            'records': records
        })
        r.raise_for_status()
        response = r.json()
        _raise_for_apigateway_errormessage(response)
    except requests.exceptions.RequestException as err:
        print('There was an error uploading to the lake ...')
        raise err

    return response

def upload_to_lake_via_object_store(records, custom_path='', disable_batching=False):
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

    payload = b'\n'.join(
        [orjson.dumps(record, option=orjson.OPT_SERIALIZE_NUMPY)for record in records])
    compressed_payload = mgzip.compress(payload, compresslevel=2)

    _upload_to_bucket((compressed_payload, store_url), no_compression=True)

    return upload_to_lake_from_bucket(object_store_bucket, file_name, storage_type, disable_batching)

def upload_to_lake_from_bucket(bucket_name, object_name, storage_type='s3', disable_batching=False):
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

    api_endpoint = os.environ.get('DIOPTRA_API_ENDPOINT', 'https://api.dioptra.ai/events')
    if storage_type == 's3':
        signed_url = _generate_s3_signed_url(bucket_name, object_name)
    elif storage_type == 'gs':
        signed_url = _generate_gs_signed_url(bucket_name, object_name)
    try:
        r = requests.post(api_endpoint, headers={
            'content-type': 'application/json',
            'x-api-key': api_key
        }, json={
            'url': signed_url
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

    app_endpoint = os.environ.get('DIOPTRA_APP_ENDPOINT', 'https://app.dioptra.ai')
    sleepTimeSecs = 1
    totalSleepTimeSecs = 0
    try:
        while True:
            if totalSleepTimeSecs > 900:
                raise RuntimeError('Timed out waiting for the upload to finish.')

            r = requests.get(f'{app_endpoint}/api/ingestion/executions/{upload_id}', headers={
                'content-type': 'application/json',
                'x-api-key': api_key
            })
            r.raise_for_status()
            upload = r.json()
            if upload['status'] in ['SUCCEEDED']:
                return upload
            elif upload['status'] in ['FAILED', 'TIMED_OUT', 'ABORTED']:
                raise RuntimeError(f'Upload failed with status {upload["status"]}. See more information in the Dioptra UI: {app_endpoint}/settings/uploads/{upload_id}')
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


def _encode_np_array(np_array):
    """
    Encode and compress a np array

    Parameters:
        np_array: the np array to be encoded

    """
    if not isinstance(np_array, np.ndarray):
        raise RuntimeError('Can only encode numpy arrays')

    bytes_buffer = io.BytesIO()
    np.save(bytes_buffer, np_array)

    return base64.b64encode(
        lz4.frame.compress(
            bytes_buffer.getvalue(),
            compression_level=lz4.frame.COMPRESSIONLEVEL_MAX
        )).decode('ascii')

def _decode_to_np_array(value):
    """
    Decode a compress a np array

    Parameters:
        value: the string containing the np array

    """
    decoded_bytes = lz4.frame.decompress(base64.b64decode(value))
    return np.load(io.BytesIO(decoded_bytes), allow_pickle=True)


def _format_groundtruth(groundtruth, gt_type, class_names=None):
    """
    Utility formatting the groundtruth field according to the model type

    Parameters:
        groundtruth: the groundtruth field
        gt_type: the type of gt. Supported types CLASSIFICATION, SEGMENTATION
        class_names: a list of class names. If the groundtruth contains indexes, it will be used to convert them to names

    """
    if getattr(groundtruth, 'numpy', None) is not None: # dealing with Tensorflow Tensors
        groundtruth = groundtruth.numpy()
    if gt_type == 'CLASSIFICATION':
        if (isinstance(groundtruth, int) or isinstance(groundtruth, np.integer)) and class_names is not None:
            class_name = class_names[groundtruth]
        else:
            class_name = groundtruth
        return {
            'task_type': gt_type,
            'class_name': class_name
        }
    if gt_type == 'SEGMENTATION':
        if isinstance(groundtruth, PIL.Image.Image):
            gt_array = np.array(groundtruth).tolist()
        if isinstance(groundtruth, np.ndarray):
            gt_array = groundtruth.tolist()
        if isinstance(groundtruth, list):
            gt_array = groundtruth
        return {
            'task_type': gt_type,
            'segmentation_class_mask': gt_array,
            **({'class_names': class_names} if class_names is not None else {})
        }

def _format_prediction(logits, pred_type, model_name, class_names=None, prediction_id=None):
    """
    Utility formatting the groundtruth field according to the model type

    Parameters:
        logits: the prediction logits (before softmax)
        pred_type: the type of gt. Supported types CLASSIFICATION, SEGMENTATION
        class_names: a list of class names. If the groundtruth contains indexes, it will be used to convert them to names
        prediction_id: the id of the prediction to be updated
    """
    if getattr(logits, 'numpy', None) is not None: # dealing with Tensorflow Tensors
        logits = logits.numpy()
    if pred_type in ['CLASSIFICATION', 'SEGMENTATION']:
        if isinstance(logits, np.ndarray):
            logits = logits.tolist()
        else:
            logits = logits
        return {
            'task_type': pred_type,
            'logits': logits,
            'model_name': model_name,
            **({'class_names': class_names} if class_names is not None else {}),
            **({'id': prediction_id} if prediction_id is not None else {})
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
    class_names=None, dataset_tags=None,
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

    dataset_metadata = []
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
        if dataset_tags is not None:
            datapoint_tags.update(dataset_tags)
        if  my_metadata is not None and 'image_metadata' in my_metadata:
            image_metadata.update(my_metadata['image_metadata'])

        image_metadata.update({
            'uri': img_url,
            'width': img_width,
            'height': img_height
        })

        img_payload.append((in_mem_file.getvalue(), img_url))
        dataset_metadata.append({
            'image_metadata': image_metadata,
            **({
                'groundtruth': _format_groundtruth(row[groundtruth_field], dataset_type, class_names)
            } if groundtruth_field is not None else {}),
            **(my_metadata if my_metadata is not None else {}),
            **({'tags': datapoint_tags} if len(datapoint_tags) > 0 else {})
        })

        if len(img_payload) > max_batch_size:
            upload_ids.append(_upload_data(
                dataset_metadata, img_payload, num_workers))
            dataset_metadata = []
            img_payload = []

    if len(img_payload) > 0:
        upload_ids.append(_upload_data(
            dataset_metadata, img_payload, num_workers))

    return upload_ids

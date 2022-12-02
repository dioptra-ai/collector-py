import os
import requests
import pandas as pd

import boto3
from botocore.exceptions import ClientError

def download_from_lake(filters, limit=None, order_by=None, desc=None):

    api_key = os.environ.get('DIOPTRA_API_KEY', None)
    if api_key is None:
        raise RuntimeError('DIOPTRA_API_KEY env var is not set')

    try:
        r = requests.post('https://app.dioptra.ai/api/metrics/select', headers={
            'content-type': 'application/json',
            'x-api-key': api_key
        }, json={
            'select': '*',
            'filters': filters,
            **({'limit': limit} if limit is not None else {}),
            **({'order_by': order_by} if order_by is not None else {}),
            **({'desc': desc} if desc is not None else {})
        })
        r.raise_for_status()
        return pd.json_normalize(r.json())
    except requests.exceptions.RequestException as err:
        print('There was an error querying the lake ...')
        raise err

def upload_to_lake(records):

    api_key = os.environ.get('DIOPTRA_API_KEY', None)
    if api_key is None:
        raise RuntimeError('DIOPTRA_API_KEY env var is not set')

    try:
        r = requests.post('https://api.dioptra.ai/events', headers={
            'content-type': 'application/json',
            'x-api-key': api_key
        }, json={
            'records': records
        })
        r.raise_for_status()
    except requests.exceptions.RequestException as err:
        print('There was an error uploading to the lake ...')
        raise err

    return r.json()

def upload_to_lake_from_s3(bucket_name, object_name):

    api_key = os.environ.get('DIOPTRA_API_KEY', None)
    if api_key is None:
        raise RuntimeError('DIOPTRA_API_KEY env var is not set')

    signed_url = _generate_s3_signed_url(bucket_name, object_name)

    try:
        r = requests.post('https://api.dioptra.ai/events', headers={
            'content-type': 'application/json',
            'x-api-key': api_key
        }, json={
            'urls': [signed_url]
        })
        r.raise_for_status()
    except requests.exceptions.RequestException as err:
        print('There was an error uploading to the lake ...')
        raise err

    return r.json()

def _generate_s3_signed_url(bucket_name, object_name):

    s3_client = boto3.client('s3')
    try:
        response = s3_client.generate_presigned_url(
            'get_object', Params={'Bucket': bucket_name, 'Key': object_name}, ExpiresIn=3600)
    except ClientError as err:
        print('There was an error getting a signed URL ...')
        raise err

    return response

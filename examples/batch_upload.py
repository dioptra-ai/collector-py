import os

from dioptra.api import Logger

API_KEY = os.environ.get('DIOPTRA_API_KEY')
DIOPTRA_BATCH_BUCKET = os.environ.get('DIOPTRA_BATCH_BUCKET')

def main():
    dioptra_logger = Logger(api_key=API_KEY)
    dioptra_logger.commit_ndjson_file(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'batch_upload_data.ndjson'
        ),
        batch_bucket=DIOPTRA_BATCH_BUCKET
    )

if __name__ == '__main__':
    main()

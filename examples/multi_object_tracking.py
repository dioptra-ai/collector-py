import os

from dioptra.api import Logger

API_KEY = os.environ.get('DIOPTRA_API_KEY')

def main():
    dioptra_logger = Logger(api_key=API_KEY)
    dioptra_logger.commit_ndjson_file(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'multi_object_tracking.ndjson'
        )
    )

if __name__ == '__main__':
    main()

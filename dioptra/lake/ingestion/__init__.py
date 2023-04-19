from . import compatibility

def process_records(records):
    for record in records:
        if record.get('predictions'):
            for i, prediction in enumerate(record['predictions']):
                record['predictions'][i] = compatibility.process_prediction(prediction, record)
        
        if record.get('groundtruths'):
            for i, groundtruth in enumerate(record['groundtruths']):
                record['groundtruths'][i] = compatibility.process_groundtruth(groundtruth, record)

    return records
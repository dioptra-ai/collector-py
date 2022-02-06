import datetime

from schema import Schema, Or, Optional

def object_detection_schema():
    return Schema([{
        'class_name': str,
        'top': Or(int, float),
        'left': Or(int, float),
        'height': Or(int, float),
        'width': Or(int, float),
        Optional('confidence'): float,
        Optional('text'): str,
        Optional('text_confidence'): str
    }])

def automated_speech_recogniton_schema():
    return Schema({
        'text': str
    })

def question_answering_groundtruth_schema():
    return Schema({
        'text': str,
        Optional('embeddings'): [float]
    })

def question_answering_prediction_schema():
    return Schema([{
        'text': str,
        Optional('embeddings'): [float],
        Optional('confidence'): float
    }])

def image_metadata_schema():
    return Schema({
        'uri': str,
        Optional('rotation'): Or(int, float),
        Optional('height'): Or(int, float),
        Optional('width'): Or(int, float)
    })

def text_metadata_schema():
    return Schema({
        Optional('uri'): str,
        Optional('num_char'): Or(int, float),
        Optional('num_punct'): Or(int, float),
        Optional('num_digits'): Or(int, float),
        Optional('num_tokens'): Or(int, float)
    })

def audio_metadata_schema():
    return Schema({
        Optional('uri'): str,
        Optional('duration'): Or(int, float),
        Optional('sampling_rate'): Or(int, float),
        Optional('speaker_id'): Or(str, int),
        Optional('start_time'): Or(int, float),
        Optional('end_time'): Or(int, float)
    })

def feature_schema():
    return Schema({str: Or(str, int, float, bool)})

def tag_schema():
    return Schema({str: Or(str, int, float, bool)})

def embeddings_schema():
    return Schema([float])

import datetime

from schema import Schema, Or, Optional


def auto_completion_prediction_schema():
    return Schema([{
        'text': str,
        Optional('displayed'): bool,
        Optional('confidence'): float,
        Optional('embeddings'): [float]
    }])

def auto_completion_groundtruth_schema():
    return Schema({
        'text': str,
        Optional('embeddings'): [float]
    })

def object_detection_schema():
    return Schema(
        Or([{
            'top': Or(int, float),
            'left': Or(int, float),
            'height': Or(int, float),
            'width': Or(int, float),
            'class_name': Or(str, [str]),
            Optional('logits'): [Or(int, float)],
            Optional('confidence'): Or(float, [float]),
            Optional('objectness'): float,
            Optional('embeddings'): Or(
                [Or(float, int)],
                [[Or(float, int)]],
                [[[Or(float, int)]]]
            )
        }],
        {
            'boxes': [[float]],
            Optional('confidences'): Or([float], [[float]]),
            Optional('logits'): [[Or(int, float)]],
            Optional('objectness'): [float],
            'class_names': [str],
            Optional('embeddings'): [Or(
                [Or(float, int)],
                [[Or(float, int)]],
                [[[Or(float, int)]]]
            )]
        }))


def classification_schema():
    return Schema({
        'class_name': [str],
        'confidence': [Or(int, float)],
        Optional('logits'): [Or(int, float)]
    })

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

def semantic_similarity_schema():
    return Schema(Or(int, float))

def semantic_similarity_input_schema():
    return Schema({
        Optional('text_1'): str,
        'embeddings_1': [float],
        Optional('text_2'): str,
        'embeddings_2': [float]
    })

def image_metadata_schema():
    return Schema({
        'uri': str,
        Optional('rotation'): Or(int, float),
        Optional('height'): Or(int, float),
        Optional('width'): Or(int, float),
        Optional('object'): Schema({
            'top': Or(int, float),
            'left': Or(int, float),
            'height': Or(int, float),
            'width': Or(int, float)
        })
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

def video_metadata_schema():
    return Schema({
        Optional('uri'): str,
        Optional('duration'): Or(int, float),
        Optional('height'): Or(int, float),
        Optional('width'): Or(int, float),
        Optional('frame_rate'): Or(int, float),
        Optional('frame'): int,
    })

def multiple_object_tracking_schema():
    return Schema([{
        Optional('class_name'): str,
        'target_id': Or(str, int),
        'top': Or(int, float),
        'left': Or(int, float),
        'height': Or(int, float),
        'width': Or(int, float),
        Optional('confidence'): float
    }])

def feature_schema():
    return Schema({str: Or(str, int, float, bool)})

def tag_schema():
    return Schema({str: Or(str, int, float, bool)})

def embeddings_schema():
    return Schema(
        Or(
            [Or(float, int)],
            [[Or(float, int)]],
            [[[Or(float, int)]]]
        ))

def logits_schema():
    return Schema([Or(float, int)])

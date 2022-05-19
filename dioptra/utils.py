"""
Utility module

"""

import datetime

from dioptra.schemas import (
    object_detection_schema,
    multiple_object_tracking_schema,
    feature_schema,
    tag_schema,
    embeddings_schema,
    logits_schema,
    image_metadata_schema,
    text_metadata_schema,
    audio_metadata_schema,
    video_metadata_schema,
    question_answering_groundtruth_schema,
    question_answering_prediction_schema,
    automated_speech_recogniton_schema,
    auto_completion_groundtruth_schema,
    auto_completion_prediction_schema,
    semantic_similarity_schema,
    semantic_similarity_input_schema,
    classification_schema
)

from dioptra.supported_types import ModelTypes, InputTypes

from dioptra.supported_types import SupportedTypes

def validate_model_type(model_type):
    return isinstance(model_type, SupportedTypes)

def validate_timestamp(timestamp):
    return isinstance(timestamp, datetime.datetime)

def validate_tags(tags):
    return tag_schema().is_valid(tags)

def validate_text(text):
    return isinstance(text, str)

def validate_input_data(input_data, model_type):
    if model_type.input_type == InputTypes.PAIRED_TEXT:
        return semantic_similarity_input_schema().is_valid(input_data)
    return False

def validate_annotations(annotations, model_type):

    if model_type.model_type == ModelTypes.CLASSIFIER:
        if isinstance(annotations, str):
            return True
        elif classification_schema().is_valid(annotations):
            return True
        else:
            return False
    elif model_type.model_type == ModelTypes.OBJECT_DETECTION:
        return object_detection_schema().is_valid(annotations)
    elif model_type.model_type == ModelTypes.MULTIPLE_OBJECT_TRACKING:
        return multiple_object_tracking_schema().is_valid(annotations)
    elif model_type.model_type == ModelTypes.QUESTION_ANSWERING:
        if question_answering_groundtruth_schema().is_valid(annotations):
            return True
        elif question_answering_prediction_schema().is_valid(annotations):
            return True
        return False
    elif model_type.model_type == ModelTypes.AUTOMATED_SPEECH_RECOGNITION:
        return automated_speech_recogniton_schema().is_valid(annotations)
    elif model_type.model_type == ModelTypes.AUTO_COMPLETION:
        if auto_completion_groundtruth_schema().is_valid(annotations):
            return True
        elif auto_completion_prediction_schema().is_valid(annotations):
            return True
        return False
    elif model_type.model_type == ModelTypes.SEMANTIC_SIMILARITY:
        return semantic_similarity_schema().is_valid(annotations)
    else:
        return False

def validate_confidence(confidence):
    return isinstance(confidence, float)

def validate_features(features):
    return feature_schema().is_valid(features)

def validate_embeddings(embeddings):
    return embeddings_schema().is_valid(embeddings)

def validate_logits(embeddings):
    return embeddings_schema().is_valid(embeddings)

def validate_image_metadata(image_metadata):
    return image_metadata_schema().is_valid(image_metadata)

def validate_text_metadata(text_metadata):
    return text_metadata_schema().is_valid(text_metadata)

def validate_audio_metadata(audio_metadata):
    return audio_metadata_schema().is_valid(audio_metadata)

def validate_video_metadata(video_metadata):
    return video_metadata_schema().is_valid(video_metadata)

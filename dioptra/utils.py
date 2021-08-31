"""
Utility module

"""

import numbers

def validate_timestamp(timestamp):
    return timestamp is not None

def validate_tags(tags):
    return isinstance(tags, dict)

def validate_prediction(prediction):
    return isinstance(prediction, str)

def validate_confidence(confidence):
    return isinstance(confidence, numbers.Number)

def validate_features(features):
    return isinstance(features, dict)

def validate_embeddings(embeddings):
    return isinstance(embeddings, list)

def validate_image_url(image_url):
    return isinstance(image_url, str)

def validate_groundtruth(groundtruth):
    return isinstance(groundtruth, str)

def add_prefix_to_keys(dictionary, prefix):
    return_dict = {}
    for key in dictionary:
        return_dict[prefix + '.' + key] = dictionary[key]

    return return_dict

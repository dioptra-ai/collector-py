"""
Utility module

"""

def validate_timestamp(timestamp):
    return True

def validate_tags(tags):
    return True

def validate_prediction(prediction):
    return True

def validate_confidence(confidence):
    return True

def validate_features(features):
    return True

def validate_embeddings(embeddings):
    return True

def validate_image_url(image_url):
    return True

def validate_groundtruth(groundtruth):
    return True

def add_prefix_to_keys(dictionary, prefix):
    return_dict = {}
    for key in dictionary:
        return_dict[prefix + '.' + key] = dictionary[key]

    return return_dict

import os
import json
import random

def _load_config():
    """
    Loading the config from file

    """
    with open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                'image_classifier_config.json')
            , 'r') as conf_file:
        conf_json = conf_file.read()
        return json.loads(conf_json)

CONFIG = _load_config()

def generate_prediction(image_index):
    """
    Generate a fake prediction for an image, including confidence.
    We simulate model errors

    """

    # Generating random confidence
    mean_prediction_conf = CONFIG['predictions']['conf']['mean']
    stdev_prediction_conf = CONFIG['predictions']['conf']['stdev']
    confidence = random.gauss(mean_prediction_conf, stdev_prediction_conf)

    if confidence < 0:
        confidence = 0.0
    if confidence > 1:
        confidence = 1.0

    image_class = CONFIG['images'][image_index]['class']

    prediction_swap_prob = CONFIG['predictions']['swap_prob']
    random_sampler = random.random()

    # Simulating model errors and a model that has never seen cheques before
    swap = False
    if image_class == 'cheques' or random_sampler < prediction_swap_prob:
        swap = True

    if swap:
        prediction_classes = CONFIG['predictions']['class']
        new_class_index = int(random.random() * (len(prediction_classes)))
        image_class = prediction_classes[new_class_index]

    return image_class, confidence

def generate_groundtruth(image_index):
    """
    Gather the groundtruth label for an image

    """

    return CONFIG['images'][image_index]['class']

def generate_tags(image_index, cheques_scenario):
    """
    Generating a client id tag
    We similate that only client 3 will send us cheques and only when scenario is 'cheques'

    """

    image_class = CONFIG['images'][image_index]['class']

    tags = CONFIG['tags']
    tags_class_name = tags['name']
    tags_classes = tags['values']

    random_sampler = random.random()

    count = 0.0
    tags_class = None

    for my_tags_class in tags_classes:
        tags_name = my_tags_class['name']
        tags_prob = my_tags_class['prob']
        count += tags_prob
        if random_sampler < count:
            tags_class = tags_name
            break

    if cheques_scenario and image_class =='cheques':
        tags_class = 'my_client_3'


    return {
            tags_class_name: tags_class
        }

def generate_feature(image_index):
    """
    Generate a fake feature diction with one feature: rotation

    """
    rotation = 0
    random_sampler = random.random()
    if random_sampler > 0.75:
        rotation = 90
    if random_sampler > 0.85:
        rotation = 180
    if random_sampler > 0.9:
        rotation = 270

    return {'rotation': rotation}

def generate_embeddings(image_index):
    """
    Gather the embeddings for a given image

    """

    return CONFIG['images'][image_index]['embeddings']

def generate_image_url(image_index):
    """
    Gather the image url

    """

    return CONFIG['images'][image_index]['url']

def get_random_image_index(cheques_scenario):

    image_index = int(random.random() * (len(CONFIG['images'])))
    image_class = CONFIG['images'][image_index]['class']

    while not cheques_scenario and image_class == 'cheques':
        image_index = int(random.random() * (len(CONFIG['images'])))
        image_class = CONFIG['images'][image_index]['class']

    return image_index

def generate_datapoint(image_index, cheques_scenario):

    model_prediction, model_confidence = generate_prediction(image_index)

    image_url = generate_image_url(image_index)
    image_embeddings = generate_embeddings(image_index)
    image_features = generate_feature(image_index)

    request_tags = generate_tags(image_index, cheques_scenario)

    return {
        'request_tags': request_tags,
        'model_prediction': model_prediction,
        'model_confidence': model_confidence,
        'image_url': image_url,
        'image_embeddings': image_embeddings,
        'image_features': image_features,
    }

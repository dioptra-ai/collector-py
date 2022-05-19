from enum import Enum


class ModelTypes(Enum):
    CLASSIFIER = 'CLASSIFIER'
    OBJECT_DETECTION = 'OBJECT_DETECTION'
    QUESTION_ANSWERING = 'Q_N_A'
    AUTOMATED_SPEECH_RECOGNITION = 'ASR'
    AUTO_COMPLETION = 'AUTO_COMPLETION'
    SEMANTIC_SIMILARITY = 'SEMANTIC_SIMILARITY'
    MULTIPLE_OBJECT_TRACKING = 'MULTIPLE_OBJECT_TRACKING'

class InputTypes(Enum):
    TABULAR = 'TABULAR'
    IMAGE = 'IMAGE'
    TEXT = 'TEXT'
    PAIRED_TEXT = 'PAIRED_TEXT'
    AUDIO = 'AUDIO'
    VIDEO = 'VIDEO'

class SupportedTypes(Enum):
    TABULAR_CLASSIFIER = 'TABULAR_CLASSIFIER', ModelTypes.CLASSIFIER, InputTypes.TABULAR
    IMAGE_CLASSIFIER = 'IMAGE_CLASSIFIER', ModelTypes.CLASSIFIER, InputTypes.IMAGE
    TEXT_CLASSIFIER = 'TEXT_CLASSIFIER', ModelTypes.CLASSIFIER, InputTypes.TEXT
    OBJECT_DETECTION = 'OBJECT_DETECTION', ModelTypes.OBJECT_DETECTION, InputTypes.IMAGE
    QUESTION_ANSWERING = 'QUESTION_ANSWERING', ModelTypes.QUESTION_ANSWERING, InputTypes.TEXT
    AUTOMATED_SPEECH_RECOGNITION = 'AUTOMATED_SPEECH_RECOGNITION', \
        ModelTypes.AUTOMATED_SPEECH_RECOGNITION, InputTypes.AUDIO
    AUTO_COMPLETION = 'AUTO_COMPLETION', ModelTypes.AUTO_COMPLETION, InputTypes.TEXT
    SEMANTIC_SIMILARITY = 'SEMANTIC_SIMILARITY', \
        ModelTypes.SEMANTIC_SIMILARITY, InputTypes.PAIRED_TEXT
    MULTIPLE_OBJECT_TRACKING = 'MULTIPLE_OBJECT_TRACKING', ModelTypes.MULTIPLE_OBJECT_TRACKING, InputTypes.VIDEO

    def __new__(cls, *args, **kwds):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _, model_type, input_type):
        self._model_type = model_type
        self._input_type = input_type

    def __str__(self):
        return self.value

    @property
    def model_type(self):
        return self._model_type
    @property
    def input_type(self):
        return self._input_type

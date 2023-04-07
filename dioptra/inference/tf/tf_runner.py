import re
from tqdm import tqdm
import tensorflow as tf

from dioptra.inference.inference_runner import InferenceRunner
from dioptra.lake.utils import _format_prediction

class TfInferenceRunner(InferenceRunner):
    def __init__(
            self, model, model_type,
            model_name = None,
            embeddings_layers=[],
            logits_layer=None, class_names=[],
            metadata=None):
        """
        Utility to perform model inference on a dataset and extract layers needed for AL.

        Parameters:
            model: model to be used to inference
            model_type: the type of the model use. Can be CLASSIFICATION or SEGMENTATION
            embeddings_layers: an array of layer names that should be used as embeddings. Can be a jq style path to an embedding layer like [0].my_embedding
            logits_layer: the name of the logit layer (pre softmax) to be used for AL. Can be a jq style path to an embedding layer like [0].my_logits
            class_names: the class names corresponding to each logit. Indexes should match the logit layer
            metadata: a list of dioptra style metadata to be added to teh datapoints. The indexes in this list should match the indexes in the dataset
        """

        super().__init__()

        self.model = model
        self.model_name = model_name
        self.embeddings_layers = embeddings_layers
        self.logits_layer = logits_layer
        self.class_names = class_names
        self.metadata = metadata
        self.model_type = model_type

        input_layer = model.inputs
        if input_layer is None:
            input_layer = model.layers[0].inputs

        output_layers = {}

        for layer in embeddings_layers + [logits_layer]:
            if layer is not None:
                output_layers[layer] = self._get_layer_by_name(layer).output

        self.logging_model = tf.keras.Model(
            inputs=input_layer,
            outputs=output_layers)

    def _get_layer_by_name(self, name):
        split = name.split('.')
        current_layer = self.model
        for part in split:
            if re.match('\[[0-9]+\]', part):
                index = int(part.replace('[', '').replace(']', ''))
                current_layer = current_layer.layers[index]
            else:
                current_layer = current_layer.get_layer(part)
        return current_layer


    def run(self, dataset):
        """
        Run the inference on a dataset and upload results to dioptra

        Parameters:
            dataset: a tf.data.Dataset
                Should be batched and pre processed to only return the data, not the groundtruth
                Should not be shuffled if used with a metadata list
        """

        records = []

        global_idx = 0

        for batch_index, batch in tqdm(enumerate(dataset), desc='running inference...'):
            output = self.logging_model(batch)
            for batch_idx in range(batch.shape[0]):
                records.extend(self._build_records(batch_idx, global_idx, output))
                global_idx += 1
                if len(records) > self.max_batch_size:
                    self._ingest_data(records)
                    records = []
        if len(records) > 0:
            self._ingest_data(records)
            records = []

    def _build_records(self, record_batch_idx, record_global_idx, output):
        logits = None
        if self.logits_layer is not None:
            logits = output[self.logits_layer][record_batch_idx].numpy()
        
        embeddings = {}
        for my_layer in self.embeddings_layers:
            embeddings[my_layer] = output[my_layer][record_batch_idx].numpy()

        return [{
            **({
                'prediction': _format_prediction(
                    logits=logits,
                    embeddings=embeddings,
                    task_type=self.model_type,
                    model_name=self.model_name,
                    class_names=self.class_names
                )
               } if logits or embeddings else {}
            ),
            **(self.metadata[record_global_idx] \
               if self.metadata and len(self.metadata) > record_global_idx else {}
            )
        }]

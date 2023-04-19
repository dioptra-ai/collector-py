import re
from tqdm import tqdm
import tensorflow as tf

from dioptra.inference.inference_runner import InferenceRunner
from dioptra.lake.utils import _format_prediction

class TfInferenceRunner(InferenceRunner):
    def __init__(
            self, model, model_type,
            model_name = None,
            datapoint_ids = [],
            embeddings_layers=[],
            logits_layer=None,
            class_names=[],
            datapoints_metadata=None,
            dataset_metadata=None,
            mc_dropout_samples=0,
            channel_last=False,):
        """
        Utility to perform model inference on a dataset and extract layers needed for AL.

        Parameters:
            model: model to be used to inference
            model_name: the name of the model
            model_type: the type of the model use. Can be CLASSIFICATION or SEGMENTATION
            datapoint_ids: alist of datapoints to update with the predictions. Should be in the same order as the dataset.
            embeddings_layers: an array of layer names that should be used as embeddings
            logits_layer: the name of the logit layer (pre softmax) to be used for AL
            class_names: the class names corresponding to each logit. Indexes should match the logit layer
            datapoints_metadata: a list of dioptra style datapoints metadata to be added to teh datapoints. The indexes in this list should match the indexes in the dataset
            dataset_metadata: a dioptra style dataset metadata to be added to the dataset
            channel_last: if the model expects the data to be in channel last format
        """

        super().__init__()

        self.model = model
        self.model_name = model_name
        self.datapoint_ids = datapoint_ids
        self.embeddings_layers = embeddings_layers
        self.logits_layer = logits_layer
        self.class_names = class_names
        self.datapoints_metadata = datapoints_metadata
        self.dataset_metadata = dataset_metadata
        self.model_type = model_type
        self.mc_dropout_samples = mc_dropout_samples
        self.channel_last = channel_last

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
            if re.match('\[[0-9-]+\]', part):
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
        nb_samples = 1 if self.mc_dropout_samples == 0 else self.mc_dropout_samples

        for _, batch in tqdm(enumerate(dataset), desc='running inference...'):
            samples_records = []
            for _ in range(nb_samples):
                batch_global_idx = global_idx
                batch_records = []
                output = self.logging_model(batch, training=self.mc_dropout_samples > 0)
                for batch_idx in range(batch.shape[0]):
                    batch_records.extend(self._build_records(batch_idx, batch_global_idx, output))
                    batch_global_idx += 1
                samples_records.append(batch_records)

            resolved_records = self._resolve_records(samples_records)
            records.extend(resolved_records)
            global_idx += len(batch)

            if len(records) > 0:
                self._ingest_data(records)
                records = []

        if len(records) > 0:
            self._ingest_data(records)
            records = []

    def _build_records(self, record_batch_idx, record_global_idx, output):

        datapoint_id = None
        if record_global_idx < len(self.datapoint_ids):
            datapoint_id = self.datapoint_ids[record_global_idx]

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
                    class_names=self.class_names,
                    channel_last=self.channel_last
                )
               } if logits is not None or embeddings is not None else {}
            ),
            **({'id': datapoint_id} if datapoint_id is not None else {}),
            **(self.datapoints_metadata[record_global_idx] \
               if self.datapoints_metadata and len(self.datapoints_metadata) > record_global_idx else {}
            ),
            **(self.dataset_metadata if self.dataset_metadata else {})
        }]

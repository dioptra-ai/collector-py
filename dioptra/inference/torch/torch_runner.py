import re
from tqdm import tqdm
import torch

from dioptra.inference.inference_runner import InferenceRunner
from dioptra.lake.utils import _format_prediction

class TorchInferenceRunner(InferenceRunner):
    def __init__(
            self, model, model_type, model_name = None, predictions_to_update = None,
            embeddings_layers=[],
            logits_layer=None, class_names=[],
            metadata=None,
            data_transform=None,
            device='cpu'):
        """
        Utility to perform model inference on a dataset and extract layers needed for AL.

        Parameters:
            model: model to be used to inference
            model_name: the name of the model
            model_type: the type of the model use. Can be CLASSIFIER or SEMANTIC_SEGMENTATION
            predictions_to_update: the predictions to be updated. If None, all predictions are expected to be new
            embeddings_layers: an array of layer names that should be used as embeddings
            logits_layer: the name of the logit layer (pre softmax) to be used for AL
            class_names: the class names corresponding to each logit. Indexes should match the logit layer
            metadata: a list of dioptra style metadata to be added to teh datapoints. The indexes in this list should match the indexes in the dataset
            data_transform: a transform function that will be called before the model is called. Should only return the data, without the groundtruth
            device: the devide to be use to perform the inference
        """

        super().__init__()

        self.model = model
        self.model_name = model_name
        self.predictions_to_update = predictions_to_update
        self.embeddings_layers = embeddings_layers
        self.logits_layer = logits_layer
        self.class_names = class_names
        self.metadata = metadata
        self.data_transform = data_transform
        self.device = device
        self.model_type = model_type

        self.activation = {}

        for my_layer_name in embeddings_layers + [logits_layer]:
            if my_layer_name is not None:
                my_layer = self._get_layer_by_name(my_layer_name)
                my_layer.register_forward_hook(self._get_activation(my_layer_name))

    def _get_layer_by_name(self, name):
        split = name.split('.')
        current_layer = self.model
        for part in split:
            if re.match('\[[0-9]+\]', part):
                index = int(part.replace('[', '').replace(']', ''))
                current_layer = current_layer[index]
            else:
                current_layer = getattr(current_layer, part)
        return current_layer

    def _get_activation(self, name):
        def hook(model, input, output):
            if hasattr(output, 'last_hidden_state'):
                self.activation[name] = output.last_hidden_state
            else:
                self.activation[name] = output.detach()
        return hook

    def run(self, dataloader):
        """
        Run the inference on a dataset and upload results to dioptra

        Parameters:
            dataset: a torch.utils.data.Dataset
                Should be batched. data_transform can be used to pre process teh data to only return the data, not the groundtruth
                Should not be shuffled if used with a metadata list
        """

        self.model.eval()
        self.model.to(self.device)

        records = []

        global_idx = 0
        if hasattr(dataloader, 'dataset'):
            dataset_size = len(dataloader.dataset) # we are using a dataloader
        else:
            dataset_size = len(dataloader)  # we are using a dataset directly
        for batch_index, batch in tqdm(enumerate(dataloader), desc='running inference...'):
            if self.data_transform:
                batch = self.data_transform(batch)
            batch = batch.to(self.device)
            with torch.no_grad():
                self.model(batch)
            for batch_idx, _ in enumerate(batch):
                records.extend(self._build_records(batch_idx, global_idx))
                global_idx += 1
                if len(records) > self.max_batch_size:
                    self._ingest_data(records)
                    records = []
                if global_idx > dataset_size:
                    break
            if global_idx > dataset_size:
                break
        if len(records) > 0:
            self._ingest_data(records)
            records = []

    def _build_records(self, record_batch_idx, record_global_idx):
        # find the prediction id that corresponds to the datapoint id in the metadata
        prediction_id = None
        if self.predictions_to_update is not None:
            # old_predictions is a dataframe with the following columns:
            #   - id: the id of the prediction
            #   - datapoint: the datapoint id for the prediction

            datapoint_id = self.metadata[record_global_idx]['id']
            prediction_id = self.predictions_to_update[self.predictions_to_update['datapoint'] == datapoint_id]['id'].values[0]
        my_record = {
            **({
                'prediction': [_format_prediction(
                                self.activation[self.logits_layer][record_batch_idx].cpu().numpy(),
                                self.model_type,
                                self.model_name,
                                self.class_names,
                                prediction_id)]
               } if self.logits_layer in self.activation  and self.class_names else {}
            ),
            **(self.metadata[record_global_idx] \
               if self.metadata and len(self.metadata) > record_global_idx else {}
            ),
        }
        my_records = []
        
        for my_layer in self.embeddings_layers:
            if my_layer not in self.activation:
                continue
            record_tags = my_record.get('tags', {})
            if 'embeddings_name' not in record_tags:
                record_tags['embeddings_name'] = my_layer
                my_record['tags'] = record_tags
            layer_record = {
                'embeddings': self.activation[my_layer][record_batch_idx].cpu().numpy(),
                **my_record
            }
            my_records.append(layer_record)

        if len(my_records) == 0:
            my_records.append(my_record)
        return my_records
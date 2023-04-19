import re
from tqdm import tqdm
import torch

from dioptra.inference.inference_runner import InferenceRunner
from dioptra.lake.utils import _format_prediction

class TorchInferenceRunner(InferenceRunner):
    def __init__(
            self, model, model_type,
            model_name = None,
            datapoint_ids = [],
            embeddings_layers=[],
            logits_layer=None,
            class_names=[],
            datapoints_metadata=None,
            dataset_metadata=None,
            data_transform=None,
            grad_embeddings_transform=None,
            logits_transform=None,
            mc_dropout_samples=0,
            channel_last=False,
            device='cpu'):
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
            data_transform: a transform function that will be called before the model is called. Should only return the data, without the groundtruth
            device: the devide to be use to perform the inference
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
        self.data_transform = data_transform
        self.logits_transform = logits_transform
        self.grad_embeddings_transform = grad_embeddings_transform
        self.device = device
        self.model_type = model_type
        self.mc_dropout_samples = mc_dropout_samples
        self.channel_last = channel_last

        self.activation = {}

        for my_layer_name in embeddings_layers + [logits_layer]:
            if my_layer_name is not None:
                my_layer = self._get_layer_by_name(my_layer_name)
                my_layer.register_forward_hook(self._get_activation(my_layer_name))

    def _get_layer_by_name(self, name):
        split = name.split('.')
        current_layer = self.model
        for part in split:
            if re.match('\[[0-9-]+\]', part):
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
                Should not be shuffled if used with a datapoints_metadata list
        """

        if self.mc_dropout_samples == 0:
            self.model.eval()
        else:
            self.model.train()

        self.model.to(self.device)

        records = []

        global_idx = 0
        nb_samples = 1 if self.mc_dropout_samples == 0 else self.mc_dropout_samples

        if hasattr(dataloader, 'dataset'):
            dataset_size = len(dataloader.dataset) # we are using a dataloader
        else:
            dataset_size = len(dataloader)  # we are using a dataset directly
        for _, batch in tqdm(enumerate(dataloader), desc='running inference...'):
            if self.data_transform:
                batch = self.data_transform(batch)
            batch = batch.to(self.device)
            samples_records = []

            for _ in range(nb_samples):
                batch_global_idx = global_idx
                batch_records = []
                with torch.no_grad():
                    self.model(batch)
                for batch_idx, _ in enumerate(batch):
                    batch_records.extend(self._build_records(batch_idx, global_idx))
                    batch_global_idx += 1
                samples_records.append(batch_records)

            resolved_records = self._resolve_records(samples_records)
            records.extend(resolved_records)

            global_idx += len(batch)

            if len(records) > self.max_batch_size:
                self._ingest_data(records)
                records = []
            if global_idx > dataset_size:
                break
        if len(records) > 0:
            self._ingest_data(records)
            records = []

    def _build_records(self, record_batch_idx, record_global_idx):
        datapoint_id = None
        if record_global_idx < len(self.datapoint_ids):
            datapoint_id = self.datapoint_ids[record_global_idx]

        logits = None
        if self.logits_layer in self.activation:
            logits = self.activation[self.logits_layer][record_batch_idx]

        transformed_logits = None
        if logits is not None and self.logits_transform is not None:
            transformed_logits = self.logits_transform(logits)

        grad_embeddings = None
        if transformed_logits is not None and self.grad_embeddings_transform is not None:
            grad_embeddings = self.grad_embeddings_transform(transformed_logits)

        embeddings = {}
        for my_layer in self.embeddings_layers:
            if my_layer not in self.activation:
                continue
            embeddings[my_layer] = self.activation[my_layer][record_batch_idx]

        return [{
            **({
                'prediction': _format_prediction(
                    logits=logits,
                    transformed_logits=transformed_logits,
                    grad_embeddings=grad_embeddings,
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

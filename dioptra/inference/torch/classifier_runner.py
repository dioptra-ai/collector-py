from tqdm import tqdm
import torch

from dioptra.inference.inference_runner import InferenceRunner

class ClassifierRunner(InferenceRunner):
    def __init__(
            self, model, embeddings_layers,
            logits_layer, class_names,
            metadata=None,
            data_transform=None,
            device='cpu'):
        """
        Utility to perform model inference on a dataset and extract layers needed for AL.

        Parameters:
            model: model to be used to inference
            embeddings_layers: an array of layer names that should be used as embeddings
            logits_layer: the name of the logit layer (pre softmax) to be used for AL
            class_names: the class names corresponding to each logit. Indexes should match the logit layer
            metadata: a list of dioptra style metadata to be added to teh datapoints. The indexes in this list should match the indexes in the dataset
            data_transform: a transform function that will be called before the model is called. Should only return the data, without the groundtruth
            device: the devide to be use to perform the inference
        """

        super().__init__()

        self.model = model
        self.embeddings_layers = embeddings_layers
        self.logits_layer = logits_layer
        self.class_names = class_names
        self.metadata = metadata
        self.data_transform = data_transform
        self.device = device

        self.activation = {}

        for my_layer_name in embeddings_layers + [logits_layer]:
            my_layer = getattr(self.model, my_layer_name)
            my_layer.register_forward_hook(self._get_activation(my_layer_name))

    def _get_activation(self, name):
        def hook(model, input, output):
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
                if global_idx > len(dataloader):
                    break
            if global_idx > len(dataloader):
                break
        if len(records) > 0:
            self._ingest_data(records)
            records = []

    def _build_records(self, record_batch_idx, record_global_idx):

        my_record = {
            **({
                'prediction': {
                    'logits': self.activation[self.logits_layer][record_batch_idx].cpu().numpy(),
                    'class_names': self.class_names
                }
               } if self.logits_layer in self.activation  and self.class_names else {}
            ),
            'model_type': 'CLASSIFIER',
            **(self.metadata[record_global_idx] \
               if self.metadata and len(self.metadata) > record_global_idx else {}
            )
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

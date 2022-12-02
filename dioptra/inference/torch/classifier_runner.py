from tqdm import tqdm
import torch

from dioptra.inference.inference_runner import InferenceRunner

class ClassifierRunner(InferenceRunner):
    def __init__(
            self, model, embeddings_layers,
            logits_layer, class_names,
            metadata=None,
            dataloader_args=None,
            data_transform=None,
            device='cpu'):

        super().__init__()

        self.model = model
        self.embeddings_layers = embeddings_layers
        self.logits_layer = logits_layer
        self.class_names = class_names
        self.metadata = metadata
        self.dataloader_args = dataloader_args
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
                    self._dump_data(records)
                    records = []
                if global_idx > len(dataloader):
                    break
            if global_idx > len(dataloader):
                break
        if len(records) > 0:
            self._dump_data(records)
            records = []

    def _build_records(self, record_batch_idx, record_global_idx):

        my_record = {
            **(self.metadata[record_global_idx] \
               if self.metadata and len(self.metadata) > record_global_idx else {}
            ),
            **({
                'prediction': {
                    'logits': self.activation[self.logits_layer][record_batch_idx].cpu().numpy(),
                    'class_names': self.class_names
                }
               } if self.logits_layer in self.activation  and self.class_names else {}
            ),
            'model_type': 'CLASSIFIER'
        }

        my_records = []

        for my_layer in self.embeddings_layers:
            if my_layer not in self.activation:
                continue
            layer_record = {
                **my_record,
                'embeddings': self.activation[my_layer][record_batch_idx].cpu().numpy()
            }
            record_tags = layer_record.get('tags', {})
            record_tags['embeddings_layer'] = my_layer
            layer_record['tags'] = record_tags
            my_records.append(layer_record)

        if len(my_records) == 0:
            my_records.append(my_record)

        return my_records

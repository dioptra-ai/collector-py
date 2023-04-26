
from functools import partial
import torch
import torchvision
import pytorch_lightning as pl

from dioptra.lake.utils import select_datapoints
from dioptra.lake.torch.object_store_datasets import ImageDataset
from dioptra.ssl.fast_siam import FastSiam, fast_siam_transform, get_coalate_method

my_df = select_datapoints([{'left': 'tags.value', 'op': '=', 'right': 'cu_lane'}])
my_dataset = ImageDataset(my_df, transform=partial(fast_siam_transform, input_size=64))
my_dataset.use_caching = True

dataloader = torch.utils.data.DataLoader(
    my_dataset,
    batch_size=512,
    collate_fn=get_coalate_method(),
    shuffle=True,
    drop_last=True,
    num_workers=16,
)

backbone = torchvision.models.resnet18()
model = FastSiam(backbone)

trainer = pl.Trainer(
    max_epochs=50,
    devices="auto",
    accelerator="gpu",
    # strategy="ddp", // for multi GPU
    # sync_batchnorm=True,
    # replace_sampler_ddp=True
)
trainer.fit(model=model, train_dataloaders=dataloader)

torch.save(backbone.state_dict(), 'fastsiam_backbone_weights.pth')

import math

import torch
import pytorch_lightning as pl

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from lightly.transforms import FastSiamTransform
from lightly.data.multi_view_collate import MultiViewCollate

def get_coalate_method():
    return MultiViewCollate()

def fast_siam_transform(batch, input_size):
    transform = FastSiamTransform(
        input_size=input_size,
        cj_prob=0
    )

    return transform(batch['image']), 0, batch['metadata']['uri']

class FastSiam(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.projection_head = SimSiamProjectionHead(512, 512, 128)
        self.prediction_head = SimSiamPredictionHead(128, 64, 128)
        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        views, _, _ = batch
        features = [self.forward(view) for view in views]
        zs = torch.stack([z for z, _ in features])
        ps = torch.stack([p for _, p in features])

        loss = 0.0
        for i in range(len(views)):
            mask = torch.arange(len(views), device=self.device) != i
            loss += self.criterion(ps[i], torch.mean(zs[mask], dim=0)) / len(views)

        output_global = features[0][0].detach()
        output_global = torch.nn.functional.normalize(output_global, dim=1)
        self.out_dim = output_global.shape[1]

        output_global_std = torch.std(output_global, 0)
        output_global_std = output_global_std.mean()
        self.avg_output_std_global = 0.9 * self.avg_output_std_global + 0.1 * output_global_std

        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_start(self):
        self.avg_output_std_global = 0.0
        self.out_dim = 0

    def on_train_epoch_end(self):
        collapse_level = max(0.0, 1 - math.sqrt(self.out_dim) * self.avg_output_std_global)
        print(f' Collapse Level: {collapse_level:.2f} / 1.00')

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim

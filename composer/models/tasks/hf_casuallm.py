# Copyright 2021 MosaicML. All Rights Reserved.

"""The ComposerModel base interface for Transformers."""
import logging
import torch
from torchmetrics import MetricCollection
from composer.models import ComposerModel
from composer.metrics import HFCrossEntropy, Perplexity

log = logging.getLogger(__name__)


class HFCausalLanguageModeling(ComposerModel):
    """
    Args:
        huggingface_model: One of the available model names.
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.model = module

        clm_metrics = MetricCollection([HFCrossEntropy(), Perplexity()])
        self.train_metrics = clm_metrics.clone(prefix='train_')
        self.val_metrics = clm_metrics.clone(prefix='val_')
        # define metrics for measurements

    def forward(self, batch):
        output = self.module(**batch)
        return output

    def loss(self, outputs, batch):
        if outputs.get('loss', None) is not None:
            return outputs['loss']
        else:
            raise NotImplementedError('Calculating loss directly not supported for this model.')

    def validate(self, batch):
        assert self.training is False, "For validation, model must be in eval mode"
        output = self.forward(batch)
        return output, None

    def metrics(self, train: bool = False):
        return self.train_metrics if train else self.val_metrics

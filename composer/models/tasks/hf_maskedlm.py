# Copyright 2021 MosaicML. All Rights Reserved.

"""The ComposerModel base interface for Transformers."""
import logging
import torch
from torchmetrics import MetricCollection
from composer.models import ComposerModel
from composer.metrics import HFCrossEntropy, Perplexity

log = logging.getLogger(__name__)


class HFMaskedLanguageModeling(ComposerModel):
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

    def forward(self, batch: Batch) -> Mapping:
        """Runs the forward pass of the model.

        Args:
            batch (~composer.core.types.Batch): A dictionary of Dict[str, Tensor] of inputs that the
                model expects, as found in :meth:`.ComposerTransformer.get_model_inputs`.

        Returns:
            output: A dictionary of model outputs as a ``Mapping``. It will include the loss if `labels` is passed as an input.
        """
        if not isinstance(batch, dict):
            raise ValueError(f'Model expects batch to be a dict, got {type(batch)}')

        for key in self.model_inputs:
            if key not in batch.keys():
                raise ValueError(f'Batch missing key: {key}')

        output = self.module(**batch)  # type: ignore (thirdparty)
        return output

    def loss(self, outputs, batch):
        if outputs.get('loss', None) is not None:
            return outputs['loss']
        else:
            raise NotImplementedError('Calculating loss directly not supported yet.')

    def validate(self, batch):
        """Runs the validation step.

        Args:
            batch (~composer.core.types.Batch): a dictionary of Dict[str, Tensor] of inputs
                that the model expects, as found in :meth:`.ComposerTransformer.get_model_inputs`.

        Returns:
            Tuple[Mapping, None]: A tuple containing the output from the forward pass.
                This is fed into directly into the output of :meth:`.ComposerModel.metrics`.
        """
        assert self.training is False, "For validation, model must be in eval mode"
        output = self.forward(batch)
        return output, None

    def validate(self, batch):
        """Runs the validation step.

        Args:
            batch (BatchDict): a dictionary of Dict[str, Tensor] of inputs
                that the model expects, as found in :meth:`.ComposerTransformer.get_model_inputs`.

        Returns:
            tuple (Tensor, Tensor): with the output from the forward pass and the correct labels.
                This is fed into directly into the output of :meth:`.ComposerModel.metrics`.
        """
        assert self.training is False, "For validation, model must be in eval mode"

        # temporary hack until eval on multiple datasets is finished
        labels = batch.pop('labels')
        output = self.forward(batch)
        output = output['logits']

        # if we are in the single class case, then remove the classes dimension
        if output.shape[1] == 1:
            output = output.squeeze(dim=1)

        return output, labels

    def metrics(self, train: bool = False):
        return self.train_metrics if train else self.val_metrics

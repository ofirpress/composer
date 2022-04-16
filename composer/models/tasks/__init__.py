# Copyright 2021 MosaicML. All Rights Reserved.

"""Model tasks are ComposerModels with forward passes and logging built-in for many common deep learning tasks."""

from composer.models.tasks.classification import ComposerClassifier as ComposerClassifier
from composer.models.tasks.hf_casuallm import HFCausalLanguageModeling as HFCausalLanguageModeling
from composer.models.tasks.hf_maskedlm import HFMaskedLanguageModeling as HFMaskedLanguageModeling

__all__ = ["ComposerClassifier", "HFCausalLanguageModeling", "HFMaskedLanguageModeling"]

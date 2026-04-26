from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseASRModel(nn.Module, ABC):
    """Abstract base class for all ASR models in the project."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x: [batch, channels/features, time] input features (e.g. Mel specs)
            lengths: [batch] sequence lengths for each element in the batch

        Returns:
            log_probs: [batch, time, vocab_size] log softmax probabilities for CTC
            output_lengths: [batch] valid sequence lengths after downsampling
        """

    def get_num_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

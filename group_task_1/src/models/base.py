import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseASRModel(nn.Module, ABC):
    """
    Abstract base class for all ASR models in the project.
    All models must inherit from this class and implement the abstract methods.
    """
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Tensor of shape [batch, channels/features, time] containing input features (e.g. Mel specs)
            lengths: Tensor of shape [batch] containing sequence lengths for each element in the batch
            
        Returns:
            log_probs: Tensor of shape [batch, time, vocab_size] containing log softmax probabilities for CTC
            output_lengths: Tensor of shape [batch] containing the valid sequence lengths after any downsampling in the model
        """
        pass
        
    def get_num_params(self) -> int:
        """
        Returns the number of trainable parameters in the model.
        Used to ensure the model satisfies the < 5M parameters constraint.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

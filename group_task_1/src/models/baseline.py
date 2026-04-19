import torch
import torch.nn as nn

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        return out

class BaselineCRNN(nn.Module):
    """
    A lightweight Convolutional Recurrent Neural Network (CRNN) for ASR.
    Designed to be < 5M parameters.
    """
    def __init__(self, n_mels: int = 80, vocab_size: int = 11, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        # Convolutional Front-End
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ResidualConvBlock(128, 128, dropout),
            ResidualConvBlock(128, 256, dropout),
            nn.MaxPool1d(2),
            ResidualConvBlock(256, 256, dropout)
        )
        
        # RNN Back-End
        self.rnn = nn.GRU(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)
        
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape [batch, n_mels, time]
            lengths: Tensor of shape [batch] containing sequence lengths
            
        Returns:
            log_probs: Tensor of shape [batch, time, vocab_size]
            output_lengths: Tensor of shape [batch] containing downsampled sequence lengths
        """
        # CNN forward pass
        x = self.cnn(x)  # [batch, channels, time]
        
        # Compute new lengths after CNN downsampling (stride=2, maxpool=2 -> total downsample by 4)
        output_lengths = lengths // 4
        
        # Prepare for RNN: [batch, channels, time] -> [batch, time, channels]
        x = x.transpose(1, 2)
        
        # Pack sequence to avoid computing over padded regions
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, output_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_out, _ = self.rnn(packed_x)
        
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        # Classifier
        logits = self.classifier(out)
        
        # Log softmax for CTCLoss
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        
        return log_probs, output_lengths
        
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = BaselineCRNN()
    print(f"Model parameters: {model.get_num_params() / 1e6:.2f} M")

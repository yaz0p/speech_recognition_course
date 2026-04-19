import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
import ast
from typing import Optional, Dict, Any, Tuple

class PrecomputedASRDataset(Dataset):
    def __init__(self, csv_path: str, is_train: bool = True, sample_rate: int = 16000, 
                 n_mels: int = 80, n_fft: int = 400, hop_length: int = 160):
        self.df = pd.read_csv(csv_path)
        self.is_train = is_train
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # SpecAugment for training data
        if self.is_train:
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)
            
    def _add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(waveform) * 0.005
        return waveform + noise
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        row = self.df.iloc[idx]
        
        # Load 1D waveform
        waveform = torch.load(row['tensor_path'])
        
        if self.is_train:
            # Apply time-domain augmentations (noise, random scaling)
            # Add Gaussian noise
            if torch.rand(1).item() > 0.5:
                waveform = self._add_noise(waveform)
                
            # Random volume scaling
            if torch.rand(1).item() > 0.5:
                scale = 0.5 + torch.rand(1).item() # 0.5 to 1.5
                waveform = waveform * scale
                
        # Compute Mel Spectrogram
        mel_spec = self.mel_transform(waveform)
        # Log scale
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # Apply SpecAugment if training
        if self.is_train:
            mel_spec = self.freq_mask(mel_spec)
            mel_spec = self.time_mask(mel_spec)
            
        input_length = mel_spec.shape[-1]
        
        # Load targets
        tokens = torch.tensor(ast.literal_eval(row['tokens']), dtype=torch.long)
        target_length = len(tokens)
        
        return mel_spec, tokens, input_length, target_length

def collate_fn(batch: list[Tuple[torch.Tensor, torch.Tensor, int, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad sequences to max length in batch and return necessary tensors for CTCLoss.
    """
    mel_specs, tokens, input_lengths, target_lengths = zip(*batch)
    
    # Pad mel specs (Shape: [batch, n_mels, time])
    # Time is the last dimension, so we pad it
    max_input_len = max(input_lengths)
    padded_mel_specs = []
    for spec in mel_specs:
        pad_amount = max_input_len - spec.shape[-1]
        padded_spec = torch.nn.functional.pad(spec, (0, pad_amount), mode='constant', value=0.0)
        padded_mel_specs.append(padded_spec)
        
    padded_mel_specs = torch.stack(padded_mel_specs)
    
    # Pad tokens (Shape: [batch, target_length])
    max_target_len = max(target_lengths)
    padded_tokens = []
    for t in tokens:
        pad_amount = max_target_len - len(t)
        padded_t = torch.nn.functional.pad(t, (0, pad_amount), mode='constant', value=0)
        padded_tokens.append(padded_t)
        
    padded_tokens = torch.stack(padded_tokens)
    
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    
    return padded_mel_specs, padded_tokens, input_lengths, target_lengths

class ASRDataModule(pl.LightningDataModule):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.train_csv = Path(config.dataset.preprocessed_dir) / "train_preprocessed.csv"
        self.dev_csv = Path(config.dataset.preprocessed_dir) / "dev_preprocessed.csv"
        self.batch_size = config.dataset.batch_size
        self.num_workers = config.dataset.num_workers
        
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = PrecomputedASRDataset(
                str(self.train_csv), 
                is_train=True,
                sample_rate=self.config.dataset.sample_rate,
                n_mels=self.config.dataset.n_mels,
                n_fft=self.config.dataset.n_fft,
                hop_length=self.config.dataset.hop_length
            )
            self.val_dataset = PrecomputedASRDataset(
                str(self.dev_csv), 
                is_train=False,
                sample_rate=self.config.dataset.sample_rate,
                n_mels=self.config.dataset.n_mels,
                n_fft=self.config.dataset.n_fft,
                hop_length=self.config.dataset.hop_length
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True
        )

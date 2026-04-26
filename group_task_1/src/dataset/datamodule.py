import ast
from pathlib import Path
from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from utils.config import Config


def normalize_log_mel(log_mel: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Per-utterance CMVN on log-mel. log_mel: [..., n_mels, time]."""
    mean = log_mel.mean(dim=-1, keepdim=True)
    std = log_mel.std(dim=-1, keepdim=True)
    return (log_mel - mean) / (std + eps)


class PrecomputedASRDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        is_train: bool = True,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        normalize: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.is_train = is_train
        self.sample_rate = sample_rate
        self.normalize = normalize

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        if self.is_train:
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int, str]:
        row = self.df.iloc[idx]
        spk_id = str(row.get("spk_id", "unknown"))

        waveform = torch.load(row["tensor_path"], weights_only=True)

        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-9)

        # CMVN applied before SpecAugment so masked zeros sit at the mean
        if self.normalize:
            mel_spec = normalize_log_mel(mel_spec)

        if self.is_train:
            mel_spec = self.freq_mask(mel_spec)
            mel_spec = self.time_mask(mel_spec)

        input_length = mel_spec.shape[-1]
        tokens = torch.tensor(ast.literal_eval(row["tokens"]), dtype=torch.long)
        target_length = len(tokens)

        return mel_spec, tokens, input_length, target_length, spk_id


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, int, int, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Pad sequences to max length in batch and return tensors for CTCLoss."""
    (mel_specs, tokens, input_lengths, target_lengths, spk_ids) = zip(
        *batch, strict=False
    )

    max_input_len = max(input_lengths)
    padded_mel_specs = []
    for spec in mel_specs:
        pad_amount = max_input_len - spec.shape[-1]
        padded_spec = torch.nn.functional.pad(
            spec, (0, pad_amount), mode="constant", value=0.0
        )
        padded_mel_specs.append(padded_spec)
    padded_mel_specs = torch.stack(padded_mel_specs)

    max_target_len = max(target_lengths)
    padded_tokens = []
    for t in tokens:
        pad_amount = max_target_len - len(t)
        padded_t = torch.nn.functional.pad(t, (0, pad_amount), mode="constant", value=0)
        padded_tokens.append(padded_t)
    padded_tokens = torch.stack(padded_tokens)

    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return padded_mel_specs, padded_tokens, input_lengths, target_lengths, list(spk_ids)


class ASRDataModule(pl.LightningDataModule):
    def __init__(self, config: Any):
        super().__init__()
        self.config: Config = config
        preprocessed = Path(config.dataset.preprocessed_dir)
        self.train_csv = preprocessed / "train_preprocessed.csv"
        self.dev_csv = preprocessed / "dev_preprocessed.csv"
        self.batch_size = config.dataset.batch_size
        self.num_workers = config.dataset.num_workers

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = PrecomputedASRDataset(
                str(self.train_csv),
                is_train=True,
                sample_rate=self.config.dataset.sample_rate,
                n_mels=self.config.dataset.n_mels,
                n_fft=self.config.dataset.n_fft,
                hop_length=self.config.dataset.hop_length,
                normalize=self.config.dataset.normalize,
            )
            self.val_dataset = PrecomputedASRDataset(
                str(self.dev_csv),
                is_train=False,
                sample_rate=self.config.dataset.sample_rate,
                n_mels=self.config.dataset.n_mels,
                n_fft=self.config.dataset.n_fft,
                hop_length=self.config.dataset.hop_length,
                normalize=self.config.dataset.normalize,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

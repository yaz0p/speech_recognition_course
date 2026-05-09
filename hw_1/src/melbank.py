from typing import Optional

import torch
from torch import nn
from torchaudio import functional as F


class LogMelFilterBanks(nn.Module):
    def __init__(
        self,
        n_fft: int = 400,
        samplerate: int = 16000,
        hop_length: int = 160,
        n_mels: int = 80,
        pad_mode: str = "reflect",
        power: float = 2.0,
        normalize_stft: bool = False,
        onesided: bool = True,
        center: bool = True,
        f_min_hz: float = 0.0,
        f_max_hz: Optional[float] = None,
        norm_mel: Optional[str] = None,
        mel_scale: str = "htk",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.samplerate = samplerate
        self.window_length = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.center = center
        self.onesided = onesided
        self.normalize_stft = normalize_stft
        self.pad_mode = pad_mode
        self.power = power

        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz if f_max_hz is not None else float(samplerate // 2)
        self.norm_mel = norm_mel
        self.mel_scale = mel_scale

        self.register_buffer("window", torch.hann_window(self.window_length))
        self.register_buffer("mel_fbanks", self._init_melscale_fbanks())

    def _init_melscale_fbanks(self):
        n_freqs = (self.n_fft // 2) + 1
        return F.melscale_fbanks(
            n_freqs=n_freqs,
            f_min=self.f_min_hz,
            f_max=self.f_max_hz,
            n_mels=self.n_mels,
            sample_rate=self.samplerate,
            norm=self.norm_mel,
            mel_scale=self.mel_scale,
        )

    def spectrogram(self, x):
        return torch.stft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalize_stft,
            onesided=self.onesided,
            return_complex=True,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): audio signal, shape (batch, time)
        Returns:
            torch.Tensor: log mel filterbanks, shape (batch, n_mels, n_frames)
        """
        spec = self.spectrogram(x)
        power_spec = spec.abs().pow(self.power)
        mel_spec = torch.matmul(power_spec.transpose(1, 2), self.mel_fbanks).transpose(1, 2)
        return torch.log(mel_spec + 1e-6)

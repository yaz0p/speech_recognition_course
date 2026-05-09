import matplotlib.pyplot as plt
import torch
import torchaudio

from dataset import BinarySpeechCommands
from melbank import LogMelFilterBanks


dataset = BinarySpeechCommands(subset="training", download=False)
signal, sr, *_ = dataset[0]

melspec = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr, n_fft=400, hop_length=160, n_mels=80,
    center=True, pad_mode="reflect", power=2.0, norm=None, mel_scale="htk",
)(signal)

logmelbanks = LogMelFilterBanks(
    n_fft=400, samplerate=sr, hop_length=160, n_mels=80,
    center=True, pad_mode="reflect", power=2.0, norm_mel=None, mel_scale="htk",
)(signal)

reference = torch.log(melspec + 1e-6)
assert reference.shape == logmelbanks.shape, "shape mismatch"
assert torch.allclose(reference, logmelbanks, atol=1e-4), "value mismatch"

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Torchaudio (Log) MelSpectrogram")
plt.imshow(reference[0].detach().numpy(), aspect="auto", origin="lower", cmap="viridis")
plt.colorbar(format="%+2.0f dB")

plt.subplot(1, 2, 2)
plt.title("Custom LogMelFilterBanks")
plt.imshow(logmelbanks[0].detach().numpy(), aspect="auto", origin="lower", cmap="viridis")
plt.colorbar(format="%+2.0f dB")

plt.tight_layout()
plt.savefig("visual/mel_comparison.png")

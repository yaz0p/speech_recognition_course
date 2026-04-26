import random
from pathlib import Path

import click
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from dataset.datamodule import normalize_log_mel
from train.train import ASRLightningModule
from utils.config import Config
from utils.metrics import greedy_decoder

MAX_DIGITS = 6   # 999_999 has 6 digits


def trim_silence(waveform: torch.Tensor, sample_rate: int,
                 threshold_db: float = -40.0, padding_ms: int = 50) -> torch.Tensor:
    """
    Trim trailing silence/noise from a 1D waveform using short-time energy.
    Keeps `padding_ms` of audio after the last active chunk.
    """
    chunk = sample_rate // 100  # 10 ms per chunk
    n_chunks = waveform.numel() // chunk
    if n_chunks == 0:
        return waveform

    threshold = 10 ** (threshold_db / 20.0)
    frames = waveform[:n_chunks * chunk].reshape(n_chunks, chunk)
    rms = frames.pow(2).mean(dim=1).sqrt()

    active = (rms > threshold).nonzero(as_tuple=False)
    if active.numel() == 0:
        return waveform

    last_active = int(active[-1].item())
    padding_chunks = max(1, padding_ms * sample_rate // (chunk * 1000))
    end = min(waveform.numel(), (last_active + padding_chunks + 1) * chunk)
    return waveform[:end]


DATA_PATH = Path("./data")
SUBMISSION_PATH = Path("submission/")


class Submission:
    def __init__(self, config_path: str, checkpoint_path: str):
        self.data_path = DATA_PATH
        self.test_df = pd.read_csv(self.data_path / "test.csv")
        self.save_path = self.data_path / SUBMISSION_PATH
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.config = Config.load(config_path)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        print(f"Loading model from {checkpoint_path}...")
        self.model = ASRLightningModule.load_from_checkpoint(
            checkpoint_path,
            config=self.config,
        )
        self.model.eval()
        self.model.to(self.device)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.dataset.sample_rate,
            n_mels=self.config.dataset.n_mels,
            n_fft=self.config.dataset.n_fft,
            hop_length=self.config.dataset.hop_length,
        ).to(self.device)

    def score_file(self, filepath: str | None = None) -> int:
        if filepath is None:
            return random.randint(1_000, 999_999)

        data, sr = sf.read(str(filepath))
        waveform = torch.from_numpy(data).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.transpose(0, 1)  # [channels, time]

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform_cpu = waveform.squeeze(0)
        if sr != self.config.dataset.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.config.dataset.sample_rate
            )
            waveform_cpu = resampler(waveform_cpu)

        waveform_cpu = trim_silence(waveform_cpu, self.config.dataset.sample_rate)

        waveform = waveform_cpu.unsqueeze(0).to(self.device)

        with torch.no_grad():
            mel_spec = self.mel_transform(waveform)
            log_mel_spec = torch.log(mel_spec + 1e-9)
            if getattr(self.config.dataset, "normalize", True):
                log_mel_spec = normalize_log_mel(log_mel_spec)

            input_length = torch.tensor(
                [log_mel_spec.shape[-1]], dtype=torch.long
            ).to(self.device)

            log_probs, _ = self.model(log_mel_spec, input_length)

            predictions = greedy_decoder(log_probs)

        tokens = predictions[0][:MAX_DIGITS]

        if len(tokens) == 0:
            return random.randint(1_000, 999_999)

        try:
            transcription = int("".join(str(p) for p in tokens))
            if not (1_000 <= transcription <= 999_999):
                return random.randint(1_000, 999_999)
        except ValueError:
            return random.randint(1_000, 999_999)

        return transcription

    def prepare_submission(self) -> None:
        print("Scoring test files...")
        tqdm.pandas()
        self.test_df["transcription"] = self.test_df["filename"].progress_apply(
            lambda x: self.score_file(str(self.data_path / x))
        )

    def save_submission(self, path: str | None = None) -> None:
        if not path:
            path = f"{self.save_path}/submission.csv"
        self.test_df[["filename", "transcription"]].to_csv(path, index=False)
        print(f"Submission saved to {path}")


@click.command()
@click.option("--config", default="configs/config.yaml", show_default=True)
@click.option("--ckpt", required=True, help="Path to model checkpoint.")
def main(config: str, ckpt: str) -> None:
    sub = Submission(config, ckpt)
    sub.prepare_submission()
    sub.save_submission()


if __name__ == "__main__":
    main()

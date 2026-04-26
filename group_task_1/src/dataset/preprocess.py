import os
from multiprocessing import Pool
from pathlib import Path

import click
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from audiomentations import (
    AddGaussianNoise,
    ClippingDistortion,
    Compose,
    Gain,
    HighPassFilter,
    LowPassFilter,
    PitchShift,
    RoomSimulator,
    TimeStretch,
)
from tqdm import tqdm

CHAR_MAP = {str(i): i for i in range(10)}

# Per-worker augmentation pipeline (initialised once per process)
_worker_transforms: Compose | None = None


def text_to_sequence(text: str) -> list[int]:
    return [CHAR_MAP[char] for char in str(text)]


def build_augmentation_pipeline() -> Compose:
    return Compose([
        HighPassFilter(p=0.3),
        LowPassFilter(p=0.3),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
        TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=False, p=0.3),
        RoomSimulator(
            min_size_x=3.0, max_size_x=8.0,
            min_size_y=3.0, max_size_y=6.0,
            min_size_z=2.5, max_size_z=4.0,
            min_absorption_value=0.1,
            max_absorption_value=0.4,
            leave_length_unchanged=True,
            p=0.3,
        ),
        ClippingDistortion(
            min_percentile_threshold=0, max_percentile_threshold=10, p=0.3
        ),
    ])


def _init_worker(aug_copies: int) -> None:
    """Called once per worker process. Builds heavy pipeline objects once."""
    global _worker_transforms  # noqa: PLW0603
    _worker_transforms = build_augmentation_pipeline() if aug_copies > 0 else None


def load_clean_waveform(audio_path: Path, sample_rate: int) -> torch.Tensor:
    """Load, resample, mixdown to mono. Returns 1D tensor."""
    data, sr = sf.read(str(audio_path))
    waveform = torch.from_numpy(data).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.transpose(0, 1)  # [channels, time]

    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform.squeeze(0)


def _make_row(row_dict: dict, tensor_path: Path) -> dict:
    new_row = dict(row_dict)
    new_row["tensor_path"] = str(tensor_path)
    trans_val = row_dict.get("transcription")
    if trans_val is not None and str(trans_val).lower() != "nan":
        if isinstance(trans_val, (float, int)):
            trans_str = str(int(trans_val))
        else:
            trans_str = str(trans_val)
        new_row["tokens"] = str(text_to_sequence(trans_str))
    return new_row


def _process_row(args: tuple) -> list[dict]:
    """Worker function: load one audio file, save clean + augmented copies."""
    row_dict, base_dir_str, target_dir_str, sample_rate, aug_copies = args

    audio_path = Path(base_dir_str) / str(row_dict["filename"])
    target_dir = Path(target_dir_str)
    stem = Path(str(row_dict["filename"])).stem

    results: list[dict] = []
    try:
        waveform = load_clean_waveform(audio_path, sample_rate)

        clean_path = target_dir / f"{stem}.pt"
        torch.save(waveform, clean_path)
        results.append(_make_row(row_dict, clean_path))

        if _worker_transforms is not None and aug_copies > 0:
            wav_np = waveform.numpy()
            for i in range(aug_copies):
                aug_wav = _worker_transforms(samples=wav_np, sample_rate=sample_rate)
                aug_path = target_dir / f"{stem}_aug{i:02d}.pt"
                torch.save(torch.from_numpy(aug_wav).float(), aug_path)
                results.append(_make_row(row_dict, aug_path))
    except Exception as e:
        print(f"Failed: {audio_path}: {e}")

    return results


def preprocess_dataset(
    csv_path: str,
    output_dir_str: str,
    sample_rate: int = 16000,
    aug_copies: int = 0,
    num_workers: int = os.cpu_count() or 1,
) -> None:
    df = pd.read_csv(csv_path)
    base_dir = Path("data")
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_dir = Path(df.iloc[0]["filename"]).parent.name
    target_dir = output_dir / split_dir
    target_dir.mkdir(exist_ok=True)

    tasks = [
        (row.to_dict(), str(base_dir), str(target_dir), sample_rate, aug_copies)
        for _, row in df.iterrows()
    ]

    total_expected = len(df) * (1 + aug_copies)
    print(
        f"Processing {csv_path} (aug_copies={aug_copies}, "
        f"workers={num_workers}, total={total_expected} files)..."
    )

    new_data: list[dict] = []
    with Pool(
        processes=num_workers, initializer=_init_worker, initargs=(aug_copies,)
    ) as pool:
        for rows in tqdm(pool.imap(_process_row, tasks), total=len(tasks)):
            new_data.extend(rows)

    new_csv_path = output_dir / f"{Path(csv_path).stem}_preprocessed.csv"
    pd.DataFrame(new_data).to_csv(new_csv_path, index=False)
    print(f"Saved {new_csv_path} ({len(new_data)} rows)")


@click.command()
@click.option("--train-csv", default="data/train.csv", show_default=True)
@click.option("--dev-csv", default="data/dev.csv", show_default=True)
@click.option("--out-dir", default="data/preprocessed", show_default=True)
@click.option("--sr", default=16000, show_default=True, type=int)
@click.option("--aug-copies", default=0, show_default=True, type=int,
              help="Offline augmented copies per train sample (0 = none).")
@click.option("--workers", default=os.cpu_count() or 1, show_default=True, type=int,
              help="Number of parallel worker processes.")
def main(
    train_csv: str,
    dev_csv: str,
    out_dir: str,
    sr: int,
    aug_copies: int,
    workers: int,
) -> None:
    """Preprocess audio datasets into .pt tensors with optional offline augmentation."""
    preprocess_dataset(
        train_csv, out_dir, sr, aug_copies=aug_copies, num_workers=workers
    )
    preprocess_dataset(dev_csv, out_dir, sr, aug_copies=0, num_workers=workers)


if __name__ == "__main__":
    main()

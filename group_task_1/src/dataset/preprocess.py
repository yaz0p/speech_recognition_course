import os
import torch
import torchaudio
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

# Dictionary mapping digits to tokens
# 0-9 are mapping to themselves (as strings), plus blank token for CTC
CHAR_MAP = {str(i): i for i in range(10)}
BLANK_TOKEN = 10  # Reserving 10 for CTC blank token

def text_to_sequence(text: str) -> list[int]:
    """Convert string of numbers to sequence of int tokens"""
    return [CHAR_MAP[char] for char in str(text)]

def process_audio(
    audio_path: Path, 
    target_path: Path, 
    sample_rate: int, 
    mel_transform: torchaudio.transforms.MelSpectrogram
):
    """Load audio, resample, compute mel spectrogram, and save to disk."""
    import soundfile as sf
    data, sr = sf.read(str(audio_path))
    waveform = torch.tensor(data).float()
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.shape[1] == 2: # Stereo handling
        waveform = waveform.transpose(0, 1)
    
    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    
    # Mixdown to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Compute Mel spectrogram
    mel_spec = mel_transform(waveform)
    # Convert to log scale (adding a small epsilon to avoid log(0))
    log_mel_spec = torch.log(mel_spec + 1e-9)
    
    # Save the tensor
    torch.save(log_mel_spec.squeeze(0), target_path)

def preprocess_dataset(csv_path: str, output_dir_str: str, sample_rate: int = 16000, n_mels: int = 80):
    df = pd.read_csv(csv_path)
    base_dir = Path("data")
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize MelTransform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=400,
        hop_length=160
    )
    
    new_data = []
    
    print(f"Processing {csv_path}...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = str(row['filename'])
        audio_path = base_dir / filename
        
        # Create a unique filename for the tensor
        tensor_filename = Path(filename).stem + ".pt"
        # Determine split (train/dev/test)
        split_dir = Path(filename).parent.name
        
        target_dir = output_dir / split_dir
        target_dir.mkdir(exist_ok=True)
        target_path = target_dir / tensor_filename
        
        try:
            process_audio(audio_path, target_path, sample_rate, mel_transform)
            
            # Save new row information
            new_row = row.to_dict()
            new_row['tensor_path'] = str(target_path)
            
            # Process transcription if available (not available in test set)
            if 'transcription' in row.index:
                trans_val = row['transcription']
                if str(trans_val).lower() != 'nan':
                    if isinstance(trans_val, (float, int)):
                        trans_str = str(int(trans_val))
                    else:
                        trans_str = str(trans_val)
                    tokens = text_to_sequence(trans_str)
                    new_row['tokens'] = str(tokens)
            
            new_data.append(new_row)
        except Exception as e:
            print(f"Failed processing {audio_path}: {e}")
            
    # Save the updated dataframe
    new_csv_path = output_dir / f"{Path(csv_path).stem}_preprocessed.csv"
    pd.DataFrame(new_data).to_csv(new_csv_path, index=False)
    print(f"Saved metadata to {new_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--dev_csv", default="data/dev.csv")
    parser.add_argument("--out_dir", default="data/preprocessed")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=80)
    
    args = parser.parse_args()
    
    preprocess_dataset(args.train_csv, args.out_dir, args.sr, args.n_mels)
    preprocess_dataset(args.dev_csv, args.out_dir, args.sr, args.n_mels)

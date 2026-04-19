import torch
import torchaudio
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from src.dataset.datamodule import PrecomputedASRDataset

def export_augmented_samples(csv_path: str, preprocessed_dir: str, output_dir_str: str, num_samples: int = 10, sample_rate: int = 16000):
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the train split to get augmented samples
    dataset = PrecomputedASRDataset(
        csv_path=str(Path(preprocessed_dir) / "train_preprocessed.csv"),
        is_train=True,
        sample_rate=sample_rate,
        n_mels=80,
        n_fft=400,
        hop_length=160
    )
    
    print(f"Exporting {num_samples} augmented samples to {output_dir}...")
    for i in tqdm(range(num_samples)):
        row = dataset.df.iloc[i]
        original_filename = Path(row['filename']).name
        
        # Load the preprocessed 1D waveform (before augmentations)
        original_waveform = torch.load(row['tensor_path'])
        
        # Save the original for comparison
        orig_path = output_dir / f"{i:02d}_original_{original_filename}"
        import soundfile as sf
        sf.write(str(orig_path), original_waveform.squeeze().numpy(), sample_rate)
        
        # Apply the exact same augmentations as the dataset
        waveform = original_waveform.clone()
        
        # 1. Add noise
        if torch.rand(1).item() > 0.5:
            waveform = dataset._add_noise(waveform)
            
        # 2. Random volume scaling
        if torch.rand(1).item() > 0.5:
            scale = 0.5 + torch.rand(1).item() # 0.5 to 1.5
            waveform = waveform * scale
            
        # Note: We don't save the SpecAugment (Mel Spectrogram masks) because 
        # those are applied in the frequency domain, not the time domain, 
        # and cannot be easily inverted back to listenable audio.
        # Here we only care about hearing the time-domain augmentations (noise/volume).
        
        aug_path = output_dir / f"{i:02d}_augmented_{original_filename}"
        sf.write(str(aug_path), waveform.squeeze().numpy(), sample_rate)
        
    print(f"Done! Samples saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/train.csv")
    parser.add_argument("--prep_dir", default="data/preprocessed")
    parser.add_argument("--out_dir", default="data/samples")
    parser.add_argument("--num", type=int, default=10)
    
    args = parser.parse_args()
    export_augmented_samples(args.csv, args.prep_dir, args.out_dir, args.num)

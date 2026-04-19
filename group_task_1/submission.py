import random
import torchaudio
import pandas as pd
import torch
from pathlib import Path
from src.train.train import ASRLightningModule
from src.utils.config import Config
from src.utils.metrics import greedy_decoder
from tqdm import tqdm


DATA_PATH = Path("./data")
SUBMISSION_PATH = Path("submission/")


class Submission(object):
    def __init__(self, config_path: str, checkpoint_path: str):
        self.data_path = DATA_PATH
        self.test_df = pd.read_csv(f"{self.data_path}/test.csv")
        self.save_path = self.data_path / SUBMISSION_PATH
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        self.config = Config.load(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f"Loading model from {checkpoint_path}...")
        self.model = ASRLightningModule.load_from_checkpoint(
            checkpoint_path, 
            config=self.config
        )
        self.model.eval()
        self.model.to(self.device)
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.dataset.sample_rate,
            n_mels=self.config.dataset.n_mels,
            n_fft=self.config.dataset.n_fft,
            hop_length=self.config.dataset.hop_length
        ).to(self.device)

    def score_file(self, filepath: str | None = None):
        if filepath is None:
            return random.randint(1_000, 999_999)
            
        import soundfile as sf
        data, sr = sf.read(filepath)
        waveform = torch.tensor(data).float().to(self.device)
        
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.shape[1] == 2:
            waveform = waveform.transpose(0, 1)
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if sr != self.config.dataset.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.config.dataset.sample_rate).to(self.device)
            waveform = resampler(waveform)
            
        with torch.no_grad():
            mel_spec = self.mel_transform(waveform)
            log_mel_spec = torch.log(mel_spec + 1e-9)
            
            input_length = torch.tensor([log_mel_spec.shape[-1]], dtype=torch.long).to(self.device)
            log_mel_spec = log_mel_spec.unsqueeze(0) # add batch dim
            
            log_probs, _ = self.model(log_mel_spec, input_length)
            
            predictions = greedy_decoder(log_probs, blank_id=10)
            
        if len(predictions[0]) == 0:
            return random.randint(1_000, 999_999)
            
        try:
            transcription = int("".join([str(p) for p in predictions[0]]))
        except ValueError:
            transcription = random.randint(1_000, 999_999)
            
        return transcription

    def prepare_submission(self):
        print("Scoring test files...")
        tqdm.pandas()
        self.test_df["transcription"] = self.test_df["filename"].progress_apply(
            lambda x: self.score_file(str(self.data_path.parent / x))
        )

    def save_submission(self, path: str | None = None):
        if not path:
            path = f"{self.save_path}/submission.csv"
        self.test_df[["filename", "transcription"]].to_csv(path, index=False)
        print(f"Submission saved to {path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    
    sub = Submission(args.config, args.ckpt)
    sub.prepare_submission()
    sub.save_submission()

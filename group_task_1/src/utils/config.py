import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatasetConfig:
    train_csv: str
    dev_csv: str
    test_csv: str
    preprocessed_dir: str
    sample_rate: int
    n_mels: int
    n_fft: int
    hop_length: int
    num_workers: int
    batch_size: int

@dataclass
class ModelConfig:
    name: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    dropout: float

@dataclass
class TrainingConfig:
    learning_rate: float
    weight_decay: float
    max_epochs: int
    patience: int
    gradient_clip_val: float
    precision: str
    devices: int
    accelerator: str
    strategy: str

@dataclass
class LoggingConfig:
    project_name: str
    experiment_name: str
    save_dir: str

@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls(
            dataset=DatasetConfig(**data["dataset"]),
            model=ModelConfig(**data["model"]),
            training=TrainingConfig(**data["training"]),
            logging=LoggingConfig(**data["logging"])
        )

import yaml
from dataclasses import dataclass
from typing import Optional, Union, Any

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
    learning_rate: Any
    weight_decay: Any
    max_epochs: int
    patience: int
    gradient_clip_val: float
    precision: str
    devices: int
    accelerator: str
    strategy: str

    def __post_init__(self):
        if isinstance(self.learning_rate, str):
            self.learning_rate = float(str(self.learning_rate))
        elif isinstance(self.learning_rate, (int, float)):
            self.learning_rate = float(self.learning_rate)
            
        if isinstance(self.weight_decay, str):
            self.weight_decay = float(str(self.weight_decay))
        elif isinstance(self.weight_decay, (int, float)):
            self.weight_decay = float(self.weight_decay)

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

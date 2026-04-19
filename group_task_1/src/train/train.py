import torch
import pytorch_lightning as pl
from torch import optim
from src.models.baseline import BaselineCRNN
from src.utils.metrics import greedy_decoder, compute_cer
import argparse
from src.utils.config import Config
from src.dataset.datamodule import ASRDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

class ASRLightningModule(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters(config.__dict__)
        self.config = config
        
        self.model = BaselineCRNN(
            n_mels=config.dataset.n_mels,
            vocab_size=config.model.vocab_size,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout
        )
        
        # CTC Loss: blank=10 (reserved blank token)
        self.loss_fn = torch.nn.CTCLoss(blank=10, zero_infinity=True)

    def forward(self, x, lengths):
        return self.model(x, lengths)

    def _shared_step(self, batch, batch_idx, prefix):
        mel_specs, tokens, input_lengths, target_lengths = batch
        
        log_probs, output_lengths = self(mel_specs, input_lengths)
        
        # CTCLoss expects log_probs to be of shape (time, batch, vocab_size)
        log_probs_t = log_probs.transpose(0, 1)
        
        loss = self.loss_fn(log_probs_t, tokens, output_lengths, target_lengths)
        
        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)
        
        # Calculate CER periodically (it's slow, so not every batch during training)
        if prefix == "val" or (prefix == "train" and batch_idx % 10 == 0):
            # Convert tokens tensor back to lists of ints for distance calculation
            targets_list = []
            for t, l in zip(tokens, target_lengths):
                targets_list.append(t[:l].tolist())
                
            predictions = greedy_decoder(log_probs, blank_id=10)
            cer = compute_cer(predictions, targets_list)
            
            self.log(f"{prefix}_cer", cer, prog_bar=True, sync_dist=True)
            
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Reduce LR on plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_cer"
            }
        }

def train(config_path: str):
    config = Config.load(config_path)
    
    pl.seed_everything(42)
    
    datamodule = ASRDataModule(config)
    model = ASRLightningModule(config)
    
    wandb_logger = WandbLogger(
        project=config.logging.project_name,
        name=config.logging.experiment_name,
        save_dir=config.logging.save_dir
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.logging.save_dir,
        filename=f"{config.logging.experiment_name}-{{epoch:02d}}-{{val_cer:.4f}}",
        monitor="val_cer",
        mode="min",
        save_top_k=3
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Enable mixed precision and DDP based on config
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        strategy=config.training.strategy,
        precision=config.training.precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=config.training.gradient_clip_val,
        deterministic=True
    )
    
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    
    train(args.config)

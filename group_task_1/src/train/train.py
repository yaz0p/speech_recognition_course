import math
from collections import defaultdict

import click
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from torch import optim

from dataset.datamodule import ASRDataModule
from models.baseline import BaselineCRNN
from utils.config import Config
from utils.metrics import BLANK_ID, compute_cer, greedy_decoder


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
            dropout=config.model.dropout,
        )

        self.loss_fn = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)

        # reset at the start of each validation epoch
        self._val_preds_per_spk: dict[str, list[list[int]]] = defaultdict(list)
        self._val_targets_per_spk: dict[str, list[list[int]]] = defaultdict(list)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x, lengths)

    def _decode_targets(
        self, tokens: torch.Tensor, target_lengths: torch.Tensor
    ) -> list[list[int]]:
        return [t[:n].tolist() for t, n in zip(tokens, target_lengths, strict=False)]

    def _shared_step(self, batch: tuple, batch_idx: int, prefix: str) -> torch.Tensor:
        mel_specs, tokens, input_lengths, target_lengths, spk_ids = batch

        log_probs, output_lengths = self(mel_specs, input_lengths)

        log_probs_t = log_probs.transpose(0, 1)  # CTCLoss: (time, batch, vocab)

        loss = self.loss_fn(log_probs_t, tokens, output_lengths, target_lengths)

        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)

        need_cer = prefix == "val" or (prefix == "train" and batch_idx % 10 == 0)
        if need_cer:
            predictions = greedy_decoder(log_probs)
            targets_list = self._decode_targets(tokens, target_lengths)
            cer = compute_cer(predictions, targets_list)
            self.log(f"{prefix}_cer", cer, prog_bar=True, sync_dist=True)

            if prefix == "val":
                for pred, target, spk in zip(
                    predictions, targets_list, spk_ids, strict=False
                ):
                    self._val_preds_per_spk[spk].append(pred)
                    self._val_targets_per_spk[spk].append(target)

        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "val")

    def on_validation_epoch_start(self) -> None:
        self._val_preds_per_spk.clear()
        self._val_targets_per_spk.clear()

    def on_validation_epoch_end(self) -> None:
        for spk in sorted(self._val_preds_per_spk.keys()):
            cer = compute_cer(
                self._val_preds_per_spk[spk], self._val_targets_per_spk[spk]
            )
            self.log(f"val_cer/{spk}", cer, sync_dist=True)

    def configure_optimizers(self) -> dict:
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        warmup_steps = max(1, int(self.config.training.warmup_steps))
        total_steps = max(
            warmup_steps + 1, int(self.trainer.estimated_stepping_batches)
        )
        min_lr_ratio = 0.1

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def train(config_path: str) -> None:
    config = Config.load(config_path)

    pl.seed_everything(42)

    datamodule = ASRDataModule(config)
    model = ASRLightningModule(config)

    wandb_logger = WandbLogger(
        project=config.logging.project_name,
        name=config.logging.experiment_name,
        save_dir=config.logging.save_dir,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.logging.save_dir,
        filename=f"{config.logging.experiment_name}-{{epoch:02d}}-{{val_cer:.4f}}",
        monitor="val_cer",
        mode="min",
        save_top_k=3,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_cer",
        mode="min",
        patience=config.training.patience,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        strategy=config.training.strategy,
        precision=config.training.precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        gradient_clip_val=config.training.gradient_clip_val,
        deterministic=False,  # CTCLoss has no deterministic CUDA implementation
    )

    trainer.fit(model, datamodule)


@click.command()
@click.option("--config", default="configs/config.yaml", show_default=True)
def main(config: str) -> None:
    train(config)


if __name__ == "__main__":
    main()

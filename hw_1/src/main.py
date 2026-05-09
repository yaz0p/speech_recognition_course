import time

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from thop import profile
from torch import nn
from torch.utils.data import DataLoader

from dataset import BinarySpeechCommands
from melbank import LogMelFilterBanks


SAMPLE_LEN = 16000


def collate_fn(batch):
    tensors, targets = [], []
    for waveform, _, label, _, _ in batch:
        seq_len = waveform.shape[1]
        if seq_len < SAMPLE_LEN:
            waveform = torch.nn.functional.pad(waveform, (0, SAMPLE_LEN - seq_len))
        elif seq_len > SAMPLE_LEN:
            waveform = waveform[:, :SAMPLE_LEN]
        tensors.append(waveform[0])
        targets.append(1 if label == "yes" else 0)
    return torch.stack(tensors), torch.tensor(targets)


class SpeechCNN(pl.LightningModule):
    def __init__(self, n_mels: int = 80, groups: int = 1, hidden_channels: int = 32):
        super().__init__()
        self.save_hyperparameters()
        self.mel_extractor = LogMelFilterBanks(n_mels=n_mels)
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, hidden_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, stride=2, padding=2, groups=groups),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, stride=2, padding=2, groups=groups),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_channels, 2)
        self.loss_fn = nn.CrossEntropyLoss()

        self.history_train_loss = []
        self.history_val_acc = []
        self._epoch_train_losses = []
        self._epoch_val_accs = []
        self._epoch_times = []
        self.epoch_start_time = 0.0
        self.avg_epoch_time = 0.0

    def forward(self, x):
        x = self.mel_extractor(x)
        x = self.conv(x)
        x = torch.mean(x, dim=-1)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self._epoch_train_losses.append(loss.item())
        return loss

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_loss = sum(self._epoch_train_losses) / len(self._epoch_train_losses)
        self.history_train_loss.append(epoch_loss)
        self._epoch_train_losses.clear()

        self._epoch_times.append(time.time() - self.epoch_start_time)
        self.avg_epoch_time = sum(self._epoch_times) / len(self._epoch_times)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = torch.argmax(self(x), dim=1)
        acc = (preds == y).float().mean()
        self._epoch_val_accs.append(acc.item())

    def on_validation_epoch_end(self):
        if self._epoch_val_accs:
            self.history_val_acc.append(sum(self._epoch_val_accs) / len(self._epoch_val_accs))
            self._epoch_val_accs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = torch.argmax(self(x), dim=1)
        acc = (preds == y).float().mean()
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def calculate_model_complexity(model, input_shape=(1, SAMPLE_LEN)):
    dummy_input = torch.randn(*input_shape).to(model.device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
    return params, macs * 2


def build_loaders(batch_size: int = 128, num_workers: int = 2):
    train_ds = BinarySpeechCommands(subset="training")
    val_ds = BinarySpeechCommands(subset="validation")
    test_ds = BinarySpeechCommands(subset="testing")
    loader_kwargs = dict(batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)
    return (
        DataLoader(train_ds, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, shuffle=False, **loader_kwargs),
    )


def run_experiments():
    train_loader, val_loader, test_loader = build_loaders()
    results = []

    print("\n=== n_mels experiment ===")
    histories_n_mels = {}
    for n_mels in [20, 40, 80]:
        model = SpeechCNN(n_mels=n_mels, groups=1)
        params, flops = calculate_model_complexity(model)
        print(f"[n_mels={n_mels}] params={params} flops={flops}")

        trainer = pl.Trainer(max_epochs=20, enable_progress_bar=True)
        trainer.fit(model, train_loader, val_loader)
        test_res = trainer.test(model, test_loader)

        histories_n_mels[f"n_mels={n_mels}"] = model.history_train_loss
        results.append({"experiment": "n_mels", "value": n_mels, "test_acc": test_res[0]["test_acc"]})

    plt.figure(figsize=(8, 5))
    for label, losses in histories_n_mels.items():
        plt.plot(range(1, len(losses) + 1), losses, marker="o", label=label)
    plt.title("Training Loss for different n_mels")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("visual/report_loss_vs_nmels.png", bbox_inches="tight", dpi=300)
    plt.close()

    print("\n=== groups experiment ===")
    groups_to_try = [2, 4, 8, 16]
    grp_times, grp_params, grp_flops = [], [], []
    for grp in groups_to_try:
        model = SpeechCNN(n_mels=80, groups=grp, hidden_channels=32)
        params, flops = calculate_model_complexity(model)
        print(f"[groups={grp}] params={params} flops={flops}")

        trainer = pl.Trainer(max_epochs=5, enable_progress_bar=True)
        trainer.fit(model, train_loader, val_loader)
        test_res = trainer.test(model, test_loader)

        grp_times.append(model.avg_epoch_time)
        grp_params.append(params / 1000)
        grp_flops.append(flops / 1e6)
        results.append({"experiment": "groups", "value": grp, "test_acc": test_res[0]["test_acc"]})

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(groups_to_try, grp_times, marker="o", color="red")
    axes[0].set_title("Avg Epoch Time vs Groups")
    axes[0].set_xlabel("Groups")
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_xticks(groups_to_try)
    axes[0].grid(True)

    axes[1].plot(groups_to_try, grp_params, marker="s", color="green")
    axes[1].set_title("Model Parameters vs Groups")
    axes[1].set_xlabel("Groups")
    axes[1].set_ylabel("Parameters (Thousands)")
    axes[1].set_xticks(groups_to_try)
    axes[1].grid(True)

    axes[2].plot(groups_to_try, grp_flops, marker="^", color="purple")
    axes[2].set_title("Model Complexity vs Groups")
    axes[2].set_xlabel("Groups")
    axes[2].set_ylabel("FLOPs (Millions)")
    axes[2].set_xticks(groups_to_try)
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("visual/report_groups_analysis.png", bbox_inches="tight", dpi=300)
    plt.close()

    print("\n=== final results ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    run_experiments()

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataloader_1to69 import get_dataloader


def set_seed(seed: int = 614):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Config:
    data_path: str = "path/to/normalized_era5"
    output_root: str = "path/to/output/linear_baseline"
    checkpoint_name: str = "linear_best.pth"
    results_name: str = "results_linear.txt"
    lr: float = 1e-2
    weight_decay: float = 0.0
    batch_size: int = 32
    epochs: int = 30
    patience: int = 10


class LinearRegressionBaseline(nn.Module):
    """Pixel-wise linear regression baseline using precipitation only."""

    def __init__(self, in_chans: int = 1, out_chans: int = 69):
        super().__init__()
        self.linear = nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == 161 or x.shape[-2] == 161:
            x = x[:, :, :160, :160]
        return self.linear(x)


class Logger:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def write(self, message: str):
        print(message)
        with open(self.file_path, "a", encoding="utf-8") as file:
            file.write(message + "\n")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    per_channel = None

    for inputs, targets in loader:
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()
        outputs = model(inputs)
        total_loss += F.mse_loss(outputs, targets).item()
        current = F.mse_loss(outputs, targets, reduction="none").mean(dim=(0, 2, 3)).cpu()
        per_channel = current if per_channel is None else per_channel + current

    mean_loss = total_loss / len(loader)
    per_channel = (per_channel / len(loader)).numpy()
    return mean_loss, per_channel


def train_linear_baseline(config: Config):
    os.makedirs(config.output_root, exist_ok=True)
    log_path = os.path.join(config.output_root, "train_linear.log")
    logger = Logger(log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.write(f"Device: {device}")

    train_loader, valid_loader, test_loader = get_dataloader(
        train_batch_size=config.batch_size,
        valid_batch_size=config.batch_size,
        test_batch_size=config.batch_size,
        base_path=config.data_path,
    )

    model = LinearRegressionBaseline(in_chans=1, out_chans=69).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_valid = float("inf")
    best_epoch = 0
    patience_counter = 0
    checkpoint_path = os.path.join(config.output_root, config.checkpoint_name)

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}", leave=False):
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        valid_loss, _ = evaluate(model, valid_loader, device)
        logger.write(f"Epoch {epoch + 1:02d} | train_mse={train_loss:.6f} | valid_mse={valid_loss:.6f}")

        if valid_loss < best_valid:
            best_valid = valid_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({"model_state": model.state_dict(), "epoch": best_epoch, "valid_mse": best_valid}, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.write("Early stopping triggered.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_per_channel = evaluate(model, test_loader, device)

    results_path = os.path.join(config.output_root, config.results_name)
    with open(results_path, "w", encoding="utf-8") as file:
        file.write("Linear regression baseline\n")
        file.write(f"Best epoch: {best_epoch}\n")
        file.write(f"Best validation MSE: {best_valid:.6f}\n")
        file.write(f"Test MSE: {test_loss:.6f}\n")
        file.write("Per-channel test MSE\n")
        for channel, value in enumerate(test_per_channel.tolist()):
            file.write(f"{channel}\t{value:.6f}\n")

    logger.write(f"Best epoch: {best_epoch}")
    logger.write(f"Test MSE: {test_loss:.6f}")
    logger.write(f"Saved results to: {results_path}")


if __name__ == "__main__":
    set_seed(614)
    train_linear_baseline(Config())

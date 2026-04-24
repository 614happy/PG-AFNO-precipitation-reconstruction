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
    output_root: str = "path/to/output/unet_baseline"
    batch_size: int = 12
    epochs: int = 100
    patience: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-5


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class WeatherUNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 69):
        super().__init__()
        self.inc = DoubleConv(in_ch, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4)
        x = self.conv1(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, x2], dim=1))
        x = self.up3(x)
        x = self.conv3(torch.cat([x, x1], dim=1))
        return self.outc(x)


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
    for inputs, targets in loader:
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()
        outputs = model(inputs)
        total_loss += F.mse_loss(outputs, targets).item()
    return total_loss / len(loader)


def train_unet(config: Config):
    os.makedirs(config.output_root, exist_ok=True)
    logger = Logger(os.path.join(config.output_root, "train_unet.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.write(f"Device: {device}")

    train_loader, valid_loader, test_loader = get_dataloader(
        train_batch_size=config.batch_size,
        valid_batch_size=config.batch_size,
        test_batch_size=config.batch_size,
        base_path=config.data_path,
    )

    model = WeatherUNet(in_ch=1, out_ch=69).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    best_valid = float("inf")
    best_epoch = 0
    patience_counter = 0
    checkpoint_path = os.path.join(config.output_root, "unet_best.pth")

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

        scheduler.step()
        train_loss /= len(train_loader)
        valid_loss = evaluate(model, valid_loader, device)
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
    test_loss = evaluate(model, test_loader, device)

    results_path = os.path.join(config.output_root, "results_unet.txt")
    with open(results_path, "w", encoding="utf-8") as file:
        file.write("Model: unet\n")
        file.write(f"Best epoch: {best_epoch}\n")
        file.write(f"Best validation MSE: {best_valid:.6f}\n")
        file.write(f"Test MSE: {test_loss:.6f}\n")

    logger.write(f"Best epoch: {best_epoch}")
    logger.write(f"Test MSE: {test_loss:.6f}")
    logger.write(f"Saved results to: {results_path}")


if __name__ == "__main__":
    set_seed(614)
    train_unet(Config())

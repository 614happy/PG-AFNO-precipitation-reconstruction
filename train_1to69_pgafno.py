import logging
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataloader_1to69_pgafno import get_dataloader
from models_1to69_pgafno import PGAFNO


def set_seed(seed: int = 614) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Config:
    data_root: str = "path/to/normalized_era5"
    output_root: str = "path/to/output_dir"
    pretrained_checkpoint: str | None = None

    seed: int = 614
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 12
    epochs: int = 100
    patience_limit: int = 30
    geostrophic_weight: float = 0.005
    num_workers: int = 0


config = Config()

TARGET_CHANNELS = list(range(69))
OMEGA = 7.2921e-5
GRID_HEIGHT = 160
GRID_WIDTH = 160
LAT_ARRAY = np.linspace(50.0, 10.0, GRID_HEIGHT)
LAT_GRID = LAT_ARRAY.reshape(GRID_HEIGHT, 1)

F_CORIOLIS_CPU = torch.tensor(
    2 * OMEGA * np.sin(np.deg2rad(LAT_GRID)),
    dtype=torch.float32,
).view(1, 1, GRID_HEIGHT, 1)
DY = 27830.0
DX_CPU = torch.tensor(
    27830.0 * np.cos(np.deg2rad(LAT_GRID)),
    dtype=torch.float32,
).view(1, 1, GRID_HEIGHT, 1)
VALID_MASK_CPU = torch.tensor(
    LAT_ARRAY >= 25.0,
    dtype=torch.bool,
).view(1, 1, GRID_HEIGHT, 1)

MEAN_PHI_500 = 2.90373724e4
STD_PHI_500 = 3.25549419e4
MEAN_U_500 = 9.80464536
STD_U_500 = 1.06490084e1
MEAN_V_500 = -4.10185719e-1
STD_V_500 = 6.01023122


def get_logger(log_dir: str, name: str, log_filename: str = "train.log") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename), mode="w")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, strict: bool = True) -> torch.nn.Module:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    cleaned_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(cleaned_state_dict, strict=strict)
    return model


def save_checkpoint(
    save_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
) -> None:
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    cpu_state_dict = {key: value.cpu() for key, value in raw_model.state_dict().items()}
    torch.save(
        {
            "model_state": cpu_state_dict,
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        save_path,
    )


def geostrophic_physics_loss(pred_norm: torch.Tensor) -> torch.Tensor:
    device = pred_norm.device
    f_coriolis = F_CORIOLIS_CPU.to(device)
    dx = DX_CPU.to(device)
    valid_mask = VALID_MASK_CPU.to(device)

    phi_norm = pred_norm[:, 7:8, :, :]
    u_norm = pred_norm[:, 33:34, :, :]
    v_norm = pred_norm[:, 46:47, :, :]

    phi_phys = phi_norm * STD_PHI_500 + MEAN_PHI_500

    grad_y, grad_x = torch.gradient(phi_phys, dim=(2, 3), spacing=(DY, 1.0))
    dphi_dy = grad_y / (-1.0)
    dphi_dx = grad_x / dx

    f_safe = torch.clamp(f_coriolis, min=1e-5)
    ug_phys = -dphi_dy / f_safe
    vg_phys = dphi_dx / f_safe

    ug_phys = torch.clamp(ug_phys, min=-200.0, max=200.0)
    vg_phys = torch.clamp(vg_phys, min=-200.0, max=200.0)

    ug_norm = (ug_phys - MEAN_U_500) / STD_U_500
    vg_norm = (vg_phys - MEAN_V_500) / STD_V_500

    valid_mask = valid_mask[:, :, 2:-2, :]
    u_error = (u_norm[:, :, 2:-2, :] - ug_norm[:, :, 2:-2, :]) ** 2
    v_error = (v_norm[:, :, 2:-2, :] - vg_norm[:, :, 2:-2, :]) ** 2

    loss_u = (u_error * valid_mask).float().mean()
    loss_v = (v_error * valid_mask).float().mean()
    return loss_u + loss_v


def evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    channel_loss_sum = torch.zeros(len(TARGET_CHANNELS), device=device)

    with torch.no_grad():
        for inputs, targets in data_loader:
            targets = targets[:, TARGET_CHANNELS, :, :]
            inputs = torch.nan_to_num(inputs.to(device).float())
            targets = torch.nan_to_num(targets.to(device).float())

            outputs = model(inputs)

            batch_loss = F.mse_loss(outputs, targets)
            loss_sum += batch_loss.item()

            batch_channel_loss = F.mse_loss(outputs, targets, reduction="none").mean(dim=(0, 2, 3))
            channel_loss_sum += batch_channel_loss

    mean_loss = loss_sum / len(data_loader)
    mean_channel_loss = (channel_loss_sum / len(data_loader)).detach().cpu().numpy()
    return mean_loss, mean_channel_loss


def train(config: Config) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training.")

    set_seed(config.seed)

    os.makedirs(config.output_root, exist_ok=True)
    logger = get_logger(config.output_root, "pgafno_train")

    logger.info("Starting PG-AFNO training.")
    logger.info(
        "Settings: lr=%s, weight_decay=%s, batch_size=%s, epochs=%s, patience=%s, alpha=%s",
        config.lr,
        config.weight_decay,
        config.batch_size,
        config.epochs,
        config.patience_limit,
        config.geostrophic_weight,
    )

    train_loader, valid_loader, test_loader = get_dataloader(
        train_batch_size=config.batch_size,
        valid_batch_size=config.batch_size,
        test_batch_size=1,
        base_path=config.data_root,
        num_workers=config.num_workers,
    )

    device = torch.device("cuda")
    num_gpus = torch.cuda.device_count()

    model = PGAFNO(in_chans=1, out_chans=len(TARGET_CHANNELS))
    if config.pretrained_checkpoint:
        logger.info("Loading pretrained checkpoint from %s", config.pretrained_checkpoint)
        model = load_checkpoint(model, config.pretrained_checkpoint, strict=False)

    model = model.to(device)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=1e-6,
    )

    best_valid_loss = float("inf")
    best_epoch = 0
    best_model_path = os.path.join(config.output_root, "best_model.pth")
    best_train_channel_loss = None
    best_valid_channel_loss = None
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_channel_loss_sum = torch.zeros(len(TARGET_CHANNELS), device=device)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}", leave=False)
        for inputs, targets in progress_bar:
            targets = targets[:, TARGET_CHANNELS, :, :]
            inputs = torch.nan_to_num(inputs.to(device).float())
            targets = torch.nan_to_num(targets.to(device).float())

            optimizer.zero_grad()
            outputs = model(inputs)

            loss_data = F.mse_loss(outputs, targets)
            loss_geo = geostrophic_physics_loss(outputs)
            loss = loss_data + config.geostrophic_weight * loss_geo

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss_sum += loss.item()
            batch_channel_loss = F.mse_loss(outputs, targets, reduction="none").mean(dim=(0, 2, 3))
            epoch_channel_loss_sum += batch_channel_loss

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                mse=f"{loss_data.item():.4f}",
                geo=f"{loss_geo.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        scheduler.step()

        train_loss = epoch_loss_sum / len(train_loader)
        train_channel_loss = (epoch_channel_loss_sum / len(train_loader)).detach().cpu().numpy()

        valid_loss, valid_channel_loss = evaluate(model, valid_loader, device)

        logger.info(
            "Epoch %d | train_loss=%.6f | valid_loss=%.6f",
            epoch + 1,
            train_loss,
            valid_loss,
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch + 1
            best_train_channel_loss = train_channel_loss
            best_valid_channel_loss = valid_channel_loss
            save_checkpoint(best_model_path, model, optimizer, best_epoch, best_valid_loss)
            patience_counter = 0
            logger.info("Saved new best model at epoch %d.", best_epoch)
        else:
            patience_counter += 1
            logger.info(
                "Validation did not improve. Early-stopping counter: %d/%d",
                patience_counter,
                config.patience_limit,
            )
            if patience_counter >= config.patience_limit:
                logger.info("Early stopping triggered.")
                break

        torch.cuda.empty_cache()

    logger.info("Loading best model for final evaluation from %s", best_model_path)
    best_model = PGAFNO(in_chans=1, out_chans=len(TARGET_CHANNELS))
    best_model = load_checkpoint(best_model, best_model_path, strict=True).to(device)
    if num_gpus > 1:
        best_model = torch.nn.DataParallel(best_model)

    test_loss, test_channel_loss = evaluate(best_model, test_loader, device)

    results_path = os.path.join(config.output_root, "results_all_69.txt")
    with open(results_path, "w", encoding="utf-8") as file:
        file.write("PG-AFNO results for the 1-to-69 reconstruction task\n")
        file.write("=" * 80 + "\n")
        file.write(f"Best epoch: {best_epoch}\n")
        file.write(f"Best validation MSE: {best_valid_loss:.6f}\n")
        file.write(f"Test MSE: {test_loss:.6f}\n")
        file.write("=" * 80 + "\n")
        file.write(f"{'Channel':<10}{'Train MSE':<18}{'Valid MSE':<18}{'Test MSE':<18}\n")
        file.write("-" * 80 + "\n")

        for index, channel in enumerate(TARGET_CHANNELS):
            file.write(
                f"{channel:<10}"
                f"{best_train_channel_loss[index]:<18.6f}"
                f"{best_valid_channel_loss[index]:<18.6f}"
                f"{test_channel_loss[index]:<18.6f}\n"
            )

        file.write("-" * 80 + "\n")
        file.write(
            f"{'Average':<10}"
            f"{float(np.mean(best_train_channel_loss)):<18.6f}"
            f"{float(np.mean(best_valid_channel_loss)):<18.6f}"
            f"{float(np.mean(test_channel_loss)):<18.6f}\n"
        )

    logger.info("Training finished. Results saved to %s", results_path)


if __name__ == "__main__":
    train(config)

import argparse
import os
from datetime import datetime
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader_1to69_pgafno import ERA5Dataset
from models_1to69_pgafno import PGAFNO

OMEGA = 7.2921e-5
H, W = 160, 160
DY = 27830.0
lat_array = np.linspace(50.0, 10.0, H)
lat_grid = lat_array.reshape(H, 1)
F_CORIOLIS_GRID = 2 * OMEGA * np.sin(np.deg2rad(lat_grid))
DX_GRID = 27830.0 * np.cos(np.deg2rad(lat_grid))

LARGE_SCALE_P_MAX = 30.0
CONVECTIVE_P_MIN = 50.0


MEAN_70 = np.array([
    1.16149490e+05, 1.16149490e+05, 1.16149490e+05, 5.80747448e+04, 5.80747448e+04, 5.80747448e+04,
    5.80747448e+04, 2.90373724e+04, 2.90373724e+04, 1.45186862e+04, 7.25939900e+03, 4.69109517e+03,
    9.93684855e+02, 1.13427236e+02, 1.13427236e+02, 1.13427236e+02, 1.13427236e+02, 1.13427236e+02,
    1.13428069e+02, 2.26854472e+02, 2.26854472e+02, 2.26854472e+02, 2.26854472e+02, 2.26854472e+02,
    2.26854472e+02, 2.26854472e+02, 1.88704771e+00, 1.24872370e+01, 1.98072294e+01, 2.16948522e+01,
    2.02551335e+01, 1.79678206e+01, 1.32261996e+01, 9.80464536e+00, 6.87079365e+00, 4.42695885e+00,
    1.01411563e+00, -2.37102261e-01, -6.60044487e-01, 3.58967726e-01, -7.59629959e-01, -1.68213408e-01,
    1.23268913e-01, -8.14977763e-02, -3.56274342e-01, -5.70230058e-01, -5.44994384e-01, -4.10185719e-01,
   -2.67485750e-01, -8.80510577e-02, -3.05537064e-01, -5.34357575e-01, 2.73344239e-06, 2.80390530e-06,
    8.31144530e-06, 3.78979373e-05, 1.14464238e-04, 2.51176913e-04, 7.49817375e-04, 1.59744166e-03,
    2.79969897e-03, 4.39261614e-03, 7.38884209e-03, 9.00895154e-03, 1.03725488e-02, 2.26854472e+02,
   -5.44162220e-01, -4.87323450e-01, 5.80747448e+04, 0.348146950
], dtype=np.float64)
STD_70 = np.array([
    1.30219768e+05, 4.16322227e+04, 4.16322227e+04, 6.51098838e+04, 6.51098838e+04, 6.51098838e+04,
    2.08161114e+04, 3.25549419e+04, 1.04080557e+04, 1.62774709e+04, 8.13874188e+03, 4.05253958e+03,
    7.16941681e+02, 1.27167742e+02, 1.27167742e+02, 1.27167742e+02, 1.27167742e+02, 1.27167742e+02,
    1.27167837e+02, 8.13129350e+01, 8.13129350e+01, 8.13129350e+01, 8.13129350e+01, 8.13129350e+01,
    8.13129350e+01, 8.13129350e+01, 1.24746978e+01, 1.75673706e+01, 2.02472881e+01, 2.13500860e+01,
    2.01455563e+01, 1.79424873e+01, 1.37866633e+01, 1.06490084e+01, 8.61709957e+00, 7.04418871e+00,
    5.93161139e+00, 5.49941563e+00, 4.22523544e+00, 3.79186460e+00, 6.40160727e+00, 9.31701532e+00,
    1.04910052e+01, 1.05093977e+01, 9.82275919e+00, 8.14054059e+00, 6.86402592e+00, 6.01023122e+00,
    5.44844223e+00, 5.04471797e+00, 4.97708382e+00, 4.15662788e+00, 1.55335822e-07, 7.57437119e-07,
    6.21876557e-06, 3.73429164e-05, 1.21325833e-04, 2.74843753e-04, 7.97787208e-04, 1.57233155e-03,
    2.35437480e-03, 3.21320856e-03, 4.67745314e-03, 5.54012691e-03, 6.48038901e-03, 8.13129350e+01,
    3.78661841e+00, 3.70222423e+00, 6.51098838e+04, 0.617549492
], dtype=np.float64)


def denorm_69(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN_70[:69], device=tensor.device, dtype=torch.float32).view(1, 69, 1, 1)
    std = torch.tensor(STD_70[:69], device=tensor.device, dtype=torch.float32).view(1, 69, 1, 1)
    return tensor * std + mean

def denorm_precip(tensor: torch.Tensor) -> torch.Tensor:
    mean, std = MEAN_70[69], STD_70[69]
    return torch.clamp(torch.expm1(tensor * std + mean), min=0.0)

def norm_precip(tensor_mm: torch.Tensor) -> torch.Tensor:
    mean, std = MEAN_70[69], STD_70[69]
    return (torch.log1p(tensor_mm) - mean) / std

def load_model(checkpoint_path: str, device: torch.device) -> PGAFNO:
    model = PGAFNO(in_chans=1, out_chans=69).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    clean_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=True)
    model.eval()
    return model

def log_message(message: str, log_path: str) -> None:
    print(message)
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(message + "\n")

def task_geostrophic(model, loader, device, out_dir, log_path):
    log_message("Running geostrophic check", log_path)
    u_corrs, v_corrs = [], []
    f_grid = torch.tensor(F_CORIOLIS_GRID, device=device, dtype=torch.float32).view(1, H, 1)
    dx_grid = torch.tensor(DX_GRID, device=device, dtype=torch.float32).view(1, H, 1)
    valid_mask = (lat_array > 25.0)[2:-2]

    with torch.no_grad():
        for ft, _ in tqdm(islice(loader, 50), total=50, desc="Geostrophic"):
            ft = ft.to(device).float()
            pred_phys = denorm_69(model(ft))
            phi = pred_phys[:, 7]
            u_actual = pred_phys[:, 33]
            v_actual = pred_phys[:, 46]
            grad_y, grad_x = torch.gradient(phi, dim=(1, 2), spacing=(DY, 1.0))
            dphi_dy = grad_y / (-1.0)
            dphi_dx = grad_x / dx_grid
            f_safe = torch.clamp(f_grid, min=1e-5)
            u_geo = -1.0 * dphi_dy / f_safe
            v_geo = dphi_dx / f_safe

            for i in range(phi.shape[0]):
                ug = u_geo[i, 2:-2, 2:-2].cpu().numpy()[valid_mask, :].flatten()
                ua = u_actual[i, 2:-2, 2:-2].cpu().numpy()[valid_mask, :].flatten()
                vg = v_geo[i, 2:-2, 2:-2].cpu().numpy()[valid_mask, :].flatten()
                va = v_actual[i, 2:-2, 2:-2].cpu().numpy()[valid_mask, :].flatten()
                if np.std(ug) > 1e-3 and np.std(ua) > 1e-3:
                    u_corrs.append(pearsonr(ug, ua)[0])
                if np.std(vg) > 1e-3 and np.std(va) > 1e-3:
                    v_corrs.append(pearsonr(vg, va)[0])

    log_message(f"500-hPa geostrophic U correlation: {np.nanmean(u_corrs):.4f}", log_path)
    log_message(f"500-hPa geostrophic V correlation: {np.nanmean(v_corrs):.4f}", log_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].scatter(ug, ua, s=2, alpha=0.3)
    axes[0].set_title(f"U correlation = {np.nanmean(u_corrs):.3f}")
    axes[1].scatter(vg, va, s=2, alpha=0.3)
    axes[1].set_title(f"V correlation = {np.nanmean(v_corrs):.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "geostrophic_scatter.png"))
    plt.close(fig)

def task_hydrostatic(model, loader, device, log_path):
    log_message("Running hydrostatic check", log_path)
    correlations = []
    with torch.no_grad():
        for ft, _ in tqdm(islice(loader, 50), total=50, desc="Hydrostatic"):
            pred_phys = denorm_69(model(ft.to(device).float()))
            thickness = pred_phys[:, 7] - pred_phys[:, 10]
            mean_temperature = (pred_phys[:, 20] + pred_phys[:, 23]) / 2.0
            for i in range(thickness.shape[0]):
                r, _ = pearsonr(thickness[i].cpu().numpy().flatten(), mean_temperature[i].cpu().numpy().flatten())
                if not np.isnan(r):
                    correlations.append(r)
    log_message(f"500–850-hPa thickness–temperature correlation: {np.nanmean(correlations):.4f}", log_path)

def task_regime(model, loader, device, log_path):
    log_message("Running regime-stratified evaluation", log_path)
    large_scale_errors, convective_errors = [], []
    with torch.no_grad():
        for ft, gt in tqdm(islice(loader, 100), total=100, desc="Regime"):
            ft = ft.to(device).float()
            gt = gt.to(device).float()
            pred = model(ft)
            precip_mm = denorm_precip(ft).squeeze(1)
            for i in range(precip_mm.shape[0]):
                p_max = precip_mm[i].max().item()
                mse_v10 = F.mse_loss(pred[i, 67], gt[i, 67]).item()
                if 1.0 < p_max <= LARGE_SCALE_P_MAX:
                    large_scale_errors.append(mse_v10)
                elif p_max > CONVECTIVE_P_MIN:
                    convective_errors.append(mse_v10)
    log_message(f"Large-scale regime V10 MSE: {np.mean(large_scale_errors):.4f} (N={len(large_scale_errors)})", log_path)
    log_message(f"Convective regime V10 MSE: {np.mean(convective_errors):.4f} (N={len(convective_errors)})", log_path)

def task_moisture(model, loader, device, log_path):
    log_message("Running precipitation–humidity correlation check", log_path)
    correlations = []
    with torch.no_grad():
        for ft, _ in tqdm(islice(loader, 50), total=50, desc="Moisture"):
            ft = ft.to(device).float()
            pred_phys = denorm_69(model(ft))
            precip_mm = denorm_precip(ft).squeeze(1)
            q850 = pred_phys[:, 62]
            for i in range(precip_mm.shape[0]):
                mask = precip_mm[i] > 0.5
                if mask.sum() > 50:
                    r, _ = pearsonr(
                        precip_mm[i][mask].cpu().numpy().flatten(),
                        q850[i][mask].cpu().numpy().flatten(),
                    )
                    if not np.isnan(r):
                        correlations.append(r)
    log_message(f"850-hPa humidity–precipitation correlation: {np.nanmean(correlations):.4f}", log_path)

def task_stability(model, loader, device, log_path):
    log_message("Running 1% perturbation stability test", log_path)
    changes = []
    with torch.no_grad():
        for ft, _ in tqdm(islice(loader, 20), total=20, desc="Stability"):
            ft = ft.to(device).float()
            out_ref = model(ft)
            precip_mm = denorm_precip(ft)
            noise = torch.randn_like(precip_mm) * (precip_mm.std() * 0.01)
            ft_noisy = norm_precip(torch.clamp(precip_mm + noise, min=0.0))
            out_noisy = model(ft_noisy)
            changes.append((torch.sqrt(F.mse_loss(out_ref, out_noisy)) / out_ref.std()).item() * 100.0)
    log_message(f"Mean output change under 1% perturbation: {np.mean(changes):.2f}%", log_path)

def task_zero_test(model, loader, device, out_dir, log_path):
    log_message("Running zero-precipitation climatology test", log_path)
    spatial_mean_accum = torch.zeros((69, H, W), device=device)
    sample_count = 0
    for _, gt in tqdm(loader, desc="Climatology"):
        gt = gt.to(device).float()
        spatial_mean_accum += gt.sum(dim=0)
        sample_count += gt.shape[0]

    spatial_clim_norm = spatial_mean_accum / sample_count
    spatial_clim_phys = denorm_69(spatial_clim_norm.unsqueeze(0))[0].cpu().numpy()

    p0 = torch.zeros((1, 1, H, W), device=device)
    p001 = torch.full((1, 1, H, W), 0.001, device=device)

    with torch.no_grad():
        out0_phys = denorm_69(model(norm_precip(p0)))[0].cpu().numpy()
        out001_phys = denorm_69(model(norm_precip(p001)))[0].cpu().numpy()

    bias = out0_phys - spatial_clim_phys
    sensitivity = out001_phys - out0_phys

    zero_dir = os.path.join(out_dir, "zero_test")
    os.makedirs(zero_dir, exist_ok=True)
    log_message("Channel | Mean(P=0) | Clim Mean | Mean |Bias| | Mean |Sensitivity|", log_path)
    log_message("-" * 72, log_path)

    for channel in tqdm(range(69), desc="Zero test plots"):
        channel_map = out0_phys[channel]
        clim_map = spatial_clim_phys[channel]
        channel_bias = bias[channel]
        channel_sens = sensitivity[channel]
        log_message(
            f"{channel:>7d} | {channel_map.mean():>9.4f} | {clim_map.mean():>9.4f} | "
            f"{np.abs(channel_bias).mean():>10.4f} | {np.abs(channel_sens).mean():>16.6f}",
            log_path,
        )

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        axes[0].imshow(channel_map, cmap="coolwarm")
        axes[0].set_title(f"Ch {channel} output at P=0")
        axes[1].imshow(clim_map, cmap="coolwarm")
        axes[1].set_title(f"Ch {channel} spatial climatology")
        value_range = np.max(clim_map) - np.min(clim_map)
        axes[2].imshow(channel_bias, cmap="bwr", vmin=-value_range / 2, vmax=value_range / 2)
        axes[2].set_title("Bias")
        axes[3].imshow(channel_sens, cmap="seismic")
        axes[3].set_title("Sensitivity")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(zero_dir, f"zero_test_channel_{channel}.png"), dpi=120)
        plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["geostrophic", "hydrostatic", "regime", "moisture", "stability", "zero_test", "all"])
    parser.add_argument("--ckpt", type=str, default="path/to/checkpoint.pth")
    parser.add_argument("--base_path", type=str, default="path/to/normalized_era5")
    parser.add_argument("--out_dir", type=str, default="path/to/output/physical_validation")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = os.path.join(args.out_dir, "validation_summary.txt")
    log_message(f"Physical validation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_path)

    model = load_model(args.ckpt, device)
    dataset = ERA5Dataset(base_path=args.base_path, split="test")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.task in ["geostrophic", "all"]:
        task_geostrophic(model, loader, device, args.out_dir, log_path)
    if args.task in ["hydrostatic", "all"]:
        task_hydrostatic(model, loader, device, log_path)
    if args.task in ["regime", "all"]:
        task_regime(model, loader, device, log_path)
    if args.task in ["moisture", "all"]:
        task_moisture(model, loader, device, log_path)
    if args.task in ["stability", "all"]:
        task_stability(model, loader, device, log_path)
    if args.task in ["zero_test", "all"]:
        task_zero_test(model, loader, device, args.out_dir, log_path)

    print(f"Saved outputs to {args.out_dir}")

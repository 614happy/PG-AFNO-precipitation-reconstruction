import argparse
import glob
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from dataloader_1to69_pgafno import ERA5Dataset
from models_1to69_pgafno import PGAFNO

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
})

H, W = 160, 160
LONS = np.linspace(100.0, 140.0, W)
LATS = np.linspace(50.0, 10.0, H)
PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


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

def load_model(checkpoint_path: str, device: torch.device) -> PGAFNO:
    model = PGAFNO(in_chans=1, out_chans=69).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    clean_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=True)
    model.eval()
    return model

def get_case_indices(case_dir: str) -> list[int]:
    if not os.path.exists(case_dir):
        return []
    indices = []
    for file_path in glob.glob(os.path.join(case_dir, "*.png")):
        match = re.search(r"idx(\d+)\.png", file_path)
        if match:
            indices.append(int(match.group(1)))
    return sorted(set(indices))

def plot_synoptic_panels(model: PGAFNO, dataset: ERA5Dataset, indices: list[int], out_dir: str, device: torch.device) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        for idx in tqdm(indices, desc="Predicted synoptic panels"):
            ft, _ = dataset[idx]
            ft = ft.unsqueeze(0).to(device).float()
            pred = denorm_69(model(ft))[0].cpu().numpy()
            precip = denorm_precip(ft)[0, 0].cpu().numpy()

            z500 = pred[7] / 9.80665
            u500, v500 = pred[33], pred[46]
            q850 = pred[62] * 1000.0
            u850, v850 = pred[36], pred[49]
            mslp = pred[68] / 100.0
            u10, v10 = pred[66], pred[67]

            fig = plt.figure(figsize=(14, 10))
            skip = 4

            ax1 = plt.subplot(2, 2, 1)
            im1 = ax1.imshow(precip, cmap="Blues", extent=[100, 140, 10, 50], origin="upper")
            ax1.set_title("Input: 6h accumulated precipitation")
            plt.colorbar(im1, ax=ax1, label="Precipitation (mm)")

            ax2 = plt.subplot(2, 2, 2)
            cs2 = ax2.contour(LONS, LATS, z500, levels=15, colors="black", linewidths=0.8)
            ax2.clabel(cs2, inline=True, fontsize=8, fmt="%1.0f")
            ax2.quiver(LONS[::skip], LATS[::skip], u500[::skip, ::skip], v500[::skip, ::skip], color="blue", alpha=0.6, scale=400)
            ax2.set_title("Predicted: 500-hPa geopotential height and wind")

            ax3 = plt.subplot(2, 2, 3)
            im3 = ax3.contourf(LONS, LATS, q850, levels=20, cmap="YlGnBu")
            plt.colorbar(im3, ax=ax3, label="Specific humidity (g kg$^-1$)")
            ax3.quiver(LONS[::skip], LATS[::skip], u850[::skip, ::skip], v850[::skip, ::skip], color="black", alpha=0.7, scale=300)
            ax3.set_title("Predicted: 850-hPa moisture transport")

            ax4 = plt.subplot(2, 2, 4)
            im4 = ax4.contourf(LONS, LATS, mslp, levels=20, cmap="RdYlBu_r")
            plt.colorbar(im4, ax=ax4, label="Mean sea level pressure (hPa)")
            ax4.quiver(LONS[::skip], LATS[::skip], u10[::skip, ::skip], v10[::skip, ::skip], color="black", scale=200)
            ax4.set_title("Predicted: surface MSLP and 10-m wind")

            for ax in [ax1, ax2, ax3, ax4]:
                ax.set_xlabel("Longitude (°E)")
                ax.set_ylabel("Latitude (°N)")

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"predicted_synoptic_idx{idx}.png"))
            plt.close(fig)

def plot_vertical_sections(model: PGAFNO, dataset: ERA5Dataset, case_indices: list[int], out_dir: str, device: torch.device, include_random: int = 5) -> None:
    os.makedirs(out_dir, exist_ok=True)
    random_indices = random.sample(range(len(dataset)), min(include_random, len(dataset)))
    all_indices = sorted(set(case_indices + random_indices))
    with torch.no_grad():
        for idx in tqdm(all_indices, desc="Predicted vertical sections"):
            ft, _ = dataset[idx]
            ft = ft.unsqueeze(0).to(device).float()
            pred = denorm_69(model(ft))[0].cpu().numpy()
            precip = denorm_precip(ft)[0, 0].cpu().numpy()

            max_y_idx = np.unravel_index(np.argmax(precip, axis=None), precip.shape)[0]
            slice_lat = LATS[max_y_idx]
            temperature_profile = pred[13:26, max_y_idx, :]
            geopotential_profile = pred[0:13, max_y_idx, :] / 9.80665

            fig, ax = plt.subplots(figsize=(10, 6))
            cf = ax.contourf(LONS, PRESSURE_LEVELS, temperature_profile, levels=20, cmap="RdBu_r")
            plt.colorbar(cf, label="Temperature (K)")
            cs = ax.contour(LONS, PRESSURE_LEVELS, geopotential_profile, levels=10, colors="black", linewidths=1.0)
            ax.clabel(cs, inline=True, fontsize=9, fmt="%1.0f")
            ax.invert_yaxis()
            ax.set_yscale("log")
            ax.set_yticks([1000, 850, 700, 500, 300, 200, 100, 50])
            ax.set_yticklabels(["1000", "850", "700", "500", "300", "200", "100", "50"])
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Pressure (hPa)")
            prefix = "random" if idx in random_indices else "case"
            ax.set_title(f"Predicted vertical profile at {slice_lat:.1f}°N ({prefix}, idx={idx})")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"predicted_vertical_idx{idx}.png"))
            plt.close(fig)

def plot_saliency(model: PGAFNO, dataset: ERA5Dataset, indices: list[int], out_dir: str, device: torch.device) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    for idx in tqdm(indices, desc="Saliency maps"):
        ft, _ = dataset[idx]
        ft = ft.unsqueeze(0).to(device).float()
        ft.requires_grad = True
        pred = model(ft)
        objective = pred[0, 7, 60:100, 60:100].mean()
        model.zero_grad()
        objective.backward()
        saliency = ft.grad.abs()[0, 0].cpu().numpy()
        saliency = gaussian_filter(saliency, sigma=1)
        precip = denorm_precip(ft.detach())[0, 0].cpu().numpy()
        z500 = denorm_69(pred.detach())[0, 7].cpu().numpy() / 9.80665

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        im0 = axes[0].imshow(precip, cmap="Blues", extent=[100, 140, 10, 50], origin="upper")
        axes[0].set_title("Input precipitation")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].contourf(LONS, LATS, z500, levels=20, cmap="viridis")
        rect = plt.Rectangle((100 + 60 * 0.25, 50 - 100 * 0.25), 40 * 0.25, 40 * 0.25, fill=False, color="red", lw=2)
        axes[1].add_patch(rect)
        axes[1].set_title("Predicted 500-hPa geopotential")
        plt.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(saliency, cmap="hot", extent=[100, 140, 10, 50], origin="upper")
        axes[2].set_title("Saliency map")
        plt.colorbar(im2, ax=axes[2])

        for ax in axes:
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"saliency_idx{idx}.png"))
        plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="all", choices=["synoptic", "vertical", "saliency", "all"])
    parser.add_argument("--ckpt", type=str, default="path/to/checkpoint.pth")
    parser.add_argument("--base_path", type=str, default="path/to/normalized_era5")
    parser.add_argument("--case_dir", type=str, default="path/to/case_selection")
    parser.add_argument("--out_dir", type=str, default="path/to/output/figure3")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    dataset = ERA5Dataset(base_path=args.base_path, split="test")

    case_indices = get_case_indices(args.case_dir)
    if not case_indices:
        raise FileNotFoundError("No case indices were found in the case directory.")

    if args.task in ["synoptic", "all"]:
        plot_synoptic_panels(model, dataset, case_indices, os.path.join(args.out_dir, "predicted_synoptic"), device)
    if args.task in ["vertical", "all"]:
        plot_vertical_sections(model, dataset, case_indices, os.path.join(args.out_dir, "predicted_vertical"), device)
    if args.task in ["saliency", "all"]:
        plot_saliency(model, dataset, case_indices, os.path.join(args.out_dir, "saliency"), device)

    print(f"Saved outputs to {args.out_dir}")

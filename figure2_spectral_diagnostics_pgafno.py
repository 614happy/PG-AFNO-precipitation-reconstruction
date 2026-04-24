import argparse
import os
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
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
    "legend.fontsize": 10,
    "figure.dpi": 300,
})

H, W = 160, 160
DX_KM = 25.0


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


def denorm(tensor: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return tensor * std + mean

def load_model(checkpoint_path: str, device: torch.device) -> PGAFNO:
    model = PGAFNO(in_chans=1, out_chans=69).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    clean_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=True)
    model.eval()
    return model

def get_radial_spectrum(u: np.ndarray, v: np.ndarray):
    window = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    u_windowed = (u - np.mean(u)) * window
    v_windowed = (v - np.mean(v)) * window

    u_fft = np.fft.fft2(u_windowed)
    v_fft = np.fft.fft2(v_windowed)
    kinetic_energy_2d = 0.5 * (np.abs(u_fft) ** 2 + np.abs(v_fft) ** 2)
    kinetic_energy_2d = np.fft.fftshift(kinetic_energy_2d)

    freq_x = np.fft.fftshift(np.fft.fftfreq(W, d=DX_KM))
    freq_y = np.fft.fftshift(np.fft.fftfreq(H, d=DX_KM))
    kx, ky = np.meshgrid(freq_x, freq_y)
    k_radial = np.sqrt(kx ** 2 + ky ** 2)

    k_min = 1.0 / (W * DX_KM)
    k_max = np.max(k_radial)
    k_bins = np.logspace(np.log10(k_min), np.log10(k_max), 50)
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])

    spectrum = np.zeros(len(k_centers))
    for i in range(len(k_centers)):
        mask = (k_radial >= k_bins[i]) & (k_radial < k_bins[i + 1])
        if np.sum(mask) > 0:
            dk = k_bins[i + 1] - k_bins[i]
            spectrum[i] = np.sum(kinetic_energy_2d[mask]) / dk

    valid = (k_centers > 0) & (spectrum > 0)
    return k_centers[valid], spectrum[valid]

def plot_threshold_curve(save_path: str) -> None:
    k = np.linspace(0, 100, 500)
    base_threshold, k_decay, min_threshold = 0.01, 20.0, 0.0005
    threshold_adaptive = np.maximum(base_threshold * np.exp(-0.5 * (k / k_decay) ** 2), min_threshold)
    threshold_constant = np.full_like(k, base_threshold)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(k, threshold_constant, color="gray", linestyle="--", linewidth=2, label="Original AFNO")
    ax.plot(k, threshold_adaptive, color="#1f77b4", linestyle="-", linewidth=2.5, label="Adaptive threshold")
    ax.fill_between(k, threshold_adaptive, threshold_constant, where=(k > 10), color="lightblue", alpha=0.3)
    ax.set_xlabel("Absolute wavenumber k")
    ax.set_ylabel("Soft-shrink threshold")
    ax.set_title("Asymmetric spectral thresholding")
    ax.set_ylim(0, 0.011)
    ax.set_xlim(0, 100)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend()
    plt.tight_layout(pad=1.5)
    plt.savefig(save_path)
    plt.close(fig)

def analyze_spectra(model: PGAFNO, loader: DataLoader, device: torch.device, save_path: str) -> None:
    model.eval()
    accumulated_pred = None
    accumulated_gt = None
    all_k = None
    sample_count = 0

    with torch.no_grad():
        for ft, gt in tqdm(islice(loader, 100), total=100, desc="Spectra"):
            ft = ft.to(device).float()
            gt = gt.to(device).float()
            pred = model(ft)

            u_pred = denorm(pred[0, 33], MEAN_70[33], STD_70[33]).cpu().numpy()
            v_pred = denorm(pred[0, 46], MEAN_70[46], STD_70[46]).cpu().numpy()
            u_gt = denorm(gt[0, 33], MEAN_70[33], STD_70[33]).cpu().numpy()
            v_gt = denorm(gt[0, 46], MEAN_70[46], STD_70[46]).cpu().numpy()

            k_centers, spectrum_pred = get_radial_spectrum(u_pred, v_pred)
            _, spectrum_gt = get_radial_spectrum(u_gt, v_gt)

            if accumulated_pred is None:
                all_k = k_centers
                accumulated_pred = spectrum_pred
                accumulated_gt = spectrum_gt
            else:
                accumulated_pred += spectrum_pred
                accumulated_gt += spectrum_gt
            sample_count += 1

    mean_pred = accumulated_pred / sample_count
    mean_gt = accumulated_gt / sample_count

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.loglog(all_k, mean_gt, "k-", linewidth=2, label="ERA5")
    ax1.loglog(all_k, mean_pred, "b--", linewidth=2, label="PG-AFNO")

    ref_idx = len(all_k) // 4
    k_ref = all_k[ref_idx:]
    energy_ref = mean_gt[ref_idx] * (k_ref / all_k[ref_idx]) ** (-3.0)
    ax1.loglog(k_ref, energy_ref, "r:", linewidth=2, label=r"Theoretical $k^{-3}$")

    ax1.set_xlabel(r"Wavenumber $k$ (km$^{-1}$)")
    ax1.set_ylabel(r"Kinetic energy density $E(k)$")
    ax1.set_title("500-hPa kinetic energy spectra")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left")

    ax2 = ax1.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ax1.get_xlim())
    wavelengths = [2000, 1000, 500, 200, 100]
    ax2.set_xticks([1 / w for w in wavelengths])
    ax2.set_xticklabels([str(w) for w in wavelengths])
    ax2.set_xlabel("Wavelength (km)")

    plt.tight_layout(pad=1.5)
    plt.savefig(save_path)
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="path/to/checkpoint.pth")
    parser.add_argument("--base_path", type=str, default="path/to/normalized_era5")
    parser.add_argument("--out_dir", type=str, default="path/to/output/figure2")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    threshold_path = os.path.join(args.out_dir, "figure2a_threshold.png")
    spectra_path = os.path.join(args.out_dir, "figure2b_kinetic_energy_spectrum.png")
    plot_threshold_curve(threshold_path)

    dataset = ERA5Dataset(base_path=args.base_path, split="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    model = load_model(args.ckpt, device)
    analyze_spectra(model, loader, device, spectra_path)
    print(f"Saved outputs to {args.out_dir}")

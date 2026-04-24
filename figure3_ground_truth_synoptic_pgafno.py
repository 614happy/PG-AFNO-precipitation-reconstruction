import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
})

H, W = 160, 160
LONS = np.linspace(100.0, 140.0, W)
LATS = np.linspace(50.0, 10.0, H)
LON_GRID, LAT_GRID = np.meshgrid(LONS, LATS)


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


def denorm_69(array: np.ndarray) -> np.ndarray:
    mean = MEAN_70[:69].reshape(69, 1, 1)
    std = STD_70[:69].reshape(69, 1, 1)
    return array * std + mean

def denorm_precip(array: np.ndarray) -> np.ndarray:
    mean, std = MEAN_70[69], STD_70[69]
    return np.clip(np.expm1(array * std + mean), a_min=0.0, a_max=None)

class ERA5TestOnlyDataset(Dataset):
    def __init__(self, base_path: str = "path/to/normalized_era5") -> None:
        super().__init__()
        data_list = []
        for year in [2017, 2018]:
            file_path = os.path.join(base_path, f"{year}.npy")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")
            data_list.append(np.load(file_path))
            print(f"Loaded {year}.npy")
        self.data = np.concatenate(data_list, axis=0)

    def __getitem__(self, idx: int):
        precip = self.data[idx, 69, :160, :160]
        target = self.data[idx, :69, :160, :160]
        return np.nan_to_num(precip, nan=0.0), np.nan_to_num(target, nan=0.0)

    def __len__(self) -> int:
        return self.data.shape[0]

def plot_ground_truth_panels(idx: int, precip_norm: np.ndarray, gt_norm: np.ndarray, out_dir: str) -> None:
    precip = denorm_precip(precip_norm)
    gt_phys = denorm_69(gt_norm)

    z500 = gt_phys[7] / 9.80665
    u500, v500 = gt_phys[33], gt_phys[46]
    q850 = gt_phys[62] * 1000.0
    u850, v850 = gt_phys[36], gt_phys[49]
    mslp = gt_phys[68] / 100.0
    u10, v10 = gt_phys[66], gt_phys[67]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    extent = [100, 140, 10, 50]
    skip = 4

    ax = axes[0, 0]
    im = ax.imshow(precip, cmap="Blues", extent=extent, origin="upper", vmin=0)
    ax.set_title("a. Ground Truth: 6h accumulated precipitation")
    plt.colorbar(im, ax=ax, label="Precipitation (mm)")

    ax = axes[0, 1]
    cs = ax.contour(LON_GRID, LAT_GRID, z500, levels=15, colors="black", linewidths=0.8, alpha=0.7)
    ax.clabel(cs, inline=True, fmt="%1.0f", fontsize=8)
    ax.quiver(LON_GRID[::skip, ::skip], LAT_GRID[::skip, ::skip], u500[::skip, ::skip], v500[::skip, ::skip],
              color="blue", alpha=0.6, scale=400)
    ax.set_title("b. Ground Truth: 500-hPa geopotential height and wind")

    ax = axes[1, 0]
    im = ax.contourf(LON_GRID, LAT_GRID, q850, levels=20, cmap="YlGnBu")
    plt.colorbar(im, ax=ax, label="Specific humidity (g kg$^-1$)")
    ax.quiver(LON_GRID[::skip, ::skip], LAT_GRID[::skip, ::skip], u850[::skip, ::skip], v850[::skip, ::skip],
              color="black", alpha=0.7, scale=300)
    ax.set_title("c. Ground Truth: 850-hPa moisture transport")

    ax = axes[1, 1]
    im = ax.contourf(LON_GRID, LAT_GRID, mslp, levels=20, cmap="RdYlBu_r")
    plt.colorbar(im, ax=ax, label="Mean sea level pressure (hPa)")
    ax.quiver(LON_GRID[::skip, ::skip], LAT_GRID[::skip, ::skip], u10[::skip, ::skip], v10[::skip, ::skip],
              color="black", alpha=0.8, scale=200)
    ax.set_title("d. Ground Truth: surface MSLP and 10-m wind")

    for ax in axes.flat:
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"ground_truth_synoptic_idx{idx}.png"))
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="path/to/normalized_era5")
    parser.add_argument("--out_dir", type=str, default="path/to/output/figure3_ground_truth_synoptic")
    parser.add_argument("--indices", type=int, nargs="+", default=[1027, 1028, 1029])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dataset = ERA5TestOnlyDataset(base_path=args.base_path)
    for idx in tqdm(args.indices, desc="Ground truth panels"):
        if idx >= len(dataset):
            continue
        precip_norm, gt_norm = dataset[idx]
        plot_ground_truth_panels(idx, precip_norm, gt_norm, args.out_dir)
    print(f"Saved outputs to {args.out_dir}")

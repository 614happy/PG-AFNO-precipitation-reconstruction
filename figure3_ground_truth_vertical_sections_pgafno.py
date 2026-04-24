import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

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
    mean = torch.tensor(MEAN_70[:69], dtype=torch.float32).view(1, 69, 1, 1)
    std = torch.tensor(STD_70[:69], dtype=torch.float32).view(1, 69, 1, 1)
    return tensor * std + mean

def denorm_precip(tensor: torch.Tensor) -> torch.Tensor:
    mean, std = MEAN_70[69], STD_70[69]
    return torch.clamp(torch.expm1(tensor * std + mean), min=0.0)

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
        input_tensor = torch.nan_to_num(torch.from_numpy(precip).float().unsqueeze(0), nan=0.0)
        target_tensor = torch.nan_to_num(torch.from_numpy(target).float(), nan=0.0)
        return input_tensor, target_tensor

    def __len__(self) -> int:
        return self.data.shape[0]

def plot_ground_truth_vertical_section(dataset: ERA5TestOnlyDataset, indices: list[int], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for idx in tqdm(indices, desc="Vertical sections"):
        ft, gt = dataset[idx]
        ft = ft.unsqueeze(0)
        gt = gt.unsqueeze(0)

        gt_phys = denorm_69(gt)[0].numpy()
        precip = denorm_precip(ft)[0, 0].numpy()

        max_y_idx = np.unravel_index(np.argmax(precip, axis=None), precip.shape)[0]
        slice_lat = LATS[max_y_idx]

        temperature_profile = gt_phys[13:26, max_y_idx, :]
        geopotential_profile = gt_phys[0:13, max_y_idx, :] / 9.80665

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
        ax.set_title(f"Ground Truth Vertical Profile at {slice_lat:.1f}°N (idx={idx})")

        save_path = os.path.join(out_dir, f"ground_truth_vertical_idx{idx}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="path/to/normalized_era5")
    parser.add_argument("--out_dir", type=str, default="path/to/output/figure3_ground_truth_vertical")
    parser.add_argument("--indices", type=int, nargs="+", default=[1027, 1028, 1029])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dataset = ERA5TestOnlyDataset(base_path=args.base_path)
    plot_ground_truth_vertical_section(dataset, args.indices, args.out_dir)
    print(f"Saved outputs to {args.out_dir}")

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300,
})

def plot_threshold_curve(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    k = np.linspace(0, 100, 500)
    base_threshold = 0.01
    k_decay = 20.0
    min_threshold = 0.0005

    threshold_adaptive = np.maximum(base_threshold * np.exp(-0.5 * (k / k_decay) ** 2), min_threshold)
    threshold_constant = np.full_like(k, base_threshold)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k, threshold_constant, color="gray", linestyle="--", linewidth=2.5, label="Original AFNO")
    ax.plot(k, threshold_adaptive, color="#1f77b4", linestyle="-", linewidth=3.0, label="Adaptive threshold")
    ax.fill_between(k, threshold_adaptive, threshold_constant, where=(k > 10), color="lightblue", alpha=0.3)
    ax.set_xlabel("Absolute wavenumber k")
    ax.set_ylabel("Soft-shrink threshold")
    ax.set_title("Asymmetric spectral thresholding")
    ax.set_ylim(0, 0.011)
    ax.set_xlim(0, 100)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="upper right")
    save_path = os.path.join(out_dir, "figure2a_threshold.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="path/to/output/figure2")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    plot_threshold_curve(args.out_dir)

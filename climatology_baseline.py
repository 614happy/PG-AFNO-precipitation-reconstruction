import os
from datetime import datetime, timedelta

import numpy as np
from tqdm import tqdm

BASE_PATH = "path/to/normalized_era5"
OUTPUT_TXT = "path/to/output/climatology_baseline_mse.txt"

TRAIN_YEARS = [2011, 2012, 2013, 2014, 2015]
VALID_YEARS = [2016]
TEST_YEARS = [2017, 2018]

NUM_CHANNELS = 69
HEIGHT = 160
WIDTH = 160
SMOOTH_WINDOW = 61


def get_time_mapping(year: int):
    """Return month, day-of-year, and 6-hour slot indices for a full year."""
    start = datetime(year, 1, 1, 0, 0)
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    days = 366 if is_leap else 365
    total_steps = days * 4

    months = np.zeros(total_steps, dtype=np.int32)
    doys = np.zeros(total_steps, dtype=np.int32)
    tods = np.zeros(total_steps, dtype=np.int32)

    for idx in range(total_steps):
        current = start + timedelta(hours=6 * idx)
        months[idx] = current.month - 1
        doys[idx] = current.timetuple().tm_yday - 1
        tods[idx] = current.hour // 6

    return months, doys, tods


def linear_decay_weights(window: int) -> np.ndarray:
    half = window // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    weights = 1.0 - np.abs(offsets) / float(half + 1)
    weights /= weights.sum()
    return weights


def load_year(path: str, year: int) -> np.ndarray:
    file_path = os.path.join(path, f"{year}.npy")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing data file: {file_path}")
    return np.load(file_path)


def compute_climatologies(base_path: str = BASE_PATH):
    """Compute monthly and smoothed day-of-year climatologies from training years."""
    monthly_sum = np.zeros((12, 4, NUM_CHANNELS, HEIGHT, WIDTH), dtype=np.float64)
    monthly_count = np.zeros((12, 4), dtype=np.float64)

    doy_sum = np.zeros((366, 4, NUM_CHANNELS, HEIGHT, WIDTH), dtype=np.float64)
    doy_count = np.zeros((366, 4), dtype=np.float64)

    for year in TRAIN_YEARS:
        data = load_year(base_path, year)
        months, doys, tods = get_time_mapping(year)

        for month in range(12):
            for tod in range(4):
                index = (months == month) & (tods == tod)
                if np.any(index):
                    subset = np.nan_to_num(data[index, :NUM_CHANNELS, :HEIGHT, :WIDTH])
                    monthly_sum[month, tod] += subset.sum(axis=0)
                    monthly_count[month, tod] += np.sum(index)

        for doy in range(366):
            for tod in range(4):
                index = (doys == doy) & (tods == tod)
                if np.any(index):
                    subset = np.nan_to_num(data[index, :NUM_CHANNELS, :HEIGHT, :WIDTH])
                    doy_sum[doy, tod] += subset.sum(axis=0)
                    doy_count[doy, tod] += np.sum(index)

    monthly = np.divide(
        monthly_sum,
        monthly_count[:, :, None, None, None],
        out=np.zeros_like(monthly_sum),
        where=monthly_count[:, :, None, None, None] > 0,
    ).astype(np.float32)

    raw_doy = np.divide(
        doy_sum,
        doy_count[:, :, None, None, None],
        out=np.zeros_like(doy_sum),
        where=doy_count[:, :, None, None, None] > 0,
    ).astype(np.float32)

    if not np.all(doy_count[365] > 0):
        raw_doy[365] = raw_doy[364]

    weights = linear_decay_weights(SMOOTH_WINDOW)
    half = SMOOTH_WINDOW // 2
    smoothed_doy = np.zeros_like(raw_doy)

    for tod in range(4):
        series = raw_doy[:, tod]
        padded = np.pad(series, ((half, half), (0, 0), (0, 0), (0, 0)), mode="wrap")
        for doy in range(366):
            smoothed_doy[doy, tod] = np.sum(
                padded[doy : doy + SMOOTH_WINDOW] * weights[:, None, None, None],
                axis=0,
            )

    return monthly, smoothed_doy


def evaluate_climatology(clim_monthly: np.ndarray, clim_doy: np.ndarray, base_path: str = BASE_PATH):
    """Evaluate climatology baselines on the test years using per-channel MSE."""
    mse_monthly = np.zeros(NUM_CHANNELS, dtype=np.float64)
    mse_doy = np.zeros(NUM_CHANNELS, dtype=np.float64)
    total_samples = 0

    for year in TEST_YEARS:
        data = load_year(base_path, year)
        months, doys, tods = get_time_mapping(year)
        for idx in tqdm(range(data.shape[0]), desc=f"Evaluate {year}"):
            gt = np.nan_to_num(data[idx, :NUM_CHANNELS, :HEIGHT, :WIDTH])
            monthly_pred = clim_monthly[months[idx], tods[idx]]
            doy_pred = clim_doy[doys[idx], tods[idx]]
            mse_monthly += np.mean((monthly_pred - gt) ** 2, axis=(1, 2))
            mse_doy += np.mean((doy_pred - gt) ** 2, axis=(1, 2))
            total_samples += 1

    return mse_monthly / total_samples, mse_doy / total_samples


def write_results(mse_monthly: np.ndarray, mse_doy: np.ndarray, output_txt: str = OUTPUT_TXT):
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    mean_monthly = float(np.mean(mse_monthly))
    mean_doy = float(np.mean(mse_doy))

    with open(output_txt, "w", encoding="utf-8") as file:
        file.write("Climatology baseline evaluation\n")
        file.write("=" * 72 + "\n")
        file.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"Training years: {TRAIN_YEARS}\n")
        file.write(f"Validation years: {VALID_YEARS}\n")
        file.write(f"Test years: {TEST_YEARS}\n")
        file.write(
            f"Smoothed day-of-year setting: {SMOOTH_WINDOW}-day running window with linearly decaying weights\n"
        )
        file.write("=" * 72 + "\n")
        file.write(f"{'Channel':<10}{'Monthly MSE':<18}{'Smoothed DOY MSE':<18}\n")
        file.write("-" * 72 + "\n")
        for channel in range(NUM_CHANNELS):
            file.write(f"{channel:<10}{mse_monthly[channel]:<18.6f}{mse_doy[channel]:<18.6f}\n")
        file.write("-" * 72 + "\n")
        file.write(f"{'Mean':<10}{mean_monthly:<18.6f}{mean_doy:<18.6f}\n")

    print(f"Monthly climatology mean MSE: {mean_monthly:.6f}")
    print(f"Smoothed DOY climatology mean MSE: {mean_doy:.6f}")
    print(f"Saved results to: {output_txt}")


if __name__ == "__main__":
    monthly_climatology, smoothed_doy_climatology = compute_climatologies()
    mse_monthly, mse_smoothed_doy = evaluate_climatology(monthly_climatology, smoothed_doy_climatology)
    write_results(mse_monthly, mse_smoothed_doy)

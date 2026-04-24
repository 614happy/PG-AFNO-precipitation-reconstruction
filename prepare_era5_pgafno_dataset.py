#!/usr/bin/env python3
"""
Download ERA5 hourly pressure-level and single-level data for a specified region and years,
construct 6-hourly samples, compute train-only z-score statistics, and save normalized yearly
NumPy files.

Output channel order (70 channels total):
  0-12   : geopotential at [50,100,150,200,250,300,400,500,600,700,850,925,1000] hPa
  13-25  : temperature at the same 13 levels
  26-38  : u-component of wind at the same 13 levels
  39-51  : v-component of wind at the same 13 levels
  52-64  : specific humidity at the same 13 levels
  65     : 2 m temperature
  66     : 10 m u-component of wind
  67     : 10 m v-component of wind
  68     : mean sea level pressure
  69     : 6-hour accumulated total precipitation ending at the current analysis time

The script uses hourly ERA5 inputs and creates samples at 00/06/12/18 UTC.
For each sample time t:
  - pressure-level and single-level state variables are taken at time t
  - total precipitation is the sum of hourly ERA5 total precipitation over the previous 6 hours,
    i.e. [t-5h, ..., t]

Normalization:
  - z-score statistics are computed from train years only
  - normalization is applied channel-wise to all splits
  - precipitation is transformed with log1p before z-score normalization
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import xarray as xr

try:
    import cdsapi
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "cdsapi is required. Install it with: pip install 'cdsapi>=0.7.7'"
    ) from exc


PRESSURE_LEVELS_HPA: List[str] = [
    "50", "100", "150", "200", "250", "300", "400",
    "500", "600", "700", "850", "925", "1000"
]

PRESSURE_VARS = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]

SINGLE_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation",
]

ANALYSIS_HOURS = [0, 6, 12, 18]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and preprocess ERA5 into yearly normalized .npy files."
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        required=True,
        help="Years to download and preprocess, e.g. 2011 2012 2013 2014 2015 2016 2017 2018",
    )
    parser.add_argument(
        "--train-years",
        type=int,
        nargs="+",
        required=True,
        help="Years used to compute z-score statistics, e.g. 2011 2012 2013 2014 2015",
    )
    parser.add_argument(
        "--area",
        type=float,
        nargs=4,
        default=[50.0, 100.0, 10.0, 140.0],
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        help="Target region in CDS order: north west south east",
    )
    parser.add_argument(
        "--grid",
        type=float,
        nargs=2,
        default=[0.25, 0.25],
        metavar=("DLAT", "DLON"),
        help="Output grid spacing in degrees",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("path/to/raw_era5"),
        help="Directory for downloaded monthly NetCDF files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("path/to/normalized_yearly_npy"),
        help="Directory for final yearly normalized .npy files and stats",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path("path/to/tmp_preprocessed"),
        help="Directory for intermediate yearly unnormalized .npz files",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload monthly files even if they already exist",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild yearly intermediate files even if they already exist",
    )
    parser.add_argument(
        "--cds-url",
        type=str,
        default="https://cds.climate.copernicus.eu/api",
        help="CDS API URL. Replace or override if needed.",
    )
    parser.add_argument(
        "--cds-key",
        type=str,
        default="YOUR_UID:YOUR_API_KEY",
        help="CDS API key placeholder. Replace or pass via command line.",
    )
    return parser.parse_args()


def get_client(url: str, key: str) -> "cdsapi.Client":
    return cdsapi.Client(url=url, key=key, quiet=False, progress=True)


def month_days(year: int, month: int) -> List[str]:
    period = pd.Period(f"{year:04d}-{month:02d}")
    return [f"{day:02d}" for day in range(1, period.days_in_month + 1)]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def monthly_pressure_path(raw_dir: Path, year: int, month: int) -> Path:
    return raw_dir / "pressure_levels" / f"era5_pressure_{year:04d}_{month:02d}.nc"


def monthly_single_path(raw_dir: Path, year: int, month: int) -> Path:
    return raw_dir / "single_levels" / f"era5_single_{year:04d}_{month:02d}.nc"


def download_monthly_pressure(
    client: "cdsapi.Client",
    raw_dir: Path,
    year: int,
    month: int,
    area: Sequence[float],
    grid: Sequence[float],
    force: bool,
) -> Path:
    target = monthly_pressure_path(raw_dir, year, month)
    ensure_dir(target.parent)
    if target.exists() and not force:
        print(f"Using existing file: {target}")
        return target

    request = {
        "product_type": "reanalysis",
        "variable": PRESSURE_VARS,
        "pressure_level": PRESSURE_LEVELS_HPA,
        "year": f"{year:04d}",
        "month": f"{month:02d}",
        "day": month_days(year, month),
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": list(area),
        "grid": list(grid),
    }
    client.retrieve("reanalysis-era5-pressure-levels", request, str(target))
    return target


def download_monthly_single(
    client: "cdsapi.Client",
    raw_dir: Path,
    year: int,
    month: int,
    area: Sequence[float],
    grid: Sequence[float],
    force: bool,
) -> Path:
    target = monthly_single_path(raw_dir, year, month)
    ensure_dir(target.parent)
    if target.exists() and not force:
        print(f"Using existing file: {target}")
        return target

    request = {
        "product_type": "reanalysis",
        "variable": SINGLE_VARS,
        "year": f"{year:04d}",
        "month": f"{month:02d}",
        "day": month_days(year, month),
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": list(area),
        "grid": list(grid),
    }
    client.retrieve("reanalysis-era5-single-levels", request, str(target))
    return target


def open_monthly_files(paths: Iterable[Path]) -> xr.Dataset:
    datasets = [xr.open_dataset(path) for path in paths]
    combined = xr.concat(datasets, dim="valid_time" if "valid_time" in datasets[0].dims or "valid_time" in datasets[0].coords else "time")
    return combined.sortby("valid_time" if "valid_time" in combined.coords else "time")


def get_time_name(ds: xr.Dataset) -> str:
    if "valid_time" in ds.coords:
        return "valid_time"
    if "time" in ds.coords:
        return "time"
    raise KeyError("No time-like coordinate named 'time' or 'valid_time' was found.")


def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    rename_map = {}
    if "latitude" in ds.dims:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.dims:
        rename_map["longitude"] = "lon"
    if "valid_time" in ds.dims:
        rename_map["valid_time"] = "time"
    if "valid_time" in ds.coords:
        rename_map["valid_time"] = "time"
    ds = ds.rename(rename_map)

    if np.diff(ds["lat"].values).mean() > 0:
        ds = ds.sortby("lat", ascending=False)
    if np.diff(ds["lon"].values).mean() < 0:
        ds = ds.sortby("lon", ascending=True)
    return ds


def build_year_sample_array(
    raw_dir: Path,
    tmp_dir: Path,
    year: int,
    force_rebuild: bool,
) -> Path:
    ensure_dir(tmp_dir)
    out_path = tmp_dir / f"{year:04d}_unnormalized.npz"
    if out_path.exists() and not force_rebuild:
        print(f"Using existing intermediate file: {out_path}")
        return out_path

    pressure_files = [monthly_pressure_path(raw_dir, year, month) for month in range(1, 13)]
    single_files = [monthly_single_path(raw_dir, year, month) for month in range(1, 13)]

    pressure_ds = standardize_coords(open_monthly_files(pressure_files))
    single_ds = standardize_coords(open_monthly_files(single_files))

    time_name_p = get_time_name(pressure_ds)
    time_name_s = get_time_name(single_ds)
    if time_name_p != "time":
        pressure_ds = pressure_ds.rename({time_name_p: "time"})
    if time_name_s != "time":
        single_ds = single_ds.rename({time_name_s: "time"})

    pressure_ds, single_ds = xr.align(pressure_ds, single_ds, join="inner")

    hourly_time = pd.DatetimeIndex(pressure_ds["time"].values)
    if len(hourly_time) == 0:
        raise RuntimeError(f"No hourly timestamps found for year {year}.")

    target_mask = hourly_time.hour.isin(ANALYSIS_HOURS)
    valid_indices = np.where(target_mask)[0]
    valid_indices = valid_indices[valid_indices >= 5]

    target_times = hourly_time[valid_indices]
    num_samples = len(valid_indices)
    nlat = pressure_ds.sizes["lat"]
    nlon = pressure_ds.sizes["lon"]

    year_data = np.empty((num_samples, 70, nlat, nlon), dtype=np.float32)

    level_order = [int(x) for x in PRESSURE_LEVELS_HPA]

    var_to_xr = {
        "geopotential": "z",
        "temperature": "t",
        "u_component_of_wind": "u",
        "v_component_of_wind": "v",
        "specific_humidity": "q",
        "2m_temperature": "t2m",
        "10m_u_component_of_wind": "u10",
        "10m_v_component_of_wind": "v10",
        "mean_sea_level_pressure": "msl",
        "total_precipitation": "tp",
    }

    pressure_stacks = []
    for var in PRESSURE_VARS:
        da = pressure_ds[var_to_xr[var]].sel(pressure_level=level_order)
        pressure_stacks.append(da.values.astype(np.float32))
    # Each entry is [time, level, lat, lon]

    t2m = single_ds[var_to_xr["2m_temperature"]].values.astype(np.float32)
    u10 = single_ds[var_to_xr["10m_u_component_of_wind"]].values.astype(np.float32)
    v10 = single_ds[var_to_xr["10m_v_component_of_wind"]].values.astype(np.float32)
    msl = single_ds[var_to_xr["mean_sea_level_pressure"]].values.astype(np.float32)
    tp_hourly = single_ds[var_to_xr["total_precipitation"]].values.astype(np.float32)

    for out_i, src_i in enumerate(valid_indices):
        channel_cursor = 0
        for stack in pressure_stacks:
            year_data[out_i, channel_cursor:channel_cursor + 13] = stack[src_i]
            channel_cursor += 13

        year_data[out_i, 65] = t2m[src_i]
        year_data[out_i, 66] = u10[src_i]
        year_data[out_i, 67] = v10[src_i]
        year_data[out_i, 68] = msl[src_i]

        tp_6h = tp_hourly[src_i - 5:src_i + 1].sum(axis=0)
        year_data[out_i, 69] = tp_6h

    np.savez_compressed(
        out_path,
        data=year_data,
        times=target_times.astype("datetime64[ns]").values,
    )
    print(f"Saved intermediate yearly samples: {out_path}")
    return out_path


def compute_train_stats(intermediate_paths: Sequence[Path]) -> tuple[np.ndarray, np.ndarray]:
    channel_sum = np.zeros(70, dtype=np.float64)
    channel_sq_sum = np.zeros(70, dtype=np.float64)
    total_count = 0

    for path in intermediate_paths:
        with np.load(path) as npz:
            data = npz["data"].astype(np.float64)

        data_for_stats = data.copy()
        data_for_stats[:, 69] = np.log1p(np.maximum(data_for_stats[:, 69], 0.0))

        n_samples, n_channels, nlat, nlon = data_for_stats.shape
        reshaped = data_for_stats.transpose(1, 0, 2, 3).reshape(n_channels, -1)
        channel_sum += reshaped.sum(axis=1)
        channel_sq_sum += np.square(reshaped).sum(axis=1)
        total_count += reshaped.shape[1]

    mean = channel_sum / total_count
    var = channel_sq_sum / total_count - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_and_save_year(
    intermediate_path: Path,
    out_dir: Path,
    mean: np.ndarray,
    std: np.ndarray,
) -> Path:
    ensure_dir(out_dir)
    year = intermediate_path.stem.split("_")[0]
    out_path = out_dir / f"{year}.npy"

    with np.load(intermediate_path) as npz:
        data = npz["data"].astype(np.float32)

    data[:, 69] = np.log1p(np.maximum(data[:, 69], 0.0))
    data = (data - mean.reshape(1, 70, 1, 1)) / std.reshape(1, 70, 1, 1)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    np.save(out_path, data)
    print(f"Saved normalized yearly array: {out_path}")
    return out_path


def save_stats(out_dir: Path, mean: np.ndarray, std: np.ndarray, args: argparse.Namespace) -> None:
    stats_npz = out_dir / "normalization_stats.npz"
    np.savez_compressed(stats_npz, mean=mean, std=std)

    metadata = {
        "pressure_levels_hpa": [int(x) for x in PRESSURE_LEVELS_HPA],
        "pressure_variables": PRESSURE_VARS,
        "single_variables": SINGLE_VARS,
        "channel_order_description": [
            "13 geopotential levels",
            "13 temperature levels",
            "13 u-wind levels",
            "13 v-wind levels",
            "13 specific-humidity levels",
            "2m temperature",
            "10m u wind",
            "10m v wind",
            "mean sea level pressure",
            "6-hour accumulated total precipitation",
        ],
        "analysis_hours_utc": ANALYSIS_HOURS,
        "precipitation_construction": "hourly total precipitation summed over [t-5h, ..., t]",
        "area_north_west_south_east": list(args.area),
        "grid_degrees": list(args.grid),
        "train_years_for_zscore": list(args.train_years),
        "precipitation_transform_before_zscore": "log1p",
    }
    with open(out_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved normalization stats: {stats_npz}")
    print(f"Saved metadata: {out_dir / 'dataset_metadata.json'}")


def main() -> None:
    args = parse_args()

    if not set(args.train_years).issubset(set(args.years)):
        raise ValueError("--train-years must be a subset of --years.")

    ensure_dir(args.raw_dir)
    ensure_dir(args.tmp_dir)
    ensure_dir(args.out_dir)

    client = get_client(args.cds_url, args.cds_key)

    years_sorted = sorted(set(args.years))
    for year in years_sorted:
        for month in range(1, 13):
            download_monthly_pressure(
                client=client,
                raw_dir=args.raw_dir,
                year=year,
                month=month,
                area=args.area,
                grid=args.grid,
                force=args.force_download,
            )
            download_monthly_single(
                client=client,
                raw_dir=args.raw_dir,
                year=year,
                month=month,
                area=args.area,
                grid=args.grid,
                force=args.force_download,
            )

    intermediate_paths = []
    for year in years_sorted:
        intermediate_paths.append(
            build_year_sample_array(
                raw_dir=args.raw_dir,
                tmp_dir=args.tmp_dir,
                year=year,
                force_rebuild=args.force_rebuild,
            )
        )

    train_paths = [
        path for path in intermediate_paths if int(path.stem.split("_")[0]) in set(args.train_years)
    ]
    mean, std = compute_train_stats(train_paths)
    save_stats(args.out_dir, mean, std, args)

    for path in intermediate_paths:
        normalize_and_save_year(path, args.out_dir, mean, std)

    print("Done.")


if __name__ == "__main__":
    main()

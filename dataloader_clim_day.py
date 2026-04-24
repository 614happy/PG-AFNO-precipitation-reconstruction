import os
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_BASE_PATH = "path/to/normalized_era5"
HEIGHT = 160
WIDTH = 160
INPUT_CHANNEL_INDEX = 69
NUM_TARGET_CHANNELS = 69


def get_time_mapping(year: int):
    """Return day-of-year and 6-hour slot indices for a full year."""
    start = datetime(year, 1, 1, 0, 0)
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    days = 366 if is_leap else 365
    total_steps = days * 4

    doys = np.zeros(total_steps, dtype=np.int32)
    tods = np.zeros(total_steps, dtype=np.int32)

    for idx in range(total_steps):
        current = start + timedelta(hours=6 * idx)
        doys[idx] = current.timetuple().tm_yday - 1
        tods[idx] = current.hour // 6

    return doys, tods


class ERA5ClimDoyDataset(Dataset):
    """Dataset for experiments that use precipitation plus smoothed DOY climatology."""

    def __init__(self, base_path=DEFAULT_BASE_PATH, split="train", clim_doy_matrix=None):
        super().__init__()
        if clim_doy_matrix is None:
            raise ValueError("clim_doy_matrix must be provided.")

        self.base_path = base_path
        self.clim_doy = clim_doy_matrix
        split_config = {
            "train": [2011, 2012, 2013, 2014, 2015],
            "valid": [2016],
            "test": [2017, 2018],
        }
        if split not in split_config:
            raise ValueError(f"Unknown split: {split}")

        self.data, self.doys, self.tods = self._load_and_concat_data(split_config[split])

    def _load_and_concat_data(self, years):
        data_list = []
        doy_list = []
        tod_list = []

        for year in years:
            file_path = os.path.join(self.base_path, f"{year}.npy")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing data file: {file_path}")
            year_data = np.load(file_path)
            doys, tods = get_time_mapping(year)
            data_list.append(year_data)
            doy_list.append(doys)
            tod_list.append(tods)

        return (
            np.concatenate(data_list, axis=0),
            np.concatenate(doy_list, axis=0),
            np.concatenate(tod_list, axis=0),
        )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        precip = self.data[idx, INPUT_CHANNEL_INDEX : INPUT_CHANNEL_INDEX + 1, :HEIGHT, :WIDTH]
        doy = self.doys[idx]
        tod = self.tods[idx]
        clim_features = self.clim_doy[doy, tod]
        target = self.data[idx, :NUM_TARGET_CHANNELS, :HEIGHT, :WIDTH]

        input_array = np.concatenate([precip, clim_features], axis=0)
        input_tensor = torch.nan_to_num(torch.from_numpy(input_array).float(), nan=0.0)
        target_tensor = torch.nan_to_num(torch.from_numpy(target).float(), nan=0.0)
        return input_tensor, target_tensor


def get_dataloader_clim_doy(
    train_batch_size=16,
    valid_batch_size=16,
    test_batch_size=1,
    base_path=DEFAULT_BASE_PATH,
    clim_doy_matrix=None,
    num_workers=0,
):
    train_dataset = ERA5ClimDoyDataset(base_path=base_path, split="train", clim_doy_matrix=clim_doy_matrix)
    valid_dataset = ERA5ClimDoyDataset(base_path=base_path, split="valid", clim_doy_matrix=clim_doy_matrix)
    test_dataset = ERA5ClimDoyDataset(base_path=base_path, split="test", clim_doy_matrix=clim_doy_matrix)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, valid_loader, test_loader

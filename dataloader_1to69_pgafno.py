import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ERA5Dataset(Dataset):
    """
    ERA5 dataset for the 1-to-69 reconstruction task.

    Input:
        channel 69, corresponding to 6-hour accumulated precipitation

    Output:
        channels 0-68, corresponding to the 69 atmospheric-state variables
    """

    def __init__(self, base_path: str = "path/to/normalized_era5", split: str = "train") -> None:
        super().__init__()
        self.base_path = base_path
        self.split = split

        split_config = {
            "train": [2011, 2012, 2013, 2014, 2015],
            "valid": [2016],
            "test": [2017, 2018],
        }
        if split not in split_config:
            raise ValueError(f"Unknown split: {split}. Available splits: {list(split_config.keys())}")

        self.data = self._load_and_concat_data(split_config[split])
        self.num_samples = self.data.shape[0]

    def _load_and_concat_data(self, years: list[int]) -> np.ndarray:
        data_list = []
        for year in years:
            file_path = os.path.join(self.base_path, f"{year}.npy")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")
            year_data = np.load(file_path)
            data_list.append(year_data)
            print(f"Loaded {year}.npy")

        return np.concatenate(data_list, axis=0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_data = self.data[idx, 69, :160, :160]
        target_data = self.data[idx, :69, :160, :160]

        input_tensor = torch.from_numpy(input_data).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_data).float()

        input_tensor = torch.nan_to_num(input_tensor, nan=0.0)
        target_tensor = torch.nan_to_num(target_tensor, nan=0.0)

        return input_tensor, target_tensor

    def __len__(self) -> int:
        return self.num_samples


def get_dataloader(
    train_batch_size: int = 12,
    valid_batch_size: int = 12,
    test_batch_size: int = 1,
    base_path: str = "path/to/normalized_era5",
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = ERA5Dataset(base_path=base_path, split="train")
    valid_dataset = ERA5Dataset(base_path=base_path, split="valid")
    test_dataset = ERA5Dataset(base_path=base_path, split="test")

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


if __name__ == "__main__":
    train_loader, valid_loader, test_loader = get_dataloader(
        train_batch_size=8,
        valid_batch_size=8,
        test_batch_size=1,
    )

    print("=== Train split ===")
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of training batches: {len(train_loader)}")
    for inputs, targets in train_loader:
        print(f"Input shape: {inputs.shape}")
        print(f"Target shape: {targets.shape}")
        break

    print("\n=== Validation split ===")
    print(f"Number of validation samples: {len(valid_loader.dataset)}")
    print(f"Number of validation batches: {len(valid_loader)}")

    print("\n=== Test split ===")
    print(f"Number of test samples: {len(test_loader.dataset)}")
    print(f"Number of test batches: {len(test_loader)}")

# PG-AFNO for Precipitation-Constrained Atmospheric State Reconstruction

This repository contains the code used for the manuscript:

**Quantifying How 6-h Precipitation Constrains the 3D Atmospheric State with a Physics-Guided Neural Operator**

## Overview

This project studies how much of the concurrent three-dimensional atmospheric state can be reconstructed from a single 6-hour accumulated precipitation field under an ERA5 prior. The retained model is a physics-guided Adaptive Fourier Neural Operator (PG-AFNO) with:

- adaptive spectral thresholding in Fourier space
- weak geostrophic regularization during training
- a 1-to-69 reconstruction setting from precipitation to multilevel atmospheric variables

The experiments include:

- the retained PG-AFNO model
- climatology baselines
- pixel-wise linear regression
- ResNet, UNet, and ViT comparison models
- physical validation and spectral diagnostics
- figure-generation scripts for the manuscript

## Repository structure

```text
.
├── requirements.txt
├── README.md
├── models_1to69_pgafno.py
├── dataloader_1to69_pgafno.py
├── train_1to69_pgafno.py
├── climatology_baseline.py
├── dataloader_clim_day.py
├── train_linear.py
├── train_resnet_unet.py
├── train_unet.py
├── train_vit.py
├── figure2_threshold_curve_pgafno.py
├── figure2_spectral_diagnostics_pgafno.py
├── figure3_case_panels_pgafno.py
├── figure3_ground_truth_synoptic_pgafno.py
├── figure3_ground_truth_vertical_sections_pgafno.py
└── physical_validation_pgafno.py
```

## Data preparation

The model is trained on ERA5 hourly pressure-level and single-level reanalysis data over East Asia
(10°N–50°N, 100°E–140°E) on a 0.25° × 0.25° grid.

The preprocessing script downloads the official ERA5 hourly fields, constructs 6-hourly samples at
00/06/12/18 UTC, and saves yearly NumPy files after train-only z-score normalization.

Channel order:
- 13 pressure levels of geopotential
- 13 pressure levels of temperature
- 13 pressure levels of u wind
- 13 pressure levels of v wind
- 13 pressure levels of specific humidity
- 2 m temperature
- 10 m u wind
- 10 m v wind
- mean sea level pressure
- 6-hour accumulated precipitation

Pressure levels:
50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa

The script used to prepare the dataset is:
`prepare_era5_pgafno_dataset.py`

## Example: dataset preparation

```bash
python prepare_era5_pgafno_dataset.py \
  --years 2011 2012 2013 2014 2015 2016 2017 2018 \
  --train-years 2011 2012 2013 2014 2015 \
  --raw-dir path/to/raw_era5 \
  --tmp-dir path/to/tmp_preprocessed \
  --out-dir path/to/normalized_yearly_npy \
  --cds-url https://cds.climate.copernicus.eu/api \
  --cds-key YOUR_UID:YOUR_API_KEY

## Environment

Create a Python environment and install dependencies:

```bash
action(){ :; }
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main training

Train the retained PG-AFNO model:

```bash
python train_1to69_pgafno.py
```

Before running, update the placeholder paths in the script to your local locations, for example:

- dataset root: `path/to/era5_normalized`
- checkpoint path: `path/to/checkpoints/pretrained.ckpt`
- output root: `path/to/output/pgafno`

## Baselines

Run the climatology baseline:

```bash
python climatology_baseline.py
```

Run the linear baseline:

```bash
python train_linear.py
```

Run the ResNet or UNet baseline:

```bash
python train_resnet_unet.py --model resnet
python train_resnet_unet.py --model unet
```

Run the standalone UNet script if needed:

```bash
python train_unet.py
```

Run the ViT baseline:

```bash
python train_vit.py
```

## Physical validation and figure generation

Generate Figure 2 threshold curve:

```bash
python figure2_threshold_curve_pgafno.py
```

Generate Figure 2 spectral diagnostics:

```bash
python figure2_spectral_diagnostics_pgafno.py --ckpt path/to/checkpoints/best_model.pth --out_dir path/to/output/figure2
```

Generate Figure 3 predicted case panels:

```bash
python figure3_case_panels_pgafno.py --ckpt path/to/checkpoints/best_model.pth --case_dir path/to/case_indices --out_dir path/to/output/figure3
```

Generate Figure 3 ground-truth synoptic panels:

```bash
python figure3_ground_truth_synoptic_pgafno.py
```

Generate Figure 3 ground-truth vertical sections:

```bash
python figure3_ground_truth_vertical_sections_pgafno.py
```

Run physical validation:

```bash
python physical_validation_pgafno.py --task all --ckpt path/to/checkpoints/best_model.pth --out_dir path/to/output/physical_validation
```

## Reproducibility notes

- The retained PG-AFNO setting uses 12 AFNO layers, patch size 8 × 8, embedding dimension 768, and dropout 0.15.
- Training uses AdamW with an initial learning rate of 1e-3, weight decay 1e-5, cosine annealing, batch size 12, and early stopping with a patience of 30 epochs.
- The geostrophic regularization weight is 0.005 in the retained training setup.
- The adaptive spectral threshold uses `base_threshold = 0.01`, `k_decay = 20.0`, and `min_threshold = 5e-4`.

## Citation

If you use this code, please cite the associated manuscript and archive record.

You may also add a `CITATION.cff` file after creating the repository archive and DOI.

## License

Choose a license before making the repository public. For research code, a permissive license such as MIT or BSD-3-Clause is often used when redistribution is intended.

## Notes

- Replace all `path/to/...` placeholders before running the scripts.
- Check that no private paths, credentials, or machine-specific information remain in the repository before publishing.
- The original ERA5 data are obtained from the Copernicus Climate Data Store using the official CDS API.
- Large data files should not be committed directly to GitHub.

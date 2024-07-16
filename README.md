# Depth alignment module

This is a module to align relative depths to SfM points.

Codes are referenced from https://github.com/maturk/dn-splatter.git

## Installation

We encourage CUDA 11.8, ubuntu 22.04 settings.

```bash
conda env create -f environment.yml
conda activate depth
```

## Usage

The dataset is expected to be in COLMAP format (contains a colmap/sparse/0 folder in data root) since SfM points are required. Below is the script to generate SfM aligned depths from images.

```bash
python align_depth.py --data [path_to_data_root] \
                      --no-skip-colmap-to-depths \
                      --no-skip-mono-depth-creation \
                      --mono_depth_network=[zoe/depth_anything_v2] \
                      --align_method=[closed_form/grad_descent]
```

"""Some useful util functions"""

import os
import random
from pathlib import Path
from typing import List, Literal, Optional, Union

import cv2
import numpy as np
import open3d as o3d
import torch
from utils.camera_utils import OPENGL_TO_OPENCV, get_means3d_backproj
from natsort import natsorted
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import resize
from tqdm import tqdm


from utils import colormaps


# Depth Scale Factor m to mm
SCALE_FACTOR = 0.001

# Pass
def get_filename_list(image_dir: Path, ends_with: Optional[str] = None) -> List:
    """List directory and save filenames

    Returns:
        image_filenames
    """
    image_filenames = os.listdir(image_dir)
    if ends_with is not None:
        image_filenames = [
            image_dir / name
            for name in image_filenames
            if name.lower().endswith(ends_with)
        ]
    else:
        image_filenames = [image_dir / name for name in image_filenames]
    image_filenames = natsorted(image_filenames)
    return image_filenames

# Pass
def image_path_to_tensor(
    image_path: Path, size: Optional[tuple] = None, black_and_white=False
) -> Tensor:
    """Convert image from path to tensor

    Returns:
        image: Tensor
    """
    img = Image.open(image_path)
    if black_and_white:
        img = img.convert("1")
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    if size:
        img_tensor = resize(
            img_tensor.permute(2, 0, 1), size=size, antialias=None
        ).permute(1, 2, 0)
    return img_tensor

# Pass
def depth_path_to_tensor(
    depth_path: Path, scale_factor: float = SCALE_FACTOR, return_color=False
) -> Tensor:
    """Load depth image in either .npy or .png format and return tensor

    Args:
        depth_path: Path
        scale_factor: float
        return_color: bool
    Returns:
        depth tensor and optionally colored depth tensor
    """
    if depth_path.suffix == ".png":
        depth = cv2.imread(str(depth_path.absolute()), cv2.IMREAD_ANYDEPTH)
    elif depth_path.suffix == ".npy":
        depth = np.load(depth_path, allow_pickle=True)
        if len(depth.shape) == 3:
            depth = depth[..., 0]
    else:
        raise Exception(f"Format is not supported {depth_path.suffix}")
    depth = depth * scale_factor
    depth = depth.astype(np.float32)
    depth = torch.from_numpy(depth).unsqueeze(-1)
    if not return_color:
        return depth
    else:
        depth_color = colormaps.apply_depth_colormap(depth)
        return depth, depth_color  # type: ignore


def save_img(image, image_path, verbose=True) -> None:
    """helper to save images

    Args:
        image: image to save (numpy, Tensor)
        image_path: path to save
        verbose: whether to print save path

    Returns:
        None
    """
    if image.shape[-1] == 1 and torch.is_tensor(image):
        image = image.repeat(1, 1, 3)
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy() * 255
        image = image.astype(np.uint8)
    if not Path(os.path.dirname(image_path)).exists():
        Path(os.path.dirname(image_path)).mkdir(parents=True)
    im = Image.fromarray(image)
    if verbose:
        print("saving to: ", image_path)
    im.save(image_path)

# Pass
def save_depth(depth, depth_path, verbose=True, scale_factor=SCALE_FACTOR) -> None:
    """helper to save metric depths

    Args:
        depth: image to save (numpy, Tensor)
        depth_path: path to save
        verbose: whether to print save path
        scale_factor: depth metric scaling factor

    Returns:
        None
    """
    if torch.is_tensor(depth):
        depth = depth.float() / scale_factor
        depth = depth.detach().cpu().numpy()
    else:
        depth = depth / scale_factor
    if not Path(os.path.dirname(depth_path)).exists():
        Path(os.path.dirname(depth_path)).mkdir(parents=True)
    if verbose:
        print("saving to: ", depth_path)
    np.save(depth_path, depth)
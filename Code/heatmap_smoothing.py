import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List
from argparse import ArgumentParser
from utils import normalize_image

def avg_pooling(mask: torch.Tensor, kernel_size: int , stride: int) -> torch.Tensor:
    """
    Apply average pooling to a tensor.

    Args:
        mask (torch.Tensor): The input tensor to pool.
        kernel_size (int): Size of the pooling kernel. Default is 129.
        stride (int): Stride of the pooling operation. Default is 1.

    Returns:
        torch.Tensor: The pooled tensor.
    """
    pooling = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=kernel_size//2,count_include_pad=False)
    return pooling(mask)

def prepare_directories(base_dir: str, model_type: str, xai_type: str, dataset_split: str, categories: List[str]) -> str:
    """
    Create necessary directories for saving processed masks.

    Args:
        base_dir (str): Base directory for the dataset and results.
        model_type (str): The model type being used.
        xai_type (str): The XAI method being used.
        dataset_split (str): Dataset split (e.g., 'test').
        categories (List[str]): List of categories.

    Returns:
        str: Path to the XAI directory.
    """
    xai_dir = os.path.join(base_dir, "auswertung", model_type, xai_type, dataset_split)
    for category in categories:
        grad_mask_dir = os.path.join(xai_dir, category, "grad_mask")
        os.makedirs(grad_mask_dir, exist_ok=True)
    return xai_dir

def process_images(xai_dir: str, categories: List[str], images_path: str, kernel_size: int, stride: int):
    """
    Process images to generate and save smoothed grad masks.

    Args:
        xai_dir (str): Directory containing XAI masks.
        categories (List[str]): List of categories.
        images_path (str): Path to the images.
        kernel_size (int): Pooling kernel size.
        stride (int): Pooling stride.
    """
    for category in categories:
        category_images = os.listdir(os.path.join(images_path, category))
        for image_name in category_images:
            mask_path = os.path.join(xai_dir, category, "mask", image_name + ".npy")
            grad_mask_path = os.path.join(xai_dir, category, "grad_mask", image_name)

            mask_raw = np.load(mask_path)
            mask_tensor = torch.tensor(mask_raw, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            pooled_mask = avg_pooling(mask_tensor, kernel_size=kernel_size, stride=stride)

            grad_mask = (
                normalize_image(mask_raw) +
                normalize_image(pooled_mask.squeeze().numpy()) / 100
            )

            np.save(grad_mask_path, grad_mask)

def main():
    """
    Main function to process heatmaps and apply smoothing.
    """
    parser = ArgumentParser(description="Heatmap Smoothing Script")
    parser.add_argument("--model_type", type=str, default="simple_cnn", help="Model type (e.g., simple_cnn)")
    parser.add_argument("--xai_type", type=str, default="gradcam", help="XAI method (e.g., gradcam)")
    parser.add_argument("--base_dir", type=str, default="data/", help="Base directory for data")
    parser.add_argument("--dataset", type=str, default="atsds_large", help="Dataset name")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split (e.g., train, test)")
    parser.add_argument("--kernel_size", type=int, default=129, help="Kernel size for pooling")
    parser.add_argument("--stride", type=int, default=1, help="Stride for pooling")

    args = parser.parse_args()

    # Paths and directories
    images_path = os.path.join(args.base_dir, args.dataset, args.dataset_split)
    categories = sorted(os.listdir(images_path))

    xai_dir = prepare_directories(
        args.base_dir, args.model_type, args.xai_type, args.dataset_split, categories
    )

    # Process images
    process_images(
        xai_dir=xai_dir,
        categories=categories,
        images_path=images_path,
        kernel_size=args.kernel_size,
        stride=args.stride
    )

if __name__ == "__main__":
    main()

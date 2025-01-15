import os
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import cv2
import argparse

from ATSDS import ATSDS
from model import get_model
from gradcam import get_gradcam

from utils import setup_environment, prepare_categories_and_images, create_output_directories, save_xai_outputs, load_checkpoint


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="XAI Method Visualization Pipeline")

    # Configuration variables
    parser.add_argument('--model_name', type=str, default="simple_cnn", help="Name of the model.")
    parser.add_argument('--model_checkpoint', type=str, default="model/simple_cnn_1_1.tar", help="Path to the model checkpoint.")
    parser.add_argument('--dataset_path', type=str, default="data", help="Path to the dataset.")
    parser.add_argument('--dataset_type', type=str, default="atsds_large", help="Type of the dataset.")
    parser.add_argument('--dataset_split', type=str, default="test", help="Dataset split (e.g., 'train', 'test').")
    parser.add_argument('--images_path', type=str, default="data/atsds_large/test", help="Path to the images.")
    parser.add_argument('--output_path', type=str, default="data/auswertung/simple_cnn/gradcam/test/", help="Path to save outputs.")
    parser.add_argument('--random_seed', type=int, default=1414, help="Random seed for reproducibility.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for data loader.")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for data loading.")

    return parser.parse_args()






def generate_gradcam_visualizations(model: torch.nn.Module, device: torch.device, categories: list[str],
                                     imagedict: dict[str, list[str]], label_idx_dict: dict[str, int],
                                     output_path: str, images_path: str) -> None:
    """
    Generate Grad-CAM visualizations for each image in the dataset and save them.

    Args:
        model (torch.nn.Module): The model to be used for generating Grad-CAM visualizations.
        device (torch.device): The device to run the model on (GPU or CPU).
        categories (list): List of categories in the dataset.
        imagedict (dict): A dictionary of image filenames for each category.
        label_idx_dict (dict): A dictionary mapping category names to indices.
        output_path (str): Path where Grad-CAM results will be saved.
        images_path (str): Path to the dataset images.
    """
    for category in categories:
        model.eval()
        images = imagedict[category]
        for image_name in images:
            with Image.open(os.path.join(images_path, category, image_name)) as img:
                image_tensor = TRANSFORM_TEST(img).unsqueeze(0).to(device)
                shape = img.size[::-1]  # PIL uses (width, height)

                mask, _ = get_gradcam(model, model.conv3, image_tensor, label_idx_dict[category], shape)
                save_xai_outputs(mask, np.array(img), category, image_name, output_path)

def main():
    # Parse command-line arguments
    args = parse_args()

    # Transforms
    global TRANSFORM_TEST
    TRANSFORM_TEST = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Setup environment
    device = setup_environment(args.random_seed)

    # Load dataset
    testset = ATSDS(root=args.dataset_path, split=args.dataset_split, dataset_type=args.dataset_type, transform=TRANSFORM_TEST)

    # Load model
    model = get_model(args.model_name, n_classes=testset.get_num_classes()).to(device)
    model.eval()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Load checkpoint
    epoch, trainstats = load_checkpoint(args.model_checkpoint, model, optimizer, scheduler)
    print(f"Model checkpoint loaded. Epoch: {epoch}")

    # Prepare categories and images
    categories, label_idx_dict, imagedict = prepare_categories_and_images(args.images_path)

    # Ensure output directories exist
    create_output_directories(args.output_path, categories)

    # Run Grad-CAM visualization
    generate_gradcam_visualizations(
        model, device, categories, imagedict, label_idx_dict, args.output_path, args.images_path
    )

if __name__ == "__main__":
    main()

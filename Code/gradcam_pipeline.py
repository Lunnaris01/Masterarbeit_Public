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
    parser.add_argument('--output_path', type=str, default="data/auswertung/", help="Path to save outputs.")
    parser.add_argument('--random_seed', type=int, default=1414, help="Random seed for reproducibility.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for data loader.")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for data loading.")
    parser.add_argument('--target_layer', type=str, default="conv3", help="Target layer for Grad-CAM (e.g., Simple CNN: 'conv3' ResNet: 'layer4.-1.conv3).")

    return parser.parse_args()



def get_target_layer(model, target_layer_name):
    """Get the target layer from the model based on the given name."""
    if '.' in target_layer_name:  # Handle complex paths like 'layer4.-1.conv3'
        layers = target_layer_name.split('.')  # e.g., ['layer4', '-1', 'conv3']
        block_layer = getattr(model,layers[0])
        target_block = block_layer[int(layers[1])]
        target_layer = getattr(target_block,layers[2])
    else:
        # Handle simple layer names like 'conv3'
        target_layer = getattr(model, target_layer_name)

    # Ensure that the layer exists
    if target_layer is None:
        raise ValueError(f"Layer '{target_layer_name}' not found in the model.")
    return target_layer



def generate_gradcam_visualizations(model: torch.nn.Module, device: torch.device, categories: list[str],
                                     imagedict: dict[str, list[str]], label_idx_dict: dict[str, int],
                                     output_path: str, images_path: str, layer_target: str) -> None:
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
    target_layer = get_target_layer(model, layer_target)
    for category in categories:
        model.eval()
        images = imagedict[category]
        for image_name in images:
            with Image.open(os.path.join(images_path, category, image_name)) as img:
                image_tensor = TRANSFORM_TEST(img).unsqueeze(0).to(device)
                shape = img.size[::-1]  # PIL uses (width, height)

                mask, _ = get_gradcam(model, target_layer, image_tensor, label_idx_dict[category], shape)
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
    output_path = args.output_path + args.model_name + "/gradcam/test/"
    create_output_directories(output_path, categories)

    # Run Grad-CAM visualization
    generate_gradcam_visualizations(
        model, device, categories, imagedict, label_idx_dict, output_path, args.images_path, args.target_layer
    )

if __name__ == "__main__":
    main()

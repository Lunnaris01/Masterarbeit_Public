import argparse
import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from skimage.segmentation import mark_boundaries
from torchprism import PRISM

from ATSDS import ATSDS
from model import get_model
from utils import setup_environment, prepare_categories_and_images, create_output_directories, save_xai_outputs , load_checkpoint, normalize_image


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="PRISM Method Visualization Pipeline")

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
    parser.add_argument('--num_samples', type=int, default=100, help="Number of images for PRISM explanation.")

    return parser.parse_args()


def generate_prism_visualizations(model, device, categories, imagedict, label_idx_dict, output_path, images_path):
    """Generate PRISM visualizations for each image in the dataset."""
    # Register hooks for PRISM
    PRISM.prune_old_hooks(model)
    PRISM.register_hooks(model)

    for category in categories:
        images = imagedict[category]
        for image_name in images:
            with open(os.path.join(images_path, category, image_name), 'rb') as f:
                with Image.open(f) as current_image:
                    current_image_tensor = TRANSFORM_TEST(current_image).unsqueeze(0).to(device)

                    # Forward pass to generate PRISM maps
                    model(current_image_tensor)

                    # Retrieve PRISM maps
                    prism_maps = PRISM.get_maps()
                    drawable_prism_maps = prism_maps.permute(0, 2, 3, 1).detach().cpu().numpy().squeeze()

                    # Save PRISM maps overlayed on original image
                    save_moi = Image.fromarray((drawable_prism_maps * 255).astype(np.uint8))
                    save_moi.save(os.path.join(output_path, category, 'mask_on_image', image_name), "PNG")

                    # Normalize and save PRISM mask
                    mask_raw = np.array(save_moi)
                    mask_norm = (mask_raw - mask_raw.mean()) // mask_raw.std()
                    mask_f = np.max(mask_norm, axis=2)
                    mask = normalize_image(F.interpolate(F.relu(torch.Tensor(mask_f)).reshape(1, 1, 224, 224),
                                                         (512, 512), mode="bilinear").squeeze().squeeze().numpy())
                    np.save(os.path.join(output_path, category, 'mask', image_name), mask)

def main():
    # Parse command-line arguments
    args = parse_args()


    # Transforms
    global TRANSFORM_TEST
    TRANSFORM_TEST = transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Setup environment
    device = setup_environment(args.random_seed)

    # Load dataset and dataloader
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
    output_path = args.output_path + args.model_name + "/prism/test/"
    create_output_directories(output_path, categories)

    # Generate PRISM visualizations
    generate_prism_visualizations(
        model, device, categories, imagedict, label_idx_dict,
        output_path, args.images_path
    )

if __name__ == "__main__":
    main()

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
import saliency.core as saliency
import pickle

from ATSDS import ATSDS
from model import get_model
from utils import setup_environment, prepare_categories_and_images, create_output_directories, save_xai_outputs , load_checkpoint, normalize_image
from utils import duration_distance_weighted_density_mask_gaussian, add_gaussian_bell_to_image, create_gaussian_bell

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
    parser.add_argument('--fixpoint_nums', type=int, default=64, help="Amount of ig attributions used as fixpoints in IGF")
    parser.add_argument('--scale', type=float, default =1.2e-3, help = "Scale parameter controlling the impact on the surroundings of a fixpoint")
    return parser.parse_args()
    
def ig_fixpoints(attributions,fixpoint_nums=256,scale = 1.20e-3):
    x,y = attributions.shape[1:3]
    ig_fix_mask = np.zeros_like(attributions)
    pixels = [(i,j) for i in range(x) for j in range(y)]
    weights = attributions.flatten()
    weighted_pixels = [(w,x) for w,x in zip(weights,pixels)]
    sorted_weighted_pixels = sorted(weighted_pixels)[::-1][:fixpoint_nums]
    w = [i for i, j in sorted_weighted_pixels]
    p = [j for i, j in sorted_weighted_pixels]

    ig_fix_mask = duration_distance_weighted_density_mask_gaussian(p,w,bell_scale=scale)
    return ig_fix_mask[1]


def generate_xrai_visualizations(model, model_name, device, categories, imagedict, label_idx_dict, output_path, images_path,fixpoint_nums,scale):
    """Generate IGF visualizations for each image in the dataset."""

    for category in categories:
        images = imagedict[category]
        for image_name in images:
            with open(os.path.join(images_path, category, image_name), 'rb') as f:
                with Image.open(f) as current_image:
                    current_image_tensor = TRANSFORM_TEST(current_image).to(device)
                    with open("data/auswertung/" + model_name + "/ig/test/" + category + "/" +  image_name[:-4] + ".pkl", "rb") as ig_file:
                        attribs = pickle.load(ig_file)
                        mask_raw = ig_fixpoints(attribs,fixpoint_nums = fixpoint_nums,scale = scale)
                        mask = normalize_image(F.interpolate(torch.Tensor(mask_raw).reshape(1,1,224,224),(512,512),mode = "bilinear").squeeze().squeeze().numpy())
                        save_xai_outputs(mask, np.array(current_image), category, image_name, output_path)

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
    output_path = args.output_path + args.model_name + "/igf/test/"
    create_output_directories(output_path, categories)

    # Generate IGF visualizations
    generate_xrai_visualizations(
        model,args.model_name, device, categories, imagedict, label_idx_dict,
        output_path, args.images_path, args.fixpoint_nums, args.scale
    )

if __name__ == "__main__":
    main()

import os
import numpy as np
from PIL import Image
from tqdm import tqdm  # For progress bars
from argparse import ArgumentParser  # For command-line arguments
from utils import get_percentage_of_image, normalize_image

def get_image_categories(images_path: str) -> dict:
    """
    Retrieve image categories and their corresponding file lists.

    Args:
        images_path (str): Path to the directory containing categorized images.

    Returns:
        dict: A dictionary mapping categories to lists of image filenames.
    """
    categories = sorted(os.listdir(images_path))
    return {cat: os.listdir(os.path.join(images_path, cat)) for cat in categories}

def process_images(experiment: str, pct: int, categories: dict, images_path: str, background_dir: str, xai_dir: str, base_dir: str, model_type: str, xai_type: str):
    """
    Process images for a given experiment and percentage.

    Args:
        experiment (str): Type of experiment ("revelation" or "occlusion").
        pct (int): Percentage value for the experiment.
        categories (dict): Dictionary of categories and their image filenames.
        images_path (str): Path to the dataset images.
        background_dir (str): Path to the background images.
        xai_dir (str): Path to the XAI mask directory.
        base_dir (str): Base directory for saving results.
        model_type (str): Type of model being used.
        xai_type (str): Type of XAI method being used.
    """
    adv_folder = os.path.join(base_dir, "auswertung", model_type, xai_type, experiment)
    for cat, image_list in categories.items():
        output_dir = os.path.join(adv_folder, str(pct), "test", cat)
        os.makedirs(output_dir, exist_ok=True)

        for imagename in image_list, desc=f"Processing {cat} ({pct}%)", unit="image":
            try:
                # Load and normalize images
                current_img = normalize_image(np.array(Image.open(os.path.join(images_path, cat, imagename))))
                current_background = normalize_image(np.array(Image.open(os.path.join(background_dir, cat, imagename))))
                xai_mask = np.load(os.path.join(xai_dir, cat, "grad_mask", f"{imagename}.npy"))

                # Generate mask based on experiment type
                adv_mask = normalize_image(get_percentage_of_image(np.ones_like(current_img), xai_mask, (pct / 10)))
                if experiment == "revelation":
                    adv_example = np.where(adv_mask == 1, current_img, current_background)
                elif experiment == "occlusion":
                    adv_example = np.where(adv_mask == 0, current_img, current_background)
                else:
                    raise ValueError(f"Undefined experiment: {experiment}")

                # Save the processed image
                adv_example_save = Image.fromarray((adv_example * 255).astype("uint8"))
                adv_example_save.save(os.path.join(output_dir, imagename))
            except Exception as e:
                print(f"Error processing {imagename} in {cat}: {e}")

def main():
    """Main function to process all images for experiments."""
    parser = ArgumentParser(description="Process images for revelation and occlusion experiments.")
    parser.add_argument("--model_types", nargs="+", default=["simple_cnn", "convnext_tiny", "resnet50", "vgg16"], help="List of model types being used.")
    parser.add_argument("--xai_types", nargs="+", default=["gradcam", "lime", "prism", "xrai", "igf"], help="List of XAI methods being used.")
    parser.add_argument("--base_dir", type=str, default="data/", help="Base directory for data.")
    parser.add_argument("--dataset", type=str, default="atsds_large", help="Name of the dataset.")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split (e.g., 'test').")
    parser.add_argument("--pct_start", type=int, default=10, help="Starting percentage for experiments.")
    parser.add_argument("--pct_end", type=int, default=101, help="Ending percentage for experiments.")
    parser.add_argument("--pct_step", type=int, default=10, help="Step size for percentage values.")
    args = parser.parse_args()

    # Define paths
    ground_truth_dir = os.path.join(args.base_dir, f"{args.dataset}_mask", args.dataset_split)
    background_dir = os.path.join(args.base_dir, f"{args.dataset}_background", args.dataset_split)
    dataset_dir = os.path.join(args.base_dir, args.dataset, args.dataset_split)
    images_path = os.path.join(args.base_dir, args.dataset, args.dataset_split)

    # Get image categories
    categories = get_image_categories(images_path)

    # Process images for each combination of model_type, xai_type, experiment, and percentage
    experiments = ["revelation", "occlusion"]
    for model_type in tqdm(args.model_types, desc="Model Types", unit="model"):
        for xai_type in tqdm(args.xai_types, desc="XAI Types", unit="xai"):
            xai_dir = os.path.join(args.base_dir, "auswertung", model_type, xai_type, args.dataset_split)
            for experiment in tqdm(experiments, desc="Experiments", unit="experiment"):
                for pct in tqdm(range(args.pct_start, args.pct_end, args.pct_step), desc="Percentages", unit="pct"):
                    process_images(experiment, pct, categories, images_path, background_dir, xai_dir, args.base_dir, model_type, xai_type)

if __name__ == "__main__":
    main()
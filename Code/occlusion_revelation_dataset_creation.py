import os
import numpy as np
from PIL import Image
from utils import get_percentage_of_image, normalize_image

# Constants
MODEL_TYPE = "simple_cnn"
XAI_TYPE = "lime"
BASE_DIR = "data/"
DATASET = "atsds_large"
DATASET_SPLIT = "test"
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, f"{DATASET}_mask", DATASET_SPLIT)
BACKGROUND_DIR = os.path.join(BASE_DIR, f"{DATASET}_background", DATASET_SPLIT)
DATASET_DIR = os.path.join(BASE_DIR, DATASET, DATASET_SPLIT)
XAI_DIR = os.path.join(BASE_DIR, "auswertung", MODEL_TYPE, XAI_TYPE, DATASET_SPLIT)
IMAGES_PATH = os.path.join(BASE_DIR, DATASET, DATASET_SPLIT)

# Define categories and mappings
def get_image_categories():
    """Retrieve image categories and their corresponding file lists."""
    categories = sorted(os.listdir(IMAGES_PATH))
    return {cat: os.listdir(os.path.join(IMAGES_PATH, cat)) for cat in categories}

def process_images(experiment, pct, categories, imagedict):
    """Process images for a given experiment and percentage."""
    adv_folder = os.path.join(BASE_DIR, "auswertung", MODEL_TYPE, XAI_TYPE, experiment)
    for cat in categories:
        output_dir = os.path.join(adv_folder, str(pct), "test", cat)
        os.makedirs(output_dir, exist_ok=True)

        for imagename in imagedict[cat]:
            # Load images and mask
            current_img = normalize_image(np.array(Image.open(os.path.join(IMAGES_PATH, cat, imagename))))
            current_background = normalize_image(np.array(Image.open(os.path.join(BACKGROUND_DIR, cat, imagename))))
            xai_mask = np.load(os.path.join(XAI_DIR, cat, "grad_mask", f"{imagename}.npy"))

            # Generate mask based on experiment type
            adv_mask = normalize_image(get_percentage_of_image(np.ones_like(current_img), xai_mask, (pct / 10)))
            if experiment == "revelation":
                adv_example = np.where(adv_mask == 1, current_img, current_background)
            elif experiment == "occlusion":
                adv_example = np.where(adv_mask == 0, current_img, current_background)
            else:
                raise ValueError("Undefined Experiment!")
            # Save the processed image
            adv_example_save = Image.fromarray((adv_example * 255).astype("uint8"))
            adv_example_save.save(os.path.join(output_dir, imagename))

def main():
    """Main function to process all images for experiments."""
    experiments = ["revelation", "occlusion"]
    categories = sorted(os.listdir(IMAGES_PATH))
    imagedict = get_image_categories()

    for experiment in experiments:
        for pct in range(10, 101, 10):
            print(f"Processing {experiment}, {pct}%...")
            process_images(experiment, pct, categories, imagedict)

if __name__ == "__main__":
    main()

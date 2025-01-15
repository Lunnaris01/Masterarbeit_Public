import os
import numpy as np
from PIL import Image

# Constants
BASE_DIR = "data/"
STREET_DIR = os.path.join(BASE_DIR, "Gstreet/")
TARGET_DIR = os.path.join(BASE_DIR, "streetimages/")

def prepare_directories():
    """Ensure target directory exists."""
    os.makedirs(TARGET_DIR, exist_ok=True)

def process_and_save_images():
    """Process images by cropping and saving them to the target directory."""
    counter = 0
    for i in range(1, 3001):
        for j in range(1, 5):
            filename = f"{i:06d}_{j}"
            source_path = os.path.join(STREET_DIR, f"{filename}.jpg")
            target_path = os.path.join(TARGET_DIR, f"{counter:06d}.png")
            
            # Process image
            try:
                with Image.open(source_path) as im:
                    im_array = np.array(im)
                    cropped_image = Image.fromarray(im_array[:-24, 140:-140])
                    cropped_image.save(target_path)
                counter += 1
            except FileNotFoundError:
                print(f"Warning: File not found - {source_path}")
            except Exception as e:
                print(f"Error processing file {source_path}: {e}")

def main():
    """Main function to execute the script."""
    print("Preparing directories...")
    prepare_directories()
    print("Processing and saving images...")
    process_and_save_images()
    print("Processing complete.")

if __name__ == "__main__":
    main()

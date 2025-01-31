import torch
from collections import defaultdict 
import numpy as np
import pandas as pd
import os
import cv2
import torchvision
import torchvision.datasets as dataset
import random
from PIL import Image, ImageDraw

random.seed(42)
np.random.seed(42)

BASE_DIR = "data/"

# Load datasets
trainset = dataset.GTSRB(root=BASE_DIR, split="train", download=False)
testset = dataset.GTSRB(root=BASE_DIR, split="test", download=False)

# Define class categories
round_classes = [1, 2, 3, 4, 5, 7, 8, 9, 10, 17, 35, 38]
triangel_class = [11, 18, 25, 31]
triangel_class_reverse = [13]
stop_sign_class = [14]
rhombus_class = [12]

def transform_numpy_array(x):
    return np.array(x)

def unit_circle_filled(r):
    A = np.arange(-r, r)**2
    dists = np.sqrt(A[:, None] + A)
    return np.moveaxis(np.tile((np.abs(dists) < r).astype(int), (3, 1, 1)), 0, 2)

def get_cutout_mask(target_class, resize_size):
    if int(target_class) in round_classes:
        A = np.arange(-(resize_size // 2), resize_size // 2)**2
        dists = np.sqrt(A[:, None] + A)
        return np.moveaxis(np.tile((np.abs(dists) < (resize_size // 2)).astype(int), (3, 1, 1)), 0, 2)
    elif int(target_class) in triangel_class:
        mask = Image.new('L', (resize_size, resize_size))
        ImageDraw.Draw(mask).polygon(
            [(0, resize_size), (resize_size // 2, 0), (resize_size, resize_size), (0, resize_size)],
            outline=1,
            fill=1
        )
        return np.moveaxis(np.tile(np.array(mask).astype(int), (3, 1, 1)), 0, 2)
    elif int(target_class) in triangel_class_reverse:
        mask = Image.new('L', (resize_size, resize_size))
        ImageDraw.Draw(mask).polygon(
            [(0, 0), (resize_size, 0), (resize_size // 2, resize_size), (0, 0)],
            outline=1,
            fill=1
        )
        return np.moveaxis(np.tile(np.array(mask).astype(int), (3, 1, 1)), 0, 2)
    elif int(target_class) in rhombus_class:
        mask = Image.new('L', (resize_size, resize_size))
        ImageDraw.Draw(mask).polygon(
            [(resize_size // 2, resize_size), (0, resize_size // 2), (resize_size // 2, 0), (resize_size, resize_size // 2)],
            outline=1,
            fill=1
        )
        return np.moveaxis(np.tile(np.array(mask).astype(int), (3, 1, 1)), 0, 2)
    elif int(target_class) in stop_sign_class:
        mask = Image.new('L', (resize_size, resize_size))
        ImageDraw.Draw(mask).polygon(
            [
                (resize_size // 3, resize_size), (0, resize_size // 1.5), (0, resize_size // 3),
                (resize_size // 3, 0), (resize_size // 1.5, 0), (resize_size, resize_size // 3),
                (resize_size, resize_size // 1.5), (resize_size // 1.5, resize_size), (resize_size // 3, resize_size)
            ],
            outline=1,
            fill=1
        )
        return np.moveaxis(np.tile(np.array(mask).astype(int), (3, 1, 1)), 0, 2)

def create_directories(base_dir, classname, split):
    """Create necessary directories for train/test splits."""
    for subfolder in ["", "_mask", "_background"]:
        dir_path = f"{base_dir}atsds_large{subfolder}/{split}/{classname}"
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

def process_image_split(classname, object_areas, image_indices, background_indices, split, counter, labelfile):
    """Process images for train/test split, ensuring objects are at least 64 pixels from the edges."""
    margin = 0  # Minimum distance from the edges
    RESIZE_SIZE = 128

    for j in image_indices:
        # Load and preprocess background
        background = np.array(Image.open(BASE_DIR + "streetimages/" + str(background_indices[counter]).zfill(6) + ".png"))
        background = cv2.resize(background, dsize=(512, 512))
        backgroundcopy = background.copy()

        # Save background image
        backgroundsave = Image.fromarray(backgroundcopy)
        backgroundsave.save(BASE_DIR + f"atsds_large_background/{split}/{classname}/{str(counter).zfill(6)}.png")

        # Load and process object
        obj = np.array(object_areas[classname])[j][0].split(';')
        im_obj = np.array(Image.open(BASE_DIR + f"gtsrb/GTSRB/Training/{classname}/{obj[0]}"))
        obj_sx, obj_sy, obj_endx, obj_endy = map(int, obj[3:7])
        obj_cutout = im_obj[obj_sy:obj_endy, obj_sx:obj_endx]
        obj_cutout = cv2.resize(obj_cutout, dsize=(RESIZE_SIZE, RESIZE_SIZE))

        # Random insertion point, keeping at least a 64-pixel margin from the edges
        max_x = background.shape[1] - obj_cutout.shape[1] - margin
        max_y = background.shape[0] - obj_cutout.shape[0] - margin
        sx = np.random.randint(margin, max_x + 1)
        sy = np.random.randint(margin, max_y + 1)

        # Generate cutout mask
        cutout_mask = get_cutout_mask(classname, RESIZE_SIZE)
        insertionarea = backgroundcopy[sy:sy+obj_cutout.shape[1], sx:sx+obj_cutout.shape[0]].copy()
        insertionarea = np.where(cutout_mask != 1, insertionarea, obj_cutout)
        backgroundcopy[sy:sy+obj_cutout.shape[1], sx:sx+obj_cutout.shape[0]] = insertionarea

        # Save the composite image
        dataimg = Image.fromarray(backgroundcopy)
        dataimg.save(BASE_DIR + f"atsds_large/{split}/{classname}/{str(counter).zfill(6)}.png")

        # Generate and save the mask
        mask_background = np.zeros_like(background)
        mask_insertionarea = np.where(cutout_mask != 1, 0, 255)
        mask_background[sy:sy+obj_cutout.shape[1], sx:sx+obj_cutout.shape[0]] = mask_insertionarea
        maskimg = Image.fromarray(mask_background)
        maskimg.save(BASE_DIR + f"atsds_large_mask/{split}/{classname}/{str(counter).zfill(6)}.png")

        # Update label file
        labelfile.append([str(counter).zfill(6) + ".png", sx, sy, sx+obj_cutout.shape[1], sy+obj_cutout.shape[0], obj[-1]])
        counter += 1

    return counter

def main():
    labelcount = defaultdict(int)
    classfolders = []

    # Count labels and filter classes
    for i in trainset:
        labelcount[str(i[1])] += 1
    for amount, label in zip(labelcount.values(), labelcount.keys()):
        if amount > 500:
            classfolders.append(str(label).zfill(5))

    object_areas = {}
    for dir in classfolders:
        locations = pd.read_csv(BASE_DIR + "gtsrb/GTSRB/Training/" + dir + "/GT-" + dir + ".csv")
        object_areas[dir] = locations

    objectimageindex = random.sample(range(500), 500)
    backgroundindex = random.sample(range(len(classfolders) * (500 + 50)), len(classfolders) * (500 + 50))

    objectimageindex_train = objectimageindex[0:450]
    objectimageindex_test = objectimageindex[450:]
    backgroundindex_train = backgroundindex[0:len(classfolders) * 500]
    backgroundindex_test = backgroundindex[len(classfolders) * 500:]

    RESIZE_SIZE = 128
    BACKGROUNDS_PER_SIGN = 20

    counter_train = 0
    counter_test = 0
    labelfile = [["Filename", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId"]]

    for classname in classfolders:
        create_directories(BASE_DIR, classname, "train")
        create_directories(BASE_DIR, classname, "test")

        counter_train = process_image_split(classname, object_areas, objectimageindex_train, backgroundindex_train, "train", counter_train, labelfile)
        counter_test = process_image_split(classname, object_areas, objectimageindex_test, backgroundindex_test, "test", counter_test, labelfile)

    # Save label file
    df_labelfile = pd.DataFrame(labelfile)
    df_labelfile.to_csv(BASE_DIR + "labelfile.csv")

if __name__ == "__main__":
    main()

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
from lime import lime_image

from ATSDS import ATSDS
from model import get_model
from utils import setup_environment, prepare_categories_and_images, create_output_directories, save_xai_outputs , load_checkpoint, normalize_image


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="LIME Method Visualization Pipeline")

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
    parser.add_argument('--num_samples', type=int, default=100, help="Number of images for LIME explanation.")

    return parser.parse_args()

def batch_predict(images, model, preprocess_transform):
    """Generate model predictions."""
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

def get_pil_transform():
    """Return the PIL transform for resizing and cropping."""
    transf = transforms.Compose([
        transforms.Resize((224, 224))
        #transforms.CenterCrop(224)
    ])
    return transf

def get_preprocess_transform():
    """Return the preprocessing transform for normalizing images."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    return transf

def generate_lime_visualizations(model, device, categories, imagedict, label_idx_dict, output_path, images_path, explainer, preprocess_transform):
    """Generate LIME visualizations for each image in the dataset."""
    for category in categories:
        images = imagedict[category]
        for image_name in images:
            with open(os.path.join(images_path, category, image_name), 'rb') as f:
                with Image.open(f) as current_image:
                    current_image_tensor = preprocess_transform(current_image)
                    current_image_tensor = current_image_tensor.to(device)
                    shape = (np.array(current_image).shape[0], np.array(current_image).shape[1])

                    # Use lambda to pass the required arguments to batch_predict
                    explanation = explainer.explain_instance(
                        np.array(get_pil_transform()(current_image)),
                        classifier_fn=lambda imgs: batch_predict(imgs, model, preprocess_transform),
                        top_labels=20,
                        hide_color=0,
                        num_samples=100
                    )

                    # Get LIME mask and image with boundaries
                    temp, mask_raw = explanation.get_image_and_mask(label_idx_dict[category],
                                                                    positive_only=False,
                                                                    num_features=1,
                                                                    hide_rest=False)
                    img_boundry = mark_boundaries(temp, mask_raw)
                    save_moi = Image.fromarray((img_boundry * 255).astype(np.uint8))
                    save_moi.save(os.path.join(output_path, category, 'mask_on_image', image_name), "PNG")

                    # Normalize and save mask
                    mask = normalize_image(F.interpolate(torch.Tensor(mask_raw).reshape(1, 1, 224, 224),
                                                         (512, 512), mode="bilinear").squeeze().squeeze().numpy())
                    np.save(os.path.join(output_path, category, 'mask', image_name), mask)

def main():
    # Parse command-line arguments
    args = parse_args()

    # Setup environment
    device = setup_environment(args.random_seed)

    # Load dataset and dataloader
    testset = ATSDS(root=args.dataset_path, split=args.dataset_split, dataset_type=args.dataset_type, transform=get_preprocess_transform())
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

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
    output_path = args.output_path + args.model_name + "/lime/test/"
    create_output_directories(output_path, categories)

    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Generate LIME visualizations
    generate_lime_visualizations(
        model, device, categories, imagedict, label_idx_dict,
        output_path, args.images_path, explainer, get_preprocess_transform()
    )

if __name__ == "__main__":
    main()

from ATSDS import ATSDS

import torch
from torchvision import transforms as transforms
## Standard libraries
import os
import json
import math
import random
import numpy as np 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
CUDA_LAUNCH_BLOCKING=1
from collections import defaultdict 
import argparse

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
#Logging
import logging
from tqdm import tqdm


transform_train = transforms.Compose(
    [transforms.Resize(256),
    transforms.RandomCrop(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
transform_test = transforms.Compose(
    [transforms.Resize(256),
    transforms.CenterCrop(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
DATASET_PATH = "/data/horse/ws/juul507e-ju_streetsigns/data/"

CHECKPOINT_PATH = "model/"



# Used for reproducability to fix randomness in some GPU calculations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

def test_model(model, testloader,criterion):
    """
    Evaluate the model on the test dataset and calculate metrics.

    Parameters:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): Loss function used for evaluation.

    Returns:
        tuple:
            - class_correct (np.ndarray): Correct predictions per class.
            - class_correct_top5 (np.ndarray): Top-5 correct predictions per class.
            - class_total (np.ndarray): Total samples per class.
            - avg_test_loss (float): Average test loss across the dataset.
    """

    model.eval()
    num_classes = len(testloader.dataset.get_classes())
    correct = torch.zeros(num_classes, dtype=torch.int64, device=device)
    correct_top5 = torch.zeros(num_classes, dtype=torch.int64, device=device)
    total = torch.zeros(num_classes, dtype=torch.int64, device=device)
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)  # Move images and labels to the GPU
            outputs = model(images)
            
            loss = criterion(outputs, labels)  # Calculate the loss
            test_loss += loss.item()  # Accumulate the loss

            _, predicted = torch.max(outputs, 1)
            _, predicted_top5 = torch.topk(outputs, 5, 1)


            for i in range(len(predicted)):
                correct[labels[i]] += (predicted[i] == labels[i])
                correct_top5[labels[i]] += (labels[i] in predicted_top5[i])
                total[labels[i]] += 1
                

    accuracy_per_class = (correct.float() / total.float())
    top5_accuracy_per_class = (correct_top5.float() / total.float())
    test_loss /= len(testloader)  # Calculate the average test loss

    print(f'Test Total Accuracy: {accuracy_per_class.mean():.2%}')
    print(f'Test Total Top-5 Accuracy: {top5_accuracy_per_class.mean():.2%}')

    #for i in range(num_classes):
    #    print(f'Class {i} Test Accuracy: {accuracy_per_class[i]:.2%}')
    #    print(f'Class {i} Test Top-5 Accuracy: {top5_accuracy_per_class[i]:.2%}')

    model.train()  # Set the model back to training mode

    return correct.cpu().numpy(), correct_top5.cpu().numpy(), total.cpu().numpy(), test_loss
    
def save_model(model, optimizer, scheduler, train_stats, epoch, filepath="model/current/model.tar"):
    """
    Save the model, optimizer, scheduler, training statistics, and epoch number to a file.

    Args:
        model (torch.nn.Module): The model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer state to be saved.
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler state to be saved.
        train_stats (dict): A dictionary containing training statistics (e.g., losses, accuracy).
        epoch (int): The current epoch number.
        filepath (str): The path where the model checkpoint will be saved.

    Returns:
        None
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Prepare the state dict
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'train_stats': train_stats,
        'epoch': epoch
    }

    try:
        torch.save(checkpoint, filepath)
        print(f"Model checkpoint saved at epoch {epoch} to {filepath}")
    except Exception as e:
        print(f"Error saving model checkpoint: {e}")



from model import get_model




# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def calculate_accuracy_and_loss(model, dataloader, criterion, device):
    correct = torch.zeros(model.num_classes, dtype=torch.int64, device=device)
    correct_top5 = torch.zeros(model.num_classes, dtype=torch.int64, device=device)
    total = torch.zeros(model.num_classes, dtype=torch.int64, device=device)
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            _, predicted_top5 = torch.topk(outputs, 5, 1)

            for i in range(len(predicted)):
                correct[labels[i]] += (predicted[i] == labels[i])
                correct_top5[labels[i]] += (labels[i] in predicted_top5[i])
                total[labels[i]] += 1

    accuracy_per_class = (correct.float() / total.float())
    top5_accuracy_per_class = (correct_top5.float() / total.float())
    avg_loss = total_loss / len(dataloader)

    return accuracy_per_class, top5_accuracy_per_class, avg_loss


def start_training(model_name, model_number, dataset_type, initial_learning_rate, batch_size, weight_decay, max_epochs):
    # Load dataset and create dataloaders
    trainset = ATSDS(root=DATASET_PATH, dataset_type=dataset_type, split="train", transform=transform_train)
    testset = ATSDS(root=DATASET_PATH, dataset_type=dataset_type, split="test", transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)

    model = get_model(model_name=model_name, n_classes=trainset.get_num_classes())
    model = model.to(device)
    loss_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=initial_learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)

    # Initialize variables for tracking
    epoch = 0
    train_loss = 0.0
    correct_train_s, correct_top5_train_s, total_train_s = [], [], []
    correct_test_s, correct_top5_test_s, total_test_s = [], [], []
    train_losses, test_losses = [], []

    # Start training loop
    while epoch < max_epochs:
        # Training step
        model.train()
        correct = torch.zeros(trainset.get_num_classes(), dtype=torch.int64, device=device)
        correct_top5 = torch.zeros(trainset.get_num_classes(), dtype=torch.int64, device=device)
        total = torch.zeros(trainset.get_num_classes(), dtype=torch.int64, device=device)

        # Use tqdm to show progress for training
        with tqdm(trainloader, desc=f"Epoch {epoch+1}/{max_epochs} Training", unit="batch") as pbar:
            for i, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                _, predicted_top5 = torch.topk(outputs, 5, 1)

                for i in range(len(predicted)):
                    correct[labels[i]] += (predicted[i] == labels[i])
                    correct_top5[labels[i]] += (labels[i] in predicted_top5[i])
                    total[labels[i]] += 1

                # Update progress bar
                pbar.set_postfix(train_loss=train_loss / (i + 1))

        accuracy_per_class = (correct.float() / total.float())
        top5_accuracy_per_class = (correct_top5.float() / total.float())
        train_losses.append(train_loss / len(trainloader))
        logger.info(f"Epoch {epoch+1} Train Loss: {train_losses[-1]:.4f}")
        logger.info(f"Train Accuracy: {accuracy_per_class.mean():.2%} | Train Top-5 Accuracy: {top5_accuracy_per_class.mean():.2%}")

        correct_train_s.append(accuracy_per_class)
        correct_top5_train_s.append(top5_accuracy_per_class)
        total_train_s.append(total)

        # Testing step
        accuracy_per_class, top5_accuracy_per_class, test_loss = calculate_accuracy_and_loss(
            model, testloader, loss_criterion, device)
        test_losses.append(test_loss)
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {accuracy_per_class.mean():.2%} | Test Top-5 Accuracy: {top5_accuracy_per_class.mean():.2%}")

        correct_test_s.append(accuracy_per_class)
        correct_top5_test_s.append(top5_accuracy_per_class)
        total_test_s.append(total)

        # Save model
        save_model(model, optimizer, scheduler,
                   [train_losses, test_losses, [correct_train_s, correct_top5_train_s, total_train_s],
                    [correct_test_s, correct_top5_test_s, total_test_s]],
                   epoch, f"Code/model/{model_name}_{model_number}_{epoch}.tar")

        # Update learning rate
        scheduler.step()
        epoch += 1

    logger.info("Training Complete!")
    
    
def main(model_name, model_number, dataset_type, learning_rate, batch_size, weight_decay, max_epochs, random_seed):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)  
    start_training(model_name, model_number, dataset_type, learning_rate, batch_size, weight_decay, max_epochs)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="vgg16", help="Name of the model")
    parser.add_argument("--model_number", type=int, default=99, help="Model number")
    parser.add_argument("--dataset_type", type=str, default="atsds_large", help="Type of dataset")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=200, help="Maximum number of epochs")
    parser.add_argument("--random_seed", type=int, default=1500, help="Random Seed for training")

    args = parser.parse_args()
    main(args.model_name, args.model_number, args.dataset_type, args.learning_rate, args.batch_size, args.weight_decay, args.max_epochs, args.random_seed)
    
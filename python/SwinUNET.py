#%% importing libraries
import os
import nibabel as nib
import numpy as np
import re
from skimage.transform import resize
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from monai.networks.nets import SwinUNETR

#%% defining the paths
# Edit these variables to match your setup
dataset_path_training = r"C:\Users\Marcus\Documents\ACDC Resources\training"
dataset_path_testing = r"C:\Users\Marcus\Documents\ACDC Resources\testing"

### Functions to load in the data ###

# Regular expression to extract the patient number and frame number from filenames
filename_pattern = re.compile(r'patient(\d+)_frame(\d+)(_gt)?\.nii\.gz')

# Function to get sorting key from the filename
def get_sort_key(filepath):
    match = filename_pattern.search(os.path.basename(filepath))
    if match:
        patient_num = int(match.group(1))
        frame_num = int(match.group(2))
        return (patient_num, frame_num)
    else:
        raise ValueError(f'Filename does not match expected pattern: {filepath}')

#%%read in data
### Read in training data ###

# Lists to hold the file paths for images and ground truths
image_file_paths_train = []
ground_truth_file_paths_train = []

# Walk through the directory and collect all relevant file paths
for root, dirs, files in os.walk(dataset_path_training):
    for file in files:
        if 'frame' in file:
            full_path = os.path.join(root, file)
            if '_gt' in file:
                ground_truth_file_paths_train.append(full_path)
            else:
                image_file_paths_train.append(full_path)

# Sort the file paths to ensure alignment
image_file_paths_train.sort(key=get_sort_key)
ground_truth_file_paths_train.sort(key=get_sort_key)

# Check to make sure each image has a corresponding ground truth
assert len(image_file_paths_train) == len(ground_truth_file_paths_train)
for img_path, gt_path in zip(image_file_paths_train, ground_truth_file_paths_train):
    assert get_sort_key(img_path) == get_sort_key(gt_path), "Mismatch between image and ground truth files"

# Load the images and ground truths into numpy arrays
# using 2 index bc not all 0 index had a gt
images_train = [resize(nib.load(path).get_fdata()[:,:,2], (224,224)) for path in image_file_paths_train]
ground_truths_train = [resize(nib.load(path).get_fdata()[:,:,2], (224,224)) for path in ground_truth_file_paths_train]

# Stack the arrays into 4D numpy arrays
images_array_train = np.stack(images_train)
ground_truths_array_train = np.stack(ground_truths_train)

print(f'Training Images array shape: {images_array_train.shape}')
print(f'Training Ground truths array shape: {ground_truths_array_train.shape}')

### Read in testing data ###

# Lists to hold the file paths for images and ground truths
image_file_paths_test = []
ground_truth_file_paths_test = []

# Walk through the directory and collect all relevant file paths
for root, dirs, files in os.walk(dataset_path_testing):
    for file in files:
        if 'frame' in file:
            full_path = os.path.join(root, file)
            if '_gt' in file:
                ground_truth_file_paths_test.append(full_path)
            else:
                image_file_paths_test.append(full_path)

# Sort the file paths to ensure alignment
image_file_paths_test.sort(key=get_sort_key)
ground_truth_file_paths_test.sort(key=get_sort_key)

# Check to make sure each image has a corresponding ground truth
assert len(image_file_paths_test) == len(ground_truth_file_paths_test)
for img_path, gt_path in zip(image_file_paths_test, ground_truth_file_paths_test):
    assert get_sort_key(img_path) == get_sort_key(gt_path), "Mismatch between image and ground truth files"

# Load the images and ground truths into numpy arrays
# using 2 index bc not all 0 index had a gt
images_test = [resize(nib.load(path).get_fdata()[:,:,2], (224,224)) for path in image_file_paths_test]
ground_truths_test = [resize(nib.load(path).get_fdata()[:,:,2], (224,224)) for path in ground_truth_file_paths_test]

# Stack the arrays into 4D numpy arrays
images_array_test = np.stack(images_test)
ground_truths_array_test = np.stack(ground_truths_test)

print(f'Test Images array shape: {images_array_test.shape}')
print(f'Test Ground truths array shape: {ground_truths_array_test.shape}')
#%%
images_train_tensor = torch.tensor(images_array_train[:, np.newaxis, ...]).float()
ground_truths_train_tensor = torch.tensor(ground_truths_array_train).float()
images_test_tensor = torch.tensor(images_array_test[:, np.newaxis, ...]).float()
ground_truths_test_tensor = torch.tensor(ground_truths_array_test).float()

# Create TensorDatasets
train_dataset = TensorDataset(images_train_tensor, ground_truths_train_tensor)
test_dataset = TensorDataset(images_test_tensor, ground_truths_test_tensor)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

#%%
def multi_class_dice_coeff(preds, targets, num_classes, smooth=1.0):
    dice = 0.0
    for class_index in range(num_classes):
        pred_inds = (preds == class_index).float()
        target_inds = (targets == class_index).float()
        intersection = (pred_inds * target_inds).sum()
        union = pred_inds.sum() + target_inds.sum()
        dice += (2. * intersection + smooth) / (union + smooth)
    return dice / num_classes

# Adjust the visualization as needed for multiclass
def plot_test_examples(loader, model, device, num_examples=3):
    model.eval()
    with torch.no_grad():
        for images, true_masks in loader:
            images, true_masks = images.to(device), true_masks.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            images_np = images.cpu().numpy()
            true_masks_np = true_masks.cpu().numpy()
            preds_np = preds.cpu().numpy()

            for idx in range(min(num_examples, images_np.shape[0])):
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(images_np[idx, 0], cmap='gray', interpolation='none')
                plt.title('Original Image')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(true_masks_np[idx], cmap='jet', interpolation='none')
                plt.title('Ground Truth Mask')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(preds_np[idx], cmap='jet', interpolation='none')
                plt.title('Predicted Mask')
                plt.axis('off')
                
                plt.show()

                if idx >= num_examples - 1:
                    break

def validate_and_visualize(model, test_loader, device):
    model.eval()
    total_dice = 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Dice coefficient for multiclass
            dice_score = multi_class_dice_coeff(preds, masks, num_classes=4)
            total_dice += dice_score.item()

    average_dice = total_dice / len(test_loader)
    print(f'Average Dice Coefficient on the test set: {average_dice:.4f}')
    
    plot_test_examples(test_loader, model, device, num_examples=3)


def train_and_validate(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs=30):
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device).long()
                outputs = model(images)
                val_loss = criterion(outputs, masks)
                running_val_loss += val_loss.item()
        validation_loss = running_val_loss / len(val_loader)
        val_losses.append(validation_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {validation_loss}')

    # Plotting training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Visualization and Dice Coefficient calculation
    validate_and_visualize(model, test_loader, device)


#%%
model = SwinUNETR(
    img_size=(224, 224),
    in_channels=1,
    out_channels=4,  # Assuming you have 4 classes including the background
    feature_size=48,
    norm_name='batch',
    spatial_dims=2
)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()  # Suitable for multiclass segmentation
train_and_validate(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs=50)
#validate_and_visualize(model, test_loader, device)

# %%

'''
Run a mamba on the ACDC dataset
'''

####################################################################################################

####################################################################################################


# Edit these variables to match your setup
dataset_path_training = '/Users/calebhallinan/Desktop/jhu/classes/deep_learning/DL_Final_Project_2024/data/ACDC/training'
dataset_path_testing = '/Users/calebhallinan/Desktop/jhu/classes/deep_learning/DL_Final_Project_2024/data/ACDC/testing'


### Import packages 

import os
import nibabel as nib
import numpy as np
import re
from skimage.transform import resize


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
images_train = [resize(nib.load(path).get_fdata()[:,:,2], (224,224), order=0, preserve_range=True, anti_aliasing=False) for path in image_file_paths_train]
ground_truths_train = [resize(nib.load(path).get_fdata()[:,:,2], (224,224), order=0, preserve_range=True, anti_aliasing=False) for path in ground_truth_file_paths_train]

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
images_test = [resize(nib.load(path).get_fdata()[:,:,2], (224,224), order=0, preserve_range=True, anti_aliasing=False) for path in image_file_paths_test]
ground_truths_test = [resize(nib.load(path).get_fdata()[:,:,2], (224,224), order=0, preserve_range=True, anti_aliasing=False) for path in ground_truth_file_paths_test]

# Stack the arrays into 4D numpy arrays
images_array_test = np.stack(images_test)
ground_truths_array_test = np.stack(ground_truths_test)

print(f'Test Images array shape: {images_array_test.shape}')
print(f'Test Ground truths array shape: {ground_truths_array_test.shape}')


####################################################################################################

####################################################################################################


### Create and train mamba ###

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from vision_mamba import MambaUnet as VIM_seg

base_lr = 0.001
num_classes = 4
batch_size = 32
max_iterations = 15
patch_size = 224


# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Initialize the model, move it to the current device, and print the summary
model = VIM_seg(img_size=patch_size, num_classes=num_classes).to(device)
print(model)

# Convert the numpy arrays to PyTorch tensors
# Add a channel dimension with 'np.newaxis' before converting to tensor
# This changes the shape from (n_images, height, width) to (n_images, 1, height, width)
images_tensor = torch.tensor(images_array_train[:, np.newaxis, ...]).float()
ground_truths_tensor = torch.tensor(ground_truths_array_train).float()

# Create a TensorDataset and DataLoader
dataset = TensorDataset(images_tensor, ground_truths_tensor)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define the optimizer and the loss function
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()  # Change the loss function to CrossEntropyLoss

# Training loop
num_epochs = 250
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        # Move tensors to the device
        images, masks = images.to(device), masks.to(device).long()  # Ensure masks are long integers

        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute the loss
        loss = criterion(outputs, masks)  # CrossEntropyLoss takes raw scores, not sigmoid outputs
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')


####################################################################################################

####################################################################################################

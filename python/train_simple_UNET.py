'''
Run a simple CNN on the ACDC dataset
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

### Create and train a simple CNN ###

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc_conv0 = ConvBlock(1, 16)
        self.pool0 = nn.MaxPool2d(2)  # 224 -> 112
        self.enc_conv1 = ConvBlock(16, 32)
        self.pool1 = nn.MaxPool2d(2)  # 112 -> 56
        
        # Bottleneck
        self.bottleneck_conv = ConvBlock(32, 64)
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 56 -> 112
        self.dec_conv1 = ConvBlock(64, 32)
        self.upconv0 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # 112 -> 224
        self.dec_conv0 = ConvBlock(32, 16)
        
        # Classifier
        # self.classifier = nn.Conv2d(16, 1, kernel_size=1)
        self.classifier = nn.Conv2d(16, 4, kernel_size=1)  # Change the number of output channels to 4


    def forward(self, x):
        # Encoder
        e0 = self.enc_conv0(x)
        e1 = self.pool0(e0)
        
        e1 = self.enc_conv1(e1)
        e2 = self.pool1(e1)
        
        # Bottleneck
        b = self.bottleneck_conv(e2)
        
        # Decoder
        d1 = self.upconv1(b)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec_conv1(d1)
        
        d0 = self.upconv0(d1)
        d0 = torch.cat((d0, e0), dim=1)
        d0 = self.dec_conv0(d0)
        
        # Classifier
        out = self.classifier(d0)
        return out


# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Initialize the model, move it to the current device, and print the summary
model = UNet().to(device)
print(model)

# Convert the numpy arrays to PyTorch tensors
# Add a channel dimension with 'np.newaxis' before converting to tensor
# This changes the shape from (n_images, height, width) to (n_images, 1, height, width)
images_tensor = torch.tensor(images_array_train[:, np.newaxis, ...]).float()
ground_truths_tensor = torch.tensor(ground_truths_array_train).float()

# Create a TensorDataset
dataset = TensorDataset(images_tensor, ground_truths_tensor)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Define the optimizer and the loss function
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()  # Using CrossEntropyLoss suitable for multi-class segmentation

# Example usage of DataLoader in a training loop could be adjusted as shown in previous messages to handle training and validation phases.


# Training loop
num_epochs = 50
training_losses = []
validation_losses = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device).long()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    training_losses.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device).long()
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    validation_losses.append(avg_val_loss)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')


####################################################################################################

####################################################################################################


import matplotlib.pyplot as plt

# plot the training and validation losses over epochs
plt.figure(figsize=(10, 5))
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.grid(True)
plt.show()



####################################################################################################

####################################################################################################



### Evaluate the model on the test set ###

# Assuming test_images_array and test_ground_truths_array are your test data prepared in the same way as your training data
test_images_tensor = torch.tensor(images_array_test[:, np.newaxis, ...]).float()
test_ground_truths_tensor = torch.tensor(ground_truths_array_test).float()

# Create a TensorDataset and DataLoader for the test set
test_dataset = TensorDataset(test_images_tensor, test_ground_truths_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Convert outputs using softmax and get the predictions
def multi_class_dice_coeff(preds, targets, num_classes, smooth=1.0):
    dice = 0.0
    for class_index in range(num_classes):
        pred_inds = (preds == class_index).float()
        target_inds = (targets == class_index).float()
        intersection = (pred_inds * target_inds).sum()
        union = pred_inds.sum() + target_inds.sum()
        dice += (2. * intersection + smooth) / (union + smooth)
    return dice / num_classes


model.eval()  # Set the model to evaluation mode
total_dice = 0.0
with torch.no_grad():  # Disable gradient calculation
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # Calculate the Dice coefficient for each class and average them
        dice_score = multi_class_dice_coeff(preds, masks, num_classes=4)
        total_dice += dice_score.item()

average_dice = total_dice / len(test_loader)
print(f'Average Dice Coefficient on the test set: {average_dice:.4f}')




####################################################################################################

####################################################################################################

### Visualize the model predictions ###

import matplotlib.pyplot as plt

def plot_test_examples(loader, model, device, num_examples=3):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        for images, true_masks in loader:
            # Move tensors to the device
            images, true_masks = images.to(device), true_masks.to(device)
            
            # Forward pass to get outputs
            outputs = model(images)
            
            # Convert outputs to probability scores and predictions
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Convert tensors to numpy arrays for plotting
            images_np = images.cpu().numpy()
            true_masks_np = true_masks.cpu().numpy()
            preds_np = preds.cpu().numpy()

            # Plotting the first 'num_examples' in the batch
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
                break  # Stop after showing 'num_examples' examples

# Usage: Assuming you have already defined 'test_loader', 'model', and 'device'
plot_test_examples(test_loader, model, device, num_examples=2)





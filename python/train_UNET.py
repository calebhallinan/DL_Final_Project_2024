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

### Create and train an original UNET ###

import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim



class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3
        self.e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1) # output: 570x570x64 # CH: changed input channels to 1
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 138x138x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66x66x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 30x30x1024
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 28x28x1024


        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out



# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Initialize the model, move it to the current device, and print the summary
model = UNet(n_class = 4).to(device)
print(model)

# Convert the numpy arrays to PyTorch tensors
# Add a channel dimension with 'np.newaxis' before converting to tensor
# This changes the shape from (n_images, height, width) to (n_images, 1, height, width)
images_tensor = torch.tensor(images_array_train[:, np.newaxis, ...]).float()
ground_truths_tensor = torch.tensor(ground_truths_array_train).float()
images_tensor.size()

# Create a TensorDataset and DataLoader
dataset = TensorDataset(images_tensor, ground_truths_tensor)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define the optimizer and the loss function
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()  # Change the loss function to CrossEntropyLoss

# Training loop
num_epochs = 50
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





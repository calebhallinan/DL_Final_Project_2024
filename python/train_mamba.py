'''
Run a mamba on the ACDC dataset
'''


####################################################################################################

####################################################################################################

# Edit these variables to match your setup
dataset_path_training = '/home/caleb/Desktop/projects_caleb/data/ACDC/training'
dataset_path_testing = '/home/caleb/Desktop/projects_caleb/data/ACDC/testing'

### Import packages 

import os
import nibabel as nib
import numpy as np
import re
from skimage.transform import resize
import configparser
import pandas as pd


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
    
    # Function to extract the patient number and sort by it
def extract_patient_number(file_path):
    match = re.search(r"patient(\d+)", file_path)
    return int(match.group(1)) if match else None

### Read in training data ###

# Lists to hold the file paths for images and ground truths
image_file_paths_train = []
ground_truth_file_paths_train = []
class_file_paths_train = []

# Walk through the directory and collect all relevant file paths
for root, dirs, files in os.walk(dataset_path_training):
    for file in files:
        if 'frame' in file:
            full_path = os.path.join(root, file)
            if '_gt' in file:
                ground_truth_file_paths_train.append(full_path)
            else:
                image_file_paths_train.append(full_path)
        if "Info" in file:
            class_file_paths_train.append(os.path.join(root, file))


# Sort the file paths to ensure alignment
image_file_paths_train.sort(key=get_sort_key)
ground_truth_file_paths_train.sort(key=get_sort_key)
class_file_paths_train = sorted(class_file_paths_train, key=extract_patient_number)

# Check to make sure each image has a corresponding ground truth
assert len(image_file_paths_train) == len(ground_truth_file_paths_train)
for img_path, gt_path in zip(image_file_paths_train, ground_truth_file_paths_train):
    assert get_sort_key(img_path) == get_sort_key(gt_path), "Mismatch between image and ground truth files"

# Extract the class labels from the config files
class_labels_train = []  
for class_file in class_file_paths_train:
        config = pd.read_csv(class_file, sep=':', header=None)
        class_labels_train.append(config[config[0] == "Group"][1][2].strip())
        class_labels_train.append(config[config[0] == "Group"][1][2].strip()) # doing twice bc there are 2 files per patient

# Load the images and ground truths into numpy arrays
# using 2 index bc not all 0 index had a gt
images_train = [resize(nib.load(path).get_fdata()[:,:,2], (224,224)) for path in image_file_paths_train]
ground_truths_train = [resize(nib.load(path).get_fdata()[:,:,2], (224,224)) for path in ground_truth_file_paths_train]

# Stack the arrays into 4D numpy arrays
images_array_train = np.stack(images_train)
ground_truths_array_train = np.stack(ground_truths_train)

print(f'Training Images array shape: {images_array_train.shape}')
print(f'Training Ground truths array shape: {ground_truths_array_train.shape}')
print(f'Class labels: {class_labels_train}', '\n', "Total Class Labels: ",len(class_labels_train))


### Read in testing data ###

# Lists to hold the file paths for images and ground truths
image_file_paths_test = []
ground_truth_file_paths_test = []
class_file_paths_test = []

# Walk through the directory and collect all relevant file paths
for root, dirs, files in os.walk(dataset_path_testing):
    for file in files:
        if 'frame' in file:
            full_path = os.path.join(root, file)
            if '_gt' in file:
                ground_truth_file_paths_test.append(full_path)
            else:
                image_file_paths_test.append(full_path)
        if "Info" in file:
            class_file_paths_test.append(os.path.join(root, file))

# Sort the file paths to ensure alignment
image_file_paths_test.sort(key=get_sort_key)
ground_truth_file_paths_test.sort(key=get_sort_key)
class_file_paths_test = sorted(class_file_paths_test, key=extract_patient_number)

# Check to make sure each image has a corresponding ground truth
assert len(image_file_paths_test) == len(ground_truth_file_paths_test)
for img_path, gt_path in zip(image_file_paths_test, ground_truth_file_paths_test):
    assert get_sort_key(img_path) == get_sort_key(gt_path), "Mismatch between image and ground truth files"

# Extract the class labels from the config files
class_labels_test = []
for class_file in class_file_paths_test:
        config = pd.read_csv(class_file, sep=':', header=None)
        class_labels_test.append(config[config[0] == "Group"][1][2].strip())
        class_labels_test.append(config[config[0] == "Group"][1][2].strip())

    
# Load the images and ground truths into numpy arrays
# using 2 index bc not all 0 index had a gt
images_test = [resize(nib.load(path).get_fdata()[:,:,2], (224,224)) for path in image_file_paths_test]
ground_truths_test = [resize(nib.load(path).get_fdata()[:,:,2], (224,224)) for path in ground_truth_file_paths_test]

# Stack the arrays into 4D numpy arrays
images_array_test = np.stack(images_test)
ground_truths_array_test = np.stack(ground_truths_test)

print(f'Test Images array shape: {images_array_test.shape}')
print(f'Test Ground truths array shape: {ground_truths_array_test.shape}')
print(f'Class labels: {class_labels_test}', '\n', "Total Class Labels: ",len(class_labels_test))



####################################################################################################

####################################################################################################


### Create and train mamba ###

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,random_split
from losses import DiceLoss
# from vision_mamba import MambaUnet as VIM_seg


# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Initialize the model, move it to the current device, and print the summary
model = MambaUnet(img_size = 224, num_classes=4).to(device)
model.load_from("/home/caleb/Desktop/projects_caleb/python/vmamba_tiny_e292.pth")
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
# optimizer = optim.SGD(model.parameters(), lr=1e-3,
#                           momentum=0.9, weight_decay=0.0001)
criterion = DiceLoss(n_classes=4)  # Using CrossEntropyLoss suitable for multi-class segmentation
ce_loss = nn.CrossEntropyLoss()
dice_loss = DiceLoss(n_classes=4)

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
        # loss_ce = ce_loss(outputs, masks)
        # loss_dice = dice_loss(outputs, masks)
        # loss = .5 * (loss_dice + loss_ce)
        
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
            # loss_ce = ce_loss(outputs, masks)
            # loss_dice = dice_loss(outputs, masks)
            # loss = .5 * (loss_dice + loss_ce)
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



####################################################################################################

####################################################################################################



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from mamba_sys import VSSM



logger = logging.getLogger(__name__)

class MambaUnet(nn.Module):
    def __init__(self, img_size=224, num_classes=4, zero_head=False, vis=False):
        super(MambaUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        # self.mamba_unet =  VSSM(
        #                         patch_size=4,
        #                         in_chans=3,
        #                         num_classes=self.num_classes,
        #                         embed_dim=96,
        #                         depths=[2, 2, 9, 2],
        #                         mlp_ratio=4.,
        #                         drop_rate=0.1,
        #                         drop_path_rate=0.0,
        #                         patch_norm=True,
        #                         use_checkpoint=False)
        self.mamba_unet =  VSSM(
                                in_chans=3,
                                num_classes=self.num_classes,
                                drop_rate=0.2, 
                                attn_drop_rate=0.2,
                                )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.mamba_unet(x)
        return logits

    def load_from(self, path): # edit from config to path
        pretrained_path = path # edit from config to path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.mamba_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.mamba_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.mamba_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")


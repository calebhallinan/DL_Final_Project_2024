'''
Run a method to classify and segment images
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
import configparser
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F


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


# Convert the numpy arrays to PyTorch tensors
# Add a channel dimension with 'np.newaxis' before converting to tensor
# This changes the shape from (n_images, height, width) to (n_images, 1, height, width)
images_tensor = torch.tensor(images_array_train[:, np.newaxis, ...]).float()
images_test_tensor = torch.tensor(images_array_test[:, np.newaxis, ...]).float()

ground_truths_tensor = torch.tensor(ground_truths_array_train).float()
# list of class labels
class_labels_list = ['NOR', 'MINF', 'DCM', 'HCM', 'RV']
# Create a mapping from class labels to integers
label_to_int = {label: idx for idx, label in enumerate(sorted(set(class_labels_train)))}
# Convert list of class labels to integers based on the mapping
class_labels_int = [label_to_int[label] for label in class_labels_train]
class_labels_tensor = torch.tensor(class_labels_int).long()  # Convert class labels to a tensor



####################################################################################################

####################################################################################################


### Train a simple perceptron model ###
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Flatten the images and ground truths
X_train = images_array_train.reshape(images_array_train.shape[0], -1)
y_train = np.array(class_labels_train)
X_train.shape, y_train.shape

# Flatten the images and ground truths
X_test = images_array_test.reshape(images_array_test.shape[0], -1)
y_test = np.array(class_labels_test)
X_test.shape, y_test.shape

# Create the perceptron model
perceptron = Perceptron()

# Train the model
perceptron.fit(X_train, y_train)

# Make predictions
y_pred_train = perceptron.predict(X_train)
y_pred_test = perceptron.predict(X_test)

# Calculate accuracy
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

print(f'Training accuracy: {accuracy_train}')
print(f'Test accuracy: {accuracy_test}')



####################################################################################################

####################################################################################################


# set the seed for reproducibility
torch.manual_seed(0)


# Define the Perceptron model
class Perceptron(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc(x)
        return out


# Define multi-layer perceptron model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size*2)
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out


# Constants for the model
input_size = 224 * 224 
num_classes = 5  
hidden_size = 10000

# Create the model instance
model = Perceptron(input_size, num_classes)
# model = MLP(input_size, hidden_size, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

# Create a TensorDataset
dataset = TensorDataset(images_tensor, class_labels_tensor)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


import matplotlib.pyplot as plt

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for images, labels in train_loader:
            # Flatten images
            images = images.view(-1, 224*224)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.view(-1, 224*224)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        # Calculate average validation loss for the epoch
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Logging the losses
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    # Plotting the losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer)



####################################################################################################

####################################################################################################


import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

# Assuming test_images_array and test_ground_truths_array are your test data prepared in the same way as your training data
test_images_tensor = torch.tensor(images_array_test[:, np.newaxis, ...]).float()
test_ground_truths_tensor = torch.tensor(ground_truths_array_test).float()
# Create a mapping from class labels to integers
label_to_int = {label: idx for idx, label in enumerate(sorted(set(class_labels_test)))}
# Convert list of class labels to integers based on the mapping
class_labels_int = [label_to_int[label] for label in class_labels_test]
class_labels_test_tensor = torch.tensor(class_labels_int).long()  # Convert class labels to a tensor

# Create a dictionary to map class labels to integers
label_to_int = {label: i for i, label in enumerate(np.unique(class_labels_train))}


# Create a TensorDataset and DataLoader for the test set
test_dataset = TensorDataset(test_images_tensor, class_labels_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # for a nicer confusion matrix visualization

# Function to evaluate the model on the test dataset and plot a confusion matrix
def evaluate_model(model, test_loader, num_classes):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 224*224)  # Flatten the images
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect all predictions and true labels
            all_predicted.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predicted)
    print("Confusion Matrix:\n", cm)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=label_to_int.keys(), yticklabels=label_to_int.keys())
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Number of classes, assuming they are labeled from 0 to num_classes-1
num_classes = 5

# Evaluate the model
evaluate_model(model, test_loader, num_classes)



####################################################################################################

####################################################################################################



# create confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Convert the class labels to integers
conf_matrix = confusion_matrix(y_test, y_pred_test)

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_to_int.keys(), yticklabels=label_to_int.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()



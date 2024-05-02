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


#==================================
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
#==================================

# load the image
nii_img  = nib.load('/Users/calebhallinan/Desktop/jhu/classes/deep_learning/project/data/patient101_frame01.nii.gz')
nii_data = nii_img.get_fdata()

# load in the ground truth
nii_img1  = nib.load('/Users/calebhallinan/Desktop/jhu/classes/deep_learning/project/data/patient101_frame01_gt.nii.gz')
nii_data1 = nii_img1.get_fdata()

# plot them on top of each other
plt.imshow(nii_data[:,:,8], cmap='gray', interpolation=None)
plt.imshow(nii_data1[:,:,8], cmap='jet', interpolation=None, alpha=0.55)


nii_data.shape






#==================================




import os
import nibabel as nib
import numpy as np
import re
from skimage.transform import resize


# The path where the dataset is located
dataset_path = '/Users/calebhallinan/Desktop/jhu/classes/deep_learning/DL_Final_Project_2024/data/ACDC/training'

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
    


# Lists to hold the file paths for images and ground truths
image_file_paths_train = []
ground_truth_file_paths_train = []

# Walk through the directory and collect all relevant file paths
for root, dirs, files in os.walk(dataset_path):
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




# The path where the dataset is located
dataset_path = '/Users/calebhallinan/Desktop/jhu/classes/deep_learning/DL_Final_Project_2024/data/ACDC/testing'

# Regular expression to extract the patient number and frame number from filenames
filename_pattern = re.compile(r'patient(\d+)_frame(\d+)(_gt)?\.nii\.gz')


# Lists to hold the file paths for images and ground truths
image_file_paths_test = []
ground_truth_file_paths_test = []

# Walk through the directory and collect all relevant file paths
for root, dirs, files in os.walk(dataset_path):
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


#==========================================================================================================
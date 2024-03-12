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
plt.imshow(nii_data[:,:,0], cmap='gray', interpolation=None)
plt.imshow(nii_data1[:,:,1], cmap='jet', interpolation=None, alpha=0.55)


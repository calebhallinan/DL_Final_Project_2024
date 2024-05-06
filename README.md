# DL_Final_Project_2024
Final Project for Deep Learning 2024
# Datasert
The ACDC MRI image set is composed of 2D cine MRI images from patients with the following phenotypes: hypertrophic cardiomyopathy, dilated cardiomyopathy, abnormal right ventricle, myocardial infarction, and normal. Cardiac MRI images were taken from the short axis, horizontal long axis, and vertical long axis providing a holistic view of the heart for segmentation and classification training. Image sets for individual patients are composed of multiple planar views with a single plane composed of several z-stack images taken at various levels of heart depth.
# Approach

The first objective of the project was segmenting the left and right ventricular endocardium and epicardium, facilitating a comparative analysis with traditional techniques and outcomes. To achieve this, various segmentation algorithms were tested – Reduced UNet, Classic UNet, SwinUNet, UNet++, SegNet, FCDenseNet, and Mamba-UNet.

The second objective of the study was to classify the five specific heart conditions using relevant features. We achieved this task using a basic perceptron model, multilayer perceptron, basic perceptron using HOG features, kNN, and SVM classifiers. We also performed simultaneous segmentation and classification tasks using UNet, UNet++, and Mamba-UNet, making it suitable for applications requiring both spatial and categorical understanding of the data. In the segmentation process, the input passes through a series of convolutional layers and pooling operations in the encoder to extract hierarchical features. The decoder up samples features and concatenates them with skip connections to reconstruct the segmentation mask. In the classification process of this task, features from the deepest layer of the segmentation encoder are used for classification. Global average pooling compresses feature maps into a vector, followed by fully connected layers for classification.

# Results
Model	Segmentation: Dice
Reduced UNet	0.2622
Standard UNet	0.7649
SwinUNet	0.7830
UNet++	0.8050
SegNet	0.8369
FCDenseNet	0.6597
TransUNet	0.7446
Standard UNet Fine-tuned	0.9316
Mamba-UNet Fine-tuned	0.9046
![image](https://github.com/nidhisoley/DL_Final_Project_2024/assets/71967651/012f76dd-c37f-4ee4-a3ff-2c2c4f742e63)

Multi-task Learning Performance:
Fine tuned Mamba-UNet excelled in the segmentation task when combined with the classifier, with an average DICE coefficient of 0.8670, but had a poor classification accuracy of 0.27. The fine-tuned UNet with classification also showed respectable performance, with an average DICE of 0.8446 and a classification accuracy of 0.35. (Table 2). The standard UNet and UNet++ were also tested by including the classification loss to train the model but did not perform as strongly with an average DICE of 0.6345 and 0.6698 and classification accuracy of 0.34 and 0.36, respectively.






# AMLS_Assignment_22-23

## Initialising the conda environemt

The conda environement can be created/updated using the command below:
'''
conda env update --name ml-final --file environment.yml --prune   
'''

When this process if finished activate the environement with the command:
'''
conda activate ml-final  
'''

## Folder structure

This Github repository has the folder structure seen below:

->A1                    # Code for task A1
->A2                    # Code for task A2
->B1                    # Code for task B1
->B2                    # Code for task B2
->Datasets              # Empty folder for datasets
->.gitignore            
->environment.yml       # Conda environment file
->main.py               # Main file for running each task
->README.md

### A1
- gender.py: This file calls the feature extraction functions for this task and trains the model on the training data.
It prints the classification report, accuracy and predictions of the model

- landmarks.py: This file extracts faces and key feature points from each image. It is based on the AMLS Lab 6 file and this [paper](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)


### A2
- emotion.py: This file calls the feature extraction functions for this task and trains the model on the training data.
It prints the classification report, accuracy and predictions of the model

- a2_landmarks.py: This file extracts faces and key feature points from each image.


### B1
- face.py: This file calls the feature extraction functions for this task and trains the model on the training data.
It prints the confustion matrix, accuracy and loss graphs for the model.

- b1_feature_extract.py: This file extracts faces and key feature points from each image.

### B2
- eye.py: This file calls the feature extraction functions for this task and trains the model on the training data.
It prints the confustion matrix, accuracy and loss graphs for the model.

- b2_feature_extract.py: This file extracts faces and key feature points from each image.
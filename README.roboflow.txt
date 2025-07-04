
My First Project - v13 2025-05-20 11:09pm
==============================

This dataset was exported via roboflow.com on May 20, 2025 at 8:10 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 5380 images.
Objects are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Fit (black edges))
* Grayscale (CRT phosphor)
* Auto-contrast via histogram equalization

The following augmentation was applied to create 3 versions of each source image:
* Randomly crop between 0 and 26 percent of the image
* Random brigthness adjustment of between -20 and +20 percent
* Random Gaussian blur of between 0 and 2.7 pixels
* Salt and pepper noise was applied to 1.37 percent of pixels



# Exploring GAN Variants for Balancing Imbalanced Datasets

Course    : Special Topics in Artificial Intelligence  
Instructor: Dr. Yousef Sanjalawe  
Student   : Fares Hatahet  
Semester  : Spring 2024/2025  

--------------------------------------------------------

Overview

This project explores the use of Generative Adversarial Networks (GANs) to balance imbalanced datasets, specifically focusing on the **FashionMNIST** dataset.
Various GAN variants—Vanilla GAN, DCGAN, and WGAN—are implemented from scratch using Pytorch and used to generate synthetic data for minority classes.
The project includes comparative evaluation using classification performance on both imbalanced and GAN-balanced datasets.

--------------------------------------------------------

Directory Structure

├── saved_models/                                  # Generator and Discriminator classes as .py files
│   ├── DCGAN.py
│   ├── vanilla.py
│   └── WGAN.py
├── training/                                      # Main training scripts and notebooks
│   ├── Classifier_multiclass_extra_experiment.ipynb   # Classifier on all classes (GAN-balanced vs. imbalanced)
│   ├── DCGAN_fashion_mnist.ipynb                      # DCGAN training and balancing
│   ├── imbalancer.py                                  # Utility to imbalance a class by keep ratio
│   ├── Main_classifier_binary.ipynb                   # Classifier on two classes (0 vs 1) across datasets
│   ├── Vanilla_GAN_fashion_mnist.ipynb                # Vanilla GAN training and balancing
│   └── WGAN_fashion_mnist.ipynb                       # WGAN training and balancing

--------------------------------------------------------

Requirements:
Install the dependencies provided in the requirements file (reqs.txt) if you have a GPU capable of training the model.
Or it can be run in Google Colab.

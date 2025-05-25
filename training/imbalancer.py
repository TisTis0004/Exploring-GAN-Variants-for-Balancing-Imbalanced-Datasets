import torch
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import os


class Imbalancer:
    """
    A class to simulate class imbalance in a dataset by downsampling a specified minority class.
    """

    def __init__(self, data, labels, minor_class, keep_ratio, output_path):
        """
        Initialize the Imbalancer with data, labels, the target minority class, retention ratio, and save path.

        Parameters:
        - data (np.ndarray): Array of input images.
        - labels (list or np.ndarray): Corresponding class labels.
        - minor_class (int): The class to be reduced.
        - keep_ratio (float): Fraction of minority class instances to retain.
        - output_path (str): Directory path to save the resulting imbalanced dataset.
        """
        self.data = data
        self.labels = np.array(labels)
        self.minor_class = minor_class
        self.keep_ratio = keep_ratio
        self.output_path = output_path

    def imbalance(self):
        """
        Apply imbalance by reducing the number of instances in the specified minority class.
        Shuffles and stores the resulting data and labels as PyTorch tensors.
        """
        new_data = []
        new_labels = []

        for cls in np.unique(self.labels):
            cls_indices = np.where(self.labels == cls)[0]
            np.random.shuffle(cls_indices)
            if cls == self.minor_class:
                keep_count = int(len(cls_indices) * self.keep_ratio)
            else:
                keep_count = len(cls_indices)

            selected_indices = cls_indices[:keep_count]

            new_data.append(self.data[selected_indices])
            new_labels.append(self.labels[selected_indices])

        new_data = np.concatenate(new_data)
        new_labels = np.concatenate(new_labels)

        shuffle_idx = np.random.permutation(len(new_labels))
        new_data = new_data[shuffle_idx]
        new_labels = new_labels[shuffle_idx]

        if new_data.shape[-1] == 3:
            new_data = np.transpose(new_data, (0, 3, 1, 2))

        self.imbalanced_data = torch.tensor(new_data)
        self.imbalanced_labels = torch.tensor(new_labels)

        self.class_distribution = dict(Counter(new_labels))
        return TensorDataset(self.imbalanced_data, self.imbalanced_labels)

    def visualize(self):
        """
        Visualize class distribution before and after applying imbalance using a Seaborn count plot.
        """
        if not hasattr(self, 'imbalanced_labels'):
            raise ValueError("Run `create_imbalance()` before visualizing the post-imbalance distribution.")

        df_before = pd.DataFrame({'Class': self.labels, 'State': 'Before'})
        df_after = pd.DataFrame({'Class': self.imbalanced_labels.numpy(), 'State': 'After'})
        df = pd.concat([df_before, df_after])

        plt.figure(figsize=(14, 5))
        sns.countplot(x='Class', hue='State', data=df, palette='flare')
        plt.title("Class Distribution Before and After Imbalance")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def save(self):
        """
        Save the imbalanced dataset (images and labels) as PyTorch files to the specified output path.
        """
        os.makedirs(self.output_path, exist_ok=True)
        torch.save(self.imbalanced_data, os.path.join(self.output_path, 'data.pt'))
        torch.save(self.imbalanced_labels, os.path.join(self.output_path, 'labels.pt'))
        print(f"Saved imbalanced dataset to {self.output_path}")
        print(f"Class distribution: {self.class_distribution}")


# Testing the class
if __name__ == "__main__":
    transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.FashionMNIST(root="./", train=True, download=True, transform=transform)
    images = np.stack([np.array(img) for img, _ in dataset])
    labels = [label for _, label in dataset]
    minor_class = 1
    imb = Imbalancer(images, labels, minor_class=minor_class, keep_ratio=0.2, output_path="../imbalanced_dataset_fashion_mnist")
    dataset = imb.imbalance()
    imb.visualize()
    imb.save()


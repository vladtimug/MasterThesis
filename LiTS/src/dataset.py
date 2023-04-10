import glob
import torch
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T

imagenet_mean = np.mean(np.array([0.485, 0.456, 0.406]))
imagenet_std = np.mean(np.array([0.229, 0.224, 0.225]))

def get_transforms():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(
            [imagenet_mean],
            [imagenet_std]
        )
    ])

class LiTSDataset(Dataset):
    """
    Summary:
    ----
    Dataset class to retrieve samples from the the LiTS dataset one sample at the time
    """

    def __init__(self, rootDataDirectory:str, datasetSplit:str, device:str) -> None:
        """
        Summary:
        ----
        Prepare access to dataset

        Args:
        ----
            * rootDataDirectory (str): Path to the root directory of the dataset
            * datasetSplit (str): Type of split to use from the available data. Available options are "train", (add more options)
            * device (str): Device to load the dataset object on. Available options are "cuda" and "cpu".
        """
        # Initialize class members from constructor arguments
        self.root_data_directory = rootDataDirectory
        self.split = datasetSplit
        self.device = device

        # Initialize additional class memebers
        samples_directory_name = "images"
        annotations_directory_name = "labels"
        self.samples_paths = glob.glob(self.root_data_directory + self.split + "/" + samples_directory_name + "/liver*/*.bmp")
        self.annotations_paths = glob.glob(self.root_data_directory + self.split + "/" + annotations_directory_name + "/liver*/*.bmp")
        
        self.samples_count = len(self.samples_paths)
        self.annotations_count = len(self.annotations_paths)
        
        if self.samples_count != self.annotations_count:
            raise ValueError("Different number of files in the samples and annotations directories.")

    def __len__(self) -> int:
        """
        Summary:
        ----
        Return dataset size

        Returns:
        ----
            * int: Number of available samples in the dataset. Consider the dataset to have an identical number of annotations
        """
        return self.samples_count
    
    def __getitem__(self, index) -> 'tuple[torch.Tensor, torch.Tensor]':
        """
        Summary:
        ----
            Return a sample and its corresponding annotation from the dataset

        Args:
        ----
            * index (int): Index of the sample to access from the dataset
        
        Raises:
        ----
            * IndexError: Error if passed index is smaller than 0 or bigger than the maximum available number of samples

        Returns:
        ----
            * tuple[torch.Tensor, torch.Tensor]: Tuple containing a dataset sample and its corresponding annotation
        """

        # Check if the index lies within a valid range
        if index < 0 or index > self.samples_count - 1:
            raise IndexError(f"Cannot access sample with index {index} because it is outside of the [0, {self.samples_count}) interval.")

        # Read and return sample and annotation from index
        sample_path = self.samples_paths[index]
        annotation_path = self.annotations_paths[index]

        raw_sample = cv.imread(sample_path, cv.IMREAD_UNCHANGED)
        raw_annotation = cv.imread(annotation_path, cv.IMREAD_UNCHANGED)

        return raw_sample, raw_annotation
    
    def collate_fn(self, batch):
        images, masks = list(zip(*batch))

        images = torch.cat([get_transforms()(image.copy()/255.)[None] for image in images]).float().to(self.device)
        masks = torch.cat([torch.Tensor(mask[None]) for mask in masks]).long().to(self.device)

        return images, masks

    def get_sample_and_annotation_paths(self, index):
        # Check if the index lies within a valid range
        if index < 0 or index > self.samples_count - 1:
            raise IndexError(f"Cannot access sample with index {index} because it is outside of the [0, {self.samples_count}) interval.")

        # Read and return sample and annotation from index
        sample_path = self.samples_paths[index]
        annotation_path = self.annotations_paths[index]
        
        return sample_path, annotation_path
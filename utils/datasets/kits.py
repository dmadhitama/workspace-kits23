import torch
from torch.utils.data import Dataset
import re
import os
import numpy as np
import nibabel as nib

class Kits23Dataset(Dataset):
    def __init__(self, dataset_dir, transform=None, target_transform=None):
        self.nii_gz_dict, self.labels = self._get_nii_gz_data(dataset_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.nii_gz_dict.keys())
    
    def _filter_dataset(self, file_list):
        filtered_files = []
        seen_prefixes = set()

        for file in file_list:
            # Skip files without "annotation"
            if "annotation" not in file:
                filtered_files.append(file)
                continue

            # Extract the prefix (e.g., "tumor_instance-1")
            match = re.search(r'(.*?)_annotation', file)
            if match:
                prefix = match.group(1)

                # Add the file if the prefix hasn't been seen before
                if prefix not in seen_prefixes:
                    filtered_files.append(file)
                    seen_prefixes.add(prefix)
        return filtered_files

    def _get_nii_gz_data(self, directory):
        """
        Finds all .nii.gz files within a directory, including subdirectories.

        Args:
            directory: The directory to search.

        Returns:
            nii_gz_dict: A list of paths to all .nii.gz files found in a dictionary.
            labels: A list of all labels found in the dataset.
        """

        nii_gz_dict = {}
        labels = []

        n_dir = len(directory.split("/"))
        if directory.split("/")[-1] == "":
            n_dir -= 1

        for root, _, files in os.walk(directory):
            if len(root.split("/")) > n_dir: # checks if the path is a subdirectory
                key = root.split("/")[n_dir] # gets the name of the subdirectory

                if key not in nii_gz_dict.keys():
                    nii_gz_dict[key] = [] # creates a new list for the new subdirectory

                for file in files: # Exclude hidden files (starting with '.')
                    if not file.startswith('.') and file.endswith(".nii.gz"):
                        nii_gz_dict[key].append(os.path.join(root, file))

                        # check if instance exist in the filename
                        if "instance" in file:
                            label = file.split("_")[0]
                            if label not in labels:
                                labels.append(label)

                # filter annotation files
                nii_gz_dict[key] = self._filter_dataset(nii_gz_dict[key])

        return nii_gz_dict, sorted(labels)
        
    def _get_paths(self, nii_gz_dict_paths):
        img_paths = []; annot_paths = []
        for paths in nii_gz_dict_paths:
            im_paths = []; an_paths = []
            for f in paths:
                if "imaging" in f:
                    im_paths.append(f)
                elif "annotation" in f:
                    an_paths.append(f)
            img_paths.append(im_paths)
            annot_paths.append(an_paths)
        return img_paths, annot_paths
    
    def _get_annots(self, annot_paths, image, n_slice):
        annot_cyst = np.zeros((n_slice, image.shape[1], image.shape[2]))
        annot_kidney = np.zeros((n_slice, image.shape[1], image.shape[2]))
        annot_tumor = np.zeros((n_slice, image.shape[1], image.shape[2]))

        for nii_gz_file in annot_paths:
            if "annotation" in nii_gz_file:
                annotation = nib.load(nii_gz_file).get_fdata()
                annotname = nii_gz_file.split("/")[-1]
                # add the annotation to the corresponding class
                if self.labels[0] in annotname:
                    annot_cyst = np.logical_or(annot_cyst, annotation)
                elif self.labels[1] in annotname:
                    annot_kidney = np.logical_or(annot_kidney, annotation)
                elif self.labels[2] in annotname:
                    annot_tumor = np.logical_or(annot_tumor, annotation)

        return np.stack(
            [
                annot_cyst, 
                annot_kidney, 
                annot_tumor
            ],
            axis=3
        ).astype(np.uint8)

    def __getitem__(self, idx):
        list_keys = list(self.nii_gz_dict.keys())
        case_keys = list_keys[idx]
        if isinstance(case_keys, str):
            nii_gz_dict_paths = [self.nii_gz_dict[case_keys]]
        elif isinstance(case_keys, list):
            nii_gz_dict_paths = [self.nii_gz_dict[k] for k in case_keys]
        else:
            raise TypeError("case_keys must be a string or a list of strings")
        
        list_img_paths, list_annot_paths = self._get_paths(nii_gz_dict_paths)
        
        images = []
        annotations = []
        for index, img_paths in enumerate(list_img_paths):
            for img_path in img_paths:
                # print(img_path)
                image = nib.load(img_path).get_fdata()
                image_min = image.min()
                image_max = image.max()
                image = (image - image_min) / (image_max - image_min) # Normalize image
                # print(image.shape)
                n_slice = image.shape[0]
                annotation = self._get_annots(
                    list_annot_paths[index], 
                    image, 
                    n_slice
                )
                # print(annotation.shape)
            images.append(image)
            annotations.append(annotation)
        images = np.concatenate(images, axis=0)
        annotations = np.concatenate(annotations, axis=0)
        # print("*"*40)

        if self.transform:
            images = self.transform(images)
            images = torch.permute(images, (1,0,2)) # because ToTensor change shape (C,H,W) to (H,W,C)
        if self.target_transform:
            annotations = self.target_transform(annotations)
        return images, annotations
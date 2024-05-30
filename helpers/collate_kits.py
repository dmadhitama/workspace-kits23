import torch
import numpy as np

def collate_fn(data):
    """
    data: is a list of tuples with (example, label, length)
            where 'example' is a tensor of arbitrary shape
            and label/length are scalars
    """
    images, labels = zip(*data)
    new_images = []
    for slices_img in images:
        for img in slices_img:
            img = np.expand_dims(img, axis=0)
            img = torch.tensor(img)
            new_images.append(img)

    new_labels = []
    for slices_lab in labels:
        for lab in slices_lab:
            lab = np.expand_dims(lab, axis=0)
            lab = torch.tensor(lab).permute(0,3,1,2)
            new_labels.append(lab)
    return new_images, new_labels

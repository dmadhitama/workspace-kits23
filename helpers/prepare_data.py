import torch
from helpers.collate_kits import collate_fn
from torch.utils.data import Dataset

def split_dataset(
        dataset: Dataset, 
        test_size: float = 0.25,
        batch_size: int = 1,
        num_workers: int = 0,
    ): 
    len_test = int(len(dataset) * test_size)
    len_train = len(dataset) - len_test

    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset, 
        [len_train, len_test]
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train, 
        collate_fn=collate_fn, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset_val, 
        collate_fn=collate_fn, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers
    )
    return train_dataloader, val_dataloader
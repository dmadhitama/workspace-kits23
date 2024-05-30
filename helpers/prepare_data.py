import torch
from helpers.collate_kits import collate_fn

def split_dataset(dataset, test_size=0.25):
    len_test = int(len(dataset) * test_size)
    len_train = len(dataset) - len_test

    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset, 
        [len_train, len_test]
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train, 
        collate_fn=collate_fn, 
        batch_size=1,
        shuffle=True, 
        num_workers=0
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset_val, 
        collate_fn=collate_fn, 
        batch_size=1,
        shuffle=False, 
        num_workers=0
    )
    return train_dataloader, val_dataloader
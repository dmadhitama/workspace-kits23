import torch
import torch.nn as nn
from torchvision.transforms import (
    Compose, 
    Normalize, 
    ToTensor, 
    Resize,
    InterpolationMode
)
from torchvision.ops import Permute
from helpers.parser import argsparser
from utils.datasets.kits import Kits23Dataset
from helpers.prepare_data import split_dataset
from models.run import inference


if __name__ == "__main__":
    args = argsparser()
    
    DATASET_DIR = args.dataset_dir
    CHECKPOINT_PATH = "results/exp7/checkpoint_43.pt"
    IMAGE_SIZE = args.image_size
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IN_CHANNELS = args.in_channels
    N_CLASS = args.n_class
    TEST_SIZE = args.test_size
    GLOB_BATCH_SIZE = args.glob_batch_size
    NUM_WORKERS = args.num_workers

    THRESHOLD = 0.5
    IMAGE_SIZE_OUT = (512, 512) # output image size (original)

    labels = {
        0: "cyst",
        1: "kidney",
        2: "tumor",
    }
    color_labels = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
    ]

    transform = Compose(
        [
            ToTensor(),
            Permute(dims=(1,2,0)), # because ToTensor change shape (C,H,W) to (H,W,C)
            Normalize(mean=[0.5], std=[0.5]),
            Resize(size=IMAGE_SIZE), # handle non-standard shape image e.g. (512, 632) -> (512, 512)
        ]
    )
    target_transform = Compose(
        [
            ToTensor(),
            Permute(dims=(1,2,0)), # because ToTensor change shape (C,H,W) to (H,W,C)
            Resize(
                size=IMAGE_SIZE, 
                interpolation=InterpolationMode.NEAREST # handle interpolation result values between 0-1
            ), # handle non-standard shape image e.g. (512, 632) -> (512, 512)
        ]
    )
    print("=> Loading dataset...")
    kits_dataset = Kits23Dataset(
        dataset_dir=DATASET_DIR,
        input_transform=transform,
        target_transform=target_transform
    )
    _, val_dataloader = split_dataset(
        kits_dataset, 
        test_size=TEST_SIZE,
        batch_size=GLOB_BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    print("=> Loading model checkpoint...")
    model = torch.load(CHECKPOINT_PATH)
    inference(
        model=model,
        test_dataloader=val_dataloader,
        device=DEVICE,
        labels=labels,
        color_labels=color_labels,
        threshold=THRESHOLD,
        image_size_out=IMAGE_SIZE_OUT,
    )
    pass
    
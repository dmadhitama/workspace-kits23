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
from utils.datasets.kits import Kits23Dataset
from helpers.prepare_data import split_dataset
from models.single_module_unet import UNet
from models.run import train_loop, evaluate_loop, save_checkpoint
from helpers.parser import argsparser

if __name__ == "__main__":
    args = argsparser()

    DATASET_DIR = args.dataset_dir
    SAVE_MODEL_PATH = args.save_model_path
    SAVE_IMAGES_PATH = args.save_images_path
    PAD_MIRRORING = args.pad_mirroring

    TEST_SIZE = args.test_size
    IN_CHANNELS = args.in_channels
    N_CLASS = args.n_class
    IMAGE_SIZE = args.image_size

    DEVICE = args.device
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    GLOB_BATCH_SIZE = args.glob_batch_size
    LOC_BATCH_SIZE = args.loc_batch_size
    NUM_WORKERS = args.num_workers

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
    kits_dataset = Kits23Dataset(
        dataset_dir=DATASET_DIR,
        input_transform=transform,
        target_transform=target_transform
    )
    train_dataloader, val_dataloader = split_dataset(
        kits_dataset, 
        test_size=TEST_SIZE,
        batch_size=GLOB_BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    model = UNet(
        in_channels=IN_CHANNELS,
        n_class=N_CLASS,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch + 1}/{EPOCHS}")
        loss_epoch = train_loop(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
            bs=LOC_BATCH_SIZE,
        )
        val_loss_epoch = evaluate_loop(
            model=model,
            val_dataloader=val_dataloader,
            criterion=criterion,
            device=DEVICE,
        )
        save_checkpoint(
            model=model,
            folder_path=SAVE_MODEL_PATH,
            epoch=epoch,
        )
        print(f"\n\nEpoch {epoch+1}/{EPOCHS} - Current training loss: {loss_epoch}")
        print(f"Epoch {epoch+1}/{EPOCHS} - Current validation loss: {val_loss_epoch}\n\n")


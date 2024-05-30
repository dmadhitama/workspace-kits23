import torch
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor
from utils.datasets.kits import Kits23Dataset
from helpers.prepare_data import split_dataset
from models.single_module_unet import UNet
from models.run import train_loop, evaluate_loop, save_checkpoint

if __name__ == "__main__":
    DATASET_DIR = "kits23/dataset/"
    SAVE_MODEL_PATH = "results/"
    SAVE_IMAGES_PATH = "results/images/"
    PAD_MIRRORING = True

    TEST_SIZE = 0.3
    IN_CHANNELS = 1
    N_CLASS = 3

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 10
    LEARNING_RATE = 0.0001
    GLOB_BATCH_SIZE = 1
    LOC_BATCH_SIZE = 6
    NUM_WORKERS = 0

    kits_dataset = Kits23Dataset(DATASET_DIR)
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


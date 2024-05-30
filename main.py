import torch
from torchvision.transforms import Compose, Normalize, ToTensor
from utils.datasets.kits import Kits23Dataset
from helpers.prepare_data import split_dataset

if __name__ == "__main__":
    DATASET_DIR = "kits23/dataset/"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 10
    LEARNING_RATE = 0.0001
    SAVE_MODEL_PATH = "results/"
    SAVE_IMAGES_PATH = "results/images/"
    PAD_MIRRORING = True
    TEST_SIZE = 0.25

    kits_dataset = Kits23Dataset(DATASET_DIR)
    train_dataloader, val_dataloader = split_dataset(kits_dataset, TEST_SIZE)
    pass
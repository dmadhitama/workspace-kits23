import argparse
import torch

def argsparser():
    parser = argparse.ArgumentParser(description='Train a UNet model')

    parser.add_argument('--dataset_dir', type=str, default="kits23/dataset/",
                        help='Path to the dataset directory')
    parser.add_argument('--save_model_path', type=str, default="results/exp1/",
                        help='Path to save the model')
    parser.add_argument('--save_images_path', type=str, default="results/exp1/images/",
                        help='Path to save the images')
    parser.add_argument('--pad_mirroring', type=bool, default=True,
                        help='Whether to pad and mirror the images')
    parser.add_argument('--test_size', type=float, default=0.3,
                        help='Percentage of the dataset to use for testing')
    parser.add_argument('--in_channels', type=int, default=1,
                        help='Number of input channels')
    parser.add_argument('--n_class', type=int, default=3,
                        help='Number of output classes')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size of the input images, e.g. 256 -> for (256, 256) images')
    parser.add_argument('--device', type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help='Device to use for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--glob_batch_size', type=int, default=2,
                        help='Global batch size for training')
    parser.add_argument('--loc_batch_size', type=int, default=16,
                        help='Local batch size for training')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for the dataloader')
    return parser.parse_args()
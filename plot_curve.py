import matplotlib.pyplot as plt
import sys


def plot_loss(logs_path):
    with open(logs_path, "r") as f:
        lines = f.readlines()

    train_losses = []
    val_losses = []
    for line in lines:
        if "Current training loss" in line:
            train_losses.append(float(line.split(":")[-1].strip()))
        if "Current validation loss" in line:
            val_losses.append(float(line.split(":")[-1].strip()))

    plt.figure(figsize=(7,5))
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.grid()
    plt.legend()
    plt.savefig("loss_curve.png")


if __name__ == '__main__':
    logs_path = sys.argv[1]
    plot_loss(logs_path)
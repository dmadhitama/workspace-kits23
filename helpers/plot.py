import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_image_and_annotation(
    image,
    pred_mask,
    mask,
    labels,
    color_labels,
    image_save_path,
):
    """
    Plots an image and its corresponding segmentation mask.

    Args:
    
    """

    # Create a figure and axes
    fig, ax = plt.subplots(1, 1)
    # Scale the array linearly to the range 0-1
    image = image.permute(1,2,0).numpy() # (C,H,W) -> (H,W,C)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    # Display the image
    ax.imshow(image)

    # Define colors for each label class (you can customize these)
    # colors = {
    #     labels[0]: [255, 0, 0],  # Red
    #     labels[1]: [0, 255, 0],  # Green
    #     labels[2]: [0, 0, 255],  # Blue
    #     # Add more colors for other classes if needed
    # }

    rgb_pred_image = torch.zeros((3, image.shape[0], image.shape[1]), dtype=torch.uint8)
    for idx_class, color_value in enumerate(color_labels):
        for C in range(3):
            rgb_pred_image[C][pred_mask[idx_class] == 1] = color_value[C]
    rgb_pred_image = rgb_pred_image.permute(1,2,0).numpy()

    rgb_mask_image = torch.zeros((3, image.shape[0], image.shape[1]), dtype=torch.uint8)
    for idx_class, color_value in enumerate(color_labels):
        for C in range(3):
            rgb_mask_image[C][mask[idx_class] == 1] = color_value[C]

    rgb_mask_image = rgb_mask_image.permute(1,2,0).numpy()

    # Overlay the mask with the corresponding color
    masked_image = np.ma.masked_where(~rgb_mask_image, image)
    ax.imshow(
        rgb_mask_image,
        cmap=plt.cm.jet,
        interpolation='none',
        alpha=0.75
    )

            # if labels[idx] in pred_mask:
            #     # Overlay the mask with the corresponding color
            #     masked_image = np.ma.masked_where(~mask, image)
            #     ax.imshow(
            #         masked_image,
            #         cmap=plt.cm.jet,
            #         interpolation='none',
            #         alpha=0.75
            #     )

    # Set plot title and remove axes
    plt.title("Image with Segmentation Labels")
    ax.axis('off')

    # Show the plot
    # plt.show()
    plt.savefig("out_mask.png")
    plt.close()
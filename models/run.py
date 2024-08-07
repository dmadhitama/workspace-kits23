import torch
import os
from torchvision.transforms import (
    Resize,
    InterpolationMode
)
from helpers.plot import plot_image_and_annotation
from tqdm import tqdm
import numpy as np


def train_loop(
    model,
    train_dataloader,
    bs, # local batch size
    optimizer,
    criterion,
    device,
):
    model.train()
    outs = []
    avg_losses = []

    for idx, (images, masks) in enumerate(train_dataloader):
        count = 0
        loss_iter = 0
        len_iter = 0
        for id_img, image in enumerate(images):   
            # zero the gradients
            optimizer.zero_grad()
            
            image = image.float().unsqueeze(0).to(device)
            mask = masks[id_img].float().to(device)

            assert image.shape[-1] == image.shape[-2], \
                f"Image shape is {image.shape}, some image dimensions do not match! {image.shape[-1]} and {image.shape[-2]}"
            assert mask.shape[-1] == mask.shape[-2], \
                f"Mask shape is {mask.shape}, mask dimensions do not match! {mask.shape[-1]} and {mask.shape[-2]}"

            if mask.sum() > 0: 
                # if mask contains non zero values, then training is performed
                out = model(image)
                outs.append(out)
                
                if len(outs) == bs or id_img + 1 == len(images):
                    outs = torch.cat(outs, dim=0).to(device)
                    targets = torch.cat(masks[count:count+bs], dim=0).float().to(device)
                    # print(outs.shape)
                    # print(targets.shape)

                    # try:
                    loss = criterion(outs, targets)
                    # except:
                    #     raise ValueError ("Error nich!", outs.shape, targets.shape)
                    # backpropagate the loss
                    loss.backward()
                    # update the weights
                    optimizer.step()
                    loss_iter += loss.item()
                    # print("loss", loss_epoch)

                    # empty the output list
                    outs = []
                    count += bs
                    len_iter += 1
                    
        avg_loss = loss_iter/len_iter
        print(f"Iter {idx+1}/{len(train_dataloader)} - Current training average loss: {avg_loss:5f}")
        avg_losses.append(avg_loss)

        # # comment if needed
        # if idx == 4: # for testing short training
        #     outs = []
        #     break

    return np.array(avg_losses)

def evaluate_loop(
    model,
    val_dataloader,
    criterion,
    device,
    score_fn=None,
):
    avg_val_losses = []
    avg_val_scores = []
    model.eval()
    with torch.no_grad():
        for idx, (val_images, val_masks) in enumerate(val_dataloader):
            val_loss_iter = 0
            len_iter = 0
            val_score_iter = 0
            for id_img, val_image in enumerate(val_images):
                val_image = val_image.float().unsqueeze(0).to(device)
                val_mask = val_masks[id_img].float().to(device)

                assert val_image.shape[-1] == val_image.shape[-2], \
                    "image dimensions do not match!"
                assert val_mask.shape[-1] == val_mask.shape[-2], \
                    "mask dimensions do not match!"

                val_out = model(val_image)
                loss = criterion(val_out, val_mask)
                val_loss_iter += loss.item()

                if score_fn:
                    val_score = score_fn(val_out, val_mask)
                    val_score_iter += val_score
                len_iter += 1

            avg_val_loss = val_loss_iter/len_iter
            print(f"Iter {idx+1}/{len(val_dataloader)} - Current validation loss: {avg_val_loss:5f}")
            avg_val_losses.append(avg_val_loss)
            # # comment if not needed

            if score_fn:
                avg_val_score = val_score_iter/len_iter
                print(f"Iter {idx+1}/{len(val_dataloader)} - Current validation score: {avg_val_score:5f}")
                avg_val_scores.append(avg_val_score.cpu())
                return np.array(avg_val_losses), np.array(avg_val_scores)

            else:
                return np.array(avg_val_losses)
            # if idx == 4: # for testing short training
            #     break

    
def inference(
    model,
    test_dataloader,
    device,
    labels,
    color_labels,
    threshold=0.75,
    image_size_out=(512, 512)
):
    resize = Resize(
        image_size_out,
        interpolation=InterpolationMode.NEAREST
    )
    model.eval()
    with torch.no_grad():
        for idx, (test_images, test_masks) in enumerate(test_dataloader):
            print(f"Iter {idx+1}")
            for id_img, test_image in tqdm(enumerate(test_images)):
                test_image = test_image.float().unsqueeze(0).to(device)
                test_mask = test_masks[id_img].float().to(device)

                assert test_image.shape[-1] == test_image.shape[-2], "image dimensions do not match!"
                assert test_mask.shape[-1] == test_mask.shape[-2], "mask dimensions do not match!"

                test_out = torch.sigmoid(model(test_image))
                test_out = (test_out > threshold).float()

                # test_out = model(test_image)
                # test_out = torch.nn.functional.softmax(test_out, dim=1)
                # test_out = torch.argmax(test_out, dim=1)

                # resize to original image & mask size
                test_out = resize(test_out).squeeze(0).to("cpu")
                test_mask = resize(test_mask).squeeze(0).to("cpu")
                test_image = resize(test_image).squeeze(0).to("cpu")
                test_image = torch.cat(
                    [test_image, test_image, test_image], # convert to 1-channel to RGB
                    dim=0
                )
                
                if test_mask.max() > 0:
                    plot_image_and_annotation(
                        test_image,
                        test_out,
                        test_mask,
                        labels,
                        color_labels,
                        image_save_path=f"./results/{idx+1}.png"
                    )
                    pass

def save_checkpoint(
    model, 
    folder_path,
    epoch,
    filename="checkpoint"
):
    filename_epoch = f"{filename}_{epoch+1}.pt"
    filename_path = os.path.join(folder_path, filename_epoch)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print("=> Saving checkpoint...")
    torch.save(model, filename_path)
    print(f"=> Checkpoint saved to {filename_path}")

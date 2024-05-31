import torch
import os

def train_loop(
        model,
        train_dataloader,
        bs, # local batch size
        optimizer,
        criterion,
        device,
):
    loss_epoch = 0
    model.train()
    outs = []

    for idx, (images, masks) in enumerate(train_dataloader):
        count = 0
        for id_img, image in enumerate(images):
            image = image.float().unsqueeze(0).to(device)
            mask = masks[id_img].float().to(device)

            assert image.shape[-1] == image.shape[-2], \
                f"Image shape is {image.shape}, some image dimensions do not match! {image.shape[-1]} and {image.shape[-2]}"
            assert mask.shape[-1] == mask.shape[-2], \
                f"Mask shape is {mask.shape}, mask dimensions do not match! {mask.shape[-1]} and {mask.shape[-2]}"

            out = model(image)
            outs.append(out)
            
            if len(outs) == bs or id_img + 1 == len(images):
                outs = torch.cat(outs, dim=0).to(device)
                targets = torch.cat(masks[count:count+bs], dim=0).float().to(device)
                # print(outs.shape)
                # print(targets.shape)

                try:
                    loss = criterion(outs, targets)
                except:
                    raise ValueError ("Error nich!", outs.shape, targets.shape)
                    
                # zero the gradients
                optimizer.zero_grad(set_to_none=True)
                # backpropagate the loss
                loss.backward()
                # update the weights
                optimizer.step()
                loss_epoch += loss.item()
                # print("loss", loss_epoch)

                # empty the output list
                outs = []
                count += bs
        print(f"Iter {idx+1}/{len(train_dataloader)} - Current training loss: {loss_epoch:5f}")

        # comment if needed
        # if idx == 9: # for testing short training
        #     outs = []
        #     break

    return loss_epoch

def evaluate_loop(
        model,
        val_dataloader,
        criterion,
        device,
):
    model.eval()
    val_loss_epoch = 0
    with torch.no_grad():
        for idx, (val_images, val_masks) in enumerate(val_dataloader):
            for id_img, val_image in enumerate(val_images):
                val_image = val_image.float().unsqueeze(0).to(device)
                val_mask = val_masks[id_img].float().to(device)

                assert val_image.shape[-1] == val_image.shape[-2], "image dimensions do not match!"
                assert val_mask.shape[-1] == val_mask.shape[-2], "mask dimensions do not match!"

                val_out = model(val_image)
                loss = criterion(val_out, val_mask)
                val_loss_epoch += loss.item()
            print(f"Iter {idx+1}/{len(val_dataloader)} - Current validation loss: {val_loss_epoch:5f}")

            # comment if not needed
            # if idx == 9: # for testing short training
            #     break

    return val_loss_epoch

def save_checkpoint(
        model, 
        folder_path,
        epoch,
        filename="checkpoint"):
    filename_epoch = f"{filename}_{epoch+1}.pt"
    filename_path = os.path.join(folder_path, filename_epoch)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print("=> Saving checkpoint...")
    torch.save(model, filename_path)
    print(f"=> Checkpoint saved to {filename_path}")

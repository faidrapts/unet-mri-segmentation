import torch
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
import pandas as pd
import matplotlib.pyplot as plt

from loss_metrics import dice_coef_loss, dice_coef_accuracy, pixel_accuracy
from loss_metrics import weighted_bce_dice_loss, calculate_weight_map
from unet import UNet, BrainMriDataset


def train(model, train_loader, val_loader, epochs=100, init_lr=10e-3):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    train_losses = []
    val_losses = []

    print("Starting training...\n")
    for epoch in tqdm.trange(epochs):
        model.train()

        train_loss = []
        train_accuracy = []
        train_pixel_accuracy = []

        # train on batches
        for (img, mask) in train_loader:
            img = img.to(device)
            mask = mask.to(device)
            pred_mask = model(img)

            weights = calculate_weight_map(mask.data.cpu().numpy())
            weights = torch.from_numpy(weights)
            weights = weights.to(device)

            loss = weighted_bce_dice_loss(pred_mask, mask, weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_accuracy.append(
                dice_coef_accuracy(pred_mask.data.cpu().numpy(), mask.data.cpu().numpy()))
            train_pixel_accuracy.append(
                pixel_accuracy(pred_mask.data.cpu().numpy(), mask.data.cpu().numpy()))

        # evaluate predictions using validation set
        model.eval()
        with torch.no_grad():
            val_loss = []
            val_accuracy = []
            val_pixel_accuracy = []

            for (img, mask) in val_loader:
                img = img.to(device)
                mask = mask.to(device)
                pred_mask = model(img)

                weights = calculate_weight_map(mask.data.cpu().numpy())
                weights = torch.from_numpy(weights)
                weights = weights.to(device)

                loss = weighted_bce_dice_loss(pred_mask, mask, weights)

                val_loss.append(loss.item())
                val_accuracy.append(
                    dice_coef_accuracy(pred_mask.data.cpu().numpy(), mask.data.cpu().numpy()))
                val_pixel_accuracy.append(
                    pixel_accuracy(pred_mask.data.cpu().numpy(), mask.data.cpu().numpy()))

        epoch_train_loss = sum(train_loss)/len(train_loss)
        epoch_train_accuracy = sum(train_accuracy)/len(train_accuracy)
        epoch_train_pixel_accuracy = sum(
            train_pixel_accuracy)/len(train_pixel_accuracy)
        epoch_val_loss = sum(val_loss)/len(val_loss)
        epoch_val_accuracy = sum(val_accuracy)/len(val_accuracy)
        epoch_val_pixel_accuracy = sum(
            val_pixel_accuracy)/len(val_pixel_accuracy)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        print(f"\n Epoch {epoch}\n\
                Training loss: {epoch_train_loss}\n\
                Training DICE accuracy: {epoch_train_accuracy}\n\
                Training pixel accuracy: {epoch_train_pixel_accuracy}\n\
                Validation loss: {epoch_val_loss}\n\
                Validation DICE accuracy: {epoch_val_accuracy}\n\
                Validation pixel accuracy: {epoch_val_pixel_accuracy}\n")

    return train_losses, val_losses


if __name__ == "__main__":

    train_transform = A.Compose([
        A.Resize(width=128, height=128),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.05,
                           rotate_limit=0, p=0.25),
        A.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1],
            max_pixel_value=1
        )
    ])

    test_transform = A.Compose([
        A.Resize(width=128, height=128),
        A.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1],
            max_pixel_value=1
        )
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_paths_df = pd.read_csv(
        './dataset/data_paths.csv', index_col=0)

    BATCH_SIZE = 10
    EPOCHS = 50
    INIT_LR = 10e-4

    # split into train/val/test set with random seed (for reproducible results)
    train_size = int(0.8*len(data_paths_df))
    test_size = len(data_paths_df) - train_size
    train_df, test_df, _, _ = train_test_split(
        data_paths_df, data_paths_df, test_size=test_size, random_state=56)

    val_size = int(0.1*train_size)
    train_size = train_size - val_size
    train_df, val_df, _, _ = train_test_split(
        train_df, train_df, test_size=val_size, random_state=56)

    train_data = BrainMriDataset(train_df, train_transform)
    val_data = BrainMriDataset(val_df, test_transform)

    train_dataloader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


    model = UNet().to(device)

    train_loss_history, val_loss_history = train(
        model, train_dataloader, val_dataloader, EPOCHS, INIT_LR)

    torch.save(model, "./model_combined5.torch")

    epochs = [*range(1, 51)]
    fig, ax = plt.subplots()
    ax.set_facecolor("lightgrey")
    ax.plot(epochs, train_loss_history,
            c='rebeccapurple', label="Training loss")
    ax.plot(epochs, val_loss_history, c='darkcyan', label="Validation loss")
    plt.legend(loc="upper right")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Learning Curve")
    plt.grid()
    plt.savefig('./figures/learning_curve.pdf')

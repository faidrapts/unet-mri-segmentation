import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from loss_metrics import dice_coef_accuracy, pixel_accuracy
from unet import BrainMriDataset, UNet, DoubleConv


def test(model, test_loader):

    model.eval()

    imgs = []
    masks = []
    preds = []

    dice_acc = 0
    acc = 0

    print("Testing...\n")
    for i, (img, mask) in enumerate(test_loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            img = img.to(device)
            mask = mask.to(device)

        pred = model(img)

        if i < 20:
            imgs.append(img.data.cpu().numpy()[3])
            masks.append(mask.data.cpu().numpy()[3])
            preds.append(pred.data.cpu().numpy()[3])

        dice_acc += dice_coef_accuracy(pred.data.cpu().numpy(),
                                       mask.data.cpu().numpy())
        acc += pixel_accuracy(pred.data.cpu().numpy(), mask.data.cpu().numpy())

    print("\nDice accuracy: ", dice_acc/(i+1))
    print("Pixel accuracy: ", acc/(i+1))

    return imgs, masks, preds


if __name__ == "__main__":

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

    # split into train/val/test set with random seed (for reproducible results)
    # **ALL models were trained on training set created with random_state=56**
    train_size = int(0.8*len(data_paths_df))
    test_size = len(data_paths_df) - train_size
    train_df, test_df, _, _ = train_test_split(
        data_paths_df, data_paths_df, test_size=test_size, random_state=56)

    test_data = BrainMriDataset(test_df, test_transform)
    test_dataloader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = torch.load("model_combined.torch",
                       map_location=device)  # map_location=torch.device('cpu')
    imgs, masks, preds = test(model, test_dataloader)

    mpl.rcParams['figure.dpi'] = 300

    fig = plt.figure(figsize=(2, 6))
    grid = gridspec.GridSpec(5, 3)
    plt.subplots_adjust(hspace=0.1)

    imgs[6] = imgs[9]
    masks[6] = masks[9]
    preds[6] = preds[9]
 
    for i in range(5):
        ax1 = plt.subplot(grid[i, 0])
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax2 = plt.subplot(grid[i, 1])
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax3 = plt.subplot(grid[i, 2])
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)

        if i == 0:
            ax1.set_title('MRI image', fontsize=4)
            ax2.set_title('True mask', fontsize=4)
            ax3.set_title('Predicted mask', fontsize=4)
        else:
            ax1.set_title('')
            ax2.set_title('')
            ax3.set_title('')

        ax1.imshow(imgs[2*i].transpose((1, 2, 0)))
        ax2.imshow(masks[2*i].transpose((1, 2, 0)), cmap=plt.cm.magma)
        ax3.imshow(preds[2*i].transpose((1, 2, 0)), cmap=plt.cm.magma)

    plt.savefig('./figures/predicted_samples.pdf')

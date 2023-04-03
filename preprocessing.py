import os
import cv2
import glob
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from loss_metrics import weight_map


# diagnosis function
def diagnosed(mask):
    mask_img = cv2.imread(mask)

    # if maximum pixel value is above 0: there are non-black pixels in the mask image -> positive diagnosis
    if np.max(mask_img) > 0:
        return 1
    else:
        return 0


# function to remove too dark/invalid MRI scans
def is_valid(image):
    img = cv2.imread(image)

    non_black = (img > 0).sum()

    if non_black < 75000:
        return 0

    return 1


def read_tif(image_path, color):
    if color == True:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    return image


if __name__ == "__main__":

    # import image and corresponding mask paths into a dataframe
    folder_path = './dataset/lgg-mri-segmentation/kaggle_3m/'

    dataset_df = pd.DataFrame(columns=["img_file", "mask_file", "diagnosis"])

    for folder in os.listdir(folder_path):
        mask_files = glob.glob(folder_path + folder + '*/*_mask.tif')

        if mask_files:
            img_files = [file.replace('_mask', '') for file in mask_files]

            # set diagnosis value for each image-mask pair
            diagnosis = [diagnosed(mask) for mask in mask_files]
            valid = [is_valid(img) for img in img_files]

            folder_df = pd.DataFrame({"img_file": img_files,
                                      "mask_file": mask_files,
                                      "diagnosis": diagnosis,
                                      "valid": valid})
            dataset_df = pd.concat([dataset_df, folder_df], ignore_index=True)

    dataset_df = dataset_df.loc[dataset_df['valid'] == 1]
    dataset_df.to_csv('./dataset/data_paths.csv')



    mpl.rcParams['figure.dpi'] = 300

    # plot examples of removed scans in pre processing
    imgs = []

    imgs.append(
        folder_path + 'TCGA_DU_5855_19951217/TCGA_DU_5855_19951217_26.tif')
    imgs.append(
        folder_path + 'TCGA_DU_7306_19930512/TCGA_DU_7306_19930512_42.tif')
    imgs.append(
        folder_path + 'TCGA_DU_A5TW_19980228/TCGA_DU_A5TW_19980228_30.tif')
    imgs.append(
        folder_path + 'TCGA_DU_A5TU_19980312/TCGA_DU_A5TU_19980312_23.tif')

    fig = plt.figure(figsize=(3, 6))
    grid = gridspec.GridSpec(1, 4)
    # plt.subplots_adjust(hspace=0.1)
    plt.title('Images removed during pre-processing')

    for i in range(4):
        ax1 = plt.subplot(grid[0, i])
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax1.imshow(read_tif(imgs[i], color=True))

    plt.savefig('./figures/removed_samples.pdf')



    # visualization of some weight maps from positive samples with tumors>=2
    dataset_all = pd.read_csv('./dataset/data_paths.csv', index_col=0)
    dataset = dataset_all.loc[dataset_all['diagnosis'] == 1]

    # import the images and masks into a list
    masks = []

    for pair in dataset.sample(100).values:
        mask = read_tif(pair[1], color=False)
        masks.append(mask)

    wmaps = []
    wmasks = []

    for idx, mask in enumerate(masks):
        wmap, separate_masks = weight_map(mask)
        if separate_masks == 1:
            wmaps.append(wmap)
            wmasks.append(mask)

    fig1 = plt.figure(figsize=(2, 15))
    grid = gridspec.GridSpec(20, 2)
    plt.subplots_adjust(hspace=0.1)

    for i in range(5):
        with sns.axes_style('dark'):
            ax1 = plt.subplot(grid[i, 0])
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)

            ax2 = plt.subplot(grid[i, 1])
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)

            im1 = ax1.imshow(wmaps[i], cmap=plt.cm.magma)
            im2 = ax2.imshow(wmaps[i]+wmasks[i], cmap=plt.cm.magma)
            plt.tick_params(left=False, right=False, labelleft=False,
                            labelbottom=False, bottom=False)
            cb = plt.colorbar(im2)
            cb.ax.tick_params(labelsize=3)

    plt.savefig('./figures/weighted_samples.pdf')



    # locate some positive diagnosis samples
    positive_df = dataset_all.loc[dataset_all['diagnosis'] == 1]
    pos_imgs_sample = []
    pos_masks_sample = []

    for pair in positive_df.sample(10).values:
        pos_img = read_tif(pair[0], color=True)
        pos_mask = read_tif(pair[1], color=False)
        pos_imgs_sample.append(pos_img)
        pos_masks_sample.append(pos_mask)

    # visualization of some positive diagnosis samples
    fig = plt.figure(figsize=(2, 3.5))
    grid = gridspec.GridSpec(5, 3)
    plt.subplots_adjust(hspace=0.1)

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
            ax1.set_title('MRI image', fontsize=3.5)
            ax2.set_title('Segmentation mask', fontsize=3.5)
            ax3.set_title('MRI with mask', fontsize=3.5)
        ax1.imshow(pos_imgs_sample[i*2])
        ax2.imshow(pos_masks_sample[i*2], cmap=plt.cm.magma)
        ax3.imshow(pos_imgs_sample[i*2])
        ax3.imshow(pos_masks_sample[i*2], alpha=0.6, cmap=plt.cm.magma)

    plt.savefig(
        './figures/positive_diagnosis_samples.pdf')



    # visualization of some images and masks from imported dataset
    imgs_sample = []
    masks_sample = []

    for pair in dataset_all.sample(10).values:
        img = read_tif(pair[0], color=True)
        mask = read_tif(pair[1], color=False)
        imgs_sample.append(img)
        masks_sample.append(mask)

    fig = plt.figure(figsize=(1.5, 4))
    grid = gridspec.GridSpec(5, 2)
    plt.subplots_adjust(hspace=0.1)

    for i in range(5):
        ax1 = plt.subplot(grid[i, 0])
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax2 = plt.subplot(grid[i, 1])
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        if i == 0:
            ax1.set_title('MRI image', fontsize=3.5)
            ax2.set_title('Segmentation mask', fontsize=3.5)
        ax1.imshow(imgs_sample[i])
        ax2.imshow(masks_sample[i], cmap=plt.cm.magma)

    plt.savefig('./figures/mixed_diagnosis_samples.pdf')

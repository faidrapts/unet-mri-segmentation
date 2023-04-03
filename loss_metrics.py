import torch
import torch.nn as nn
import numpy as np
import skimage
from skimage.measure import label
from scipy.ndimage import distance_transform_edt


def dice_coef_loss(pred, target):

    smooth = 1.
    intersection = 2. * ((target*pred).sum()) + smooth
    union = target.sum() + pred.sum() + smooth
    loss = 1 - (intersection/union)

    return loss


def bce_dice_loss(pred, target):

    dice_loss = dice_coef_loss(pred, target)
    bce = nn.BCELoss()
    bce_loss = bce(pred, target)

    return bce_loss + dice_loss


def weighted_bce_dice_loss(pred, target, weights):

    bce_loss = nn.BCELoss(reduction='none')(pred, target)
    bce_loss = bce_loss.type(torch.double)
    weighted_loss = torch.mul(bce_loss, weights)
    dice_loss = dice_coef_loss(pred, target)

    return weighted_loss.mean() + dice_loss


def dice_coef_accuracy(pred, target):
    pred[pred >= 0.3] = 1.
    pred[pred < 0.3] = 0.

    smooth = 1.
    intersection = 2. * (target*pred).sum() + smooth
    union = target.sum() + pred.sum() + smooth

    if target.sum() == 0 and pred.sum() == 0:
        return 1

    return intersection/union


def pixel_accuracy(pred, target):
    pred[pred >= 0.3] = 1.
    pred[pred < 0.3] = 0.

    total = (target == 0).sum() + (target == 1).sum()

    return (pred == target).sum()/total


def get_class_weights(mask):

    weight = np.zeros(mask.shape)
    c0 = (mask == 0)
    count_0 = c0.sum()
    c1 = (mask == 1)
    count_1 = c1.sum()

    if count_1 < 10:
        return np.ones(mask.shape)

    total = mask.shape[0]*mask.shape[1]
    weight_0 = total / count_0
    weight += weight_0*c0
    weight_1 = total / count_1
    weight += weight_1*c1

    return weight


def weight_map(mask, w0=10, bw=5):
    # label (skimage.measure) method returns array, where all connected regions are assigned the same integer value
    mask_label, num_labels = skimage.measure.label(
        mask, background=0, return_num=True)
    unique_labels = sorted(np.unique(mask_label))

    many_regions = 0

    # if there are two (or more) separate masked regions
    if num_labels > 1:
        many_regions = 1
        distance = np.zeros((mask.shape[0], mask.shape[1], len(unique_labels)))

        for k, lbl in enumerate(unique_labels):
            distance[:, :, k] = distance_transform_edt(mask_label != lbl)
        distance = np.sort(distance, axis=2)
        d = distance[:, :, 0] + distance[:, :, 1]
        w = w0 * np.exp(-1/2*(d/bw)**2)

    else:
        w = np.zeros_like(mask)

    wc = get_class_weights(mask)

    return wc + w, many_regions


# helper function to calculate weight map for every image in batch
def calculate_weight_map(masks: np.array):
    masks = np.squeeze(masks)
    weights = []

    for i in range(len(masks)):
        weight, _ = weight_map(masks[i, :, :])
        weights.append(weight)

    weights = np.array(weights)

    return weights

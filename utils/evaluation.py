import warnings
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
import matplotlib.pyplot as plt
import skimage.io
import numpy as np
import skimage.morphology
import os
import pandas as pd
from scipy import ndimage
from tqdm import tqdm
import colorsys
import tifffile
from scipy.ndimage import find_objects, gaussian_filter, generate_binary_structure, label, maximum_filter1d, binary_fill_holes

import utils.dirtools
from config import config_vars


def imsave(filename, arr):
    ext = os.path.splitext(filename)[-1]
    if ext== '.tif' or ext=='tiff':
        tifffile.imsave(filename, arr)
    else:
        cv2.imwrite(filename, arr)
def rgb_to_hsv(arr):
    rgb_to_hsv_channels = np.vectorize(colorsys.rgb_to_hsv)
    r, g, b = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv_channels(r, g, b)
    hsv = np.stack((h,s,v), axis=-1)
    return hsv
def hsv_to_rgb(arr):
    hsv_to_rgb_channels = np.vectorize(colorsys.hsv_to_rgb)
    h, s, v = np.rollaxis(arr, axis=-1)
    r, g, b = hsv_to_rgb_channels(h, s, v)
    rgb = np.stack((r,g,b), axis=-1)
    return rgb
def mask_overlay(img, masks, colors=None):
    """ overlay masks on image (set image to grayscale)

    Parameters
    ----------------

    img: int or float, 2D or 3D array
        img is of size [Ly x Lx (x nchan)]

    masks: int, 2D array
        masks where 0=NO masks; 1,2,...=mask labels

    colors: int, 2D array (optional, default None)
        size [nmasks x 3], each entry is a color in 0-255 range

    Returns
    ----------------

    RGB: uint8, 3D array
        array of masks overlaid on grayscale image

    """
    if colors is not None:
        if colors.max()>1:
            colors = np.float32(colors)
            colors /= 255
        colors = rgb_to_hsv(colors)
    if img.ndim>2:
        img = img.astype(np.float32).mean(axis=-1)
    else:
        img = img.astype(np.float32)
    img = normalize99(img)
    img -= img.min()
    img /= img.max()
    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    HSV[:,:,2] = np.clip(img*1.5, 0, 1.0)
    for n in range(int(masks.max())):
        ipix = (masks==n+1).nonzero()
        if colors is None:
            HSV[ipix[0],ipix[1],0] = np.random.rand()
        else:
            HSV[ipix[0],ipix[1],0] = colors[n,0]
        HSV[ipix[0],ipix[1],1] = 1.0
    RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB
def normalize99(img):
    X = img.copy()
    X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1))
    return X
def masks_to_outlines(masks):
    """ get outlines of masks as a 0-1 array

    Parameters
    ----------------

    masks: int, 2D or 3D array
        size [Ly x Lx] or [Lz x Ly x Lx], 0=NO masks; 1,2,...=mask labels

    Returns
    ----------------

    outlines: 2D or 3D array
        size [Ly x Lx] or [Lz x Ly x Lx], True pixels are outlines

    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array' % masks.ndim)
    outlines = np.zeros(masks.shape, np.bool)

    if masks.ndim == 3:
        for i in range(masks.shape[0]):
            outlines[i] = masks_to_outlines(masks[i])
        return outlines
    else:
        slices = find_objects(masks.astype(int))
        for i, si in enumerate(slices):
            if si is not None:
                sr, sc = si
                mask = (masks[sr, sc] == (i + 1)).astype(np.uint8)
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T
                vr, vc = pvr + sr.start, pvc + sc.start
                outlines[vr, vc] = 1
        return outlines
def image_to_rgb(img0, channels=[0,0]):
    """ image is 2 x Ly x Lx or Ly x Lx x 2 - change to RGB Ly x Lx x 3 """
    img = img0.copy()
    img = img.astype(np.float32)
    if img.ndim<3:
        img = img[:,:,np.newaxis]
    if img.shape[0]<5:
        img = np.transpose(img, (1,2,0))
    if channels[0]==0:
        img = img.mean(axis=-1)[:,:,np.newaxis]
    for i in range(img.shape[-1]):
        if np.ptp(img[:,:,i])>0:
            img[:,:,i] = normalize99(img[:,:,i])
            img[:,:,i] = np.clip(img[:,:,i], 0, 1)
    img *= 255
    img = np.uint8(img)
    RGB = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    if img.shape[-1]==1:
        RGB = np.tile(img,(1,1,3))
    else:
        RGB[:,:,channels[0]-1] = img[:,:,0]
        if channels[1] > 0:
            RGB[:,:,channels[1]-1] = img[:,:,1]
    return RGB
def show_segmentation(fig, img, maski, channels=[0, 0], file_name=None):
    """ plot segmentation results (like on website)

    Can save each panel of figure with file_name option. Use channels option if
    img input is not an RGB image with 3 channels.

    Parameters
    -------------

    fig: matplotlib.pyplot.figure
        figure in which to make plot

    img: 2D or 3D array
        image input into cellpose

    maski: int, 2D array
        for image k, masks[k] output from Cellpose.eval, where 0=NO masks; 1,2,...=mask labels

    flowi: int, 2D array
        for image k, flows[k][0] output from Cellpose.eval (RGB of flows)

    channels: list of int (optional, default [0,0])
        channels used to run Cellpose, no need to use if image is RGB

    file_name: str (optional, default None)
        file name of image, if file_name is not None, figure panels are saved

    """
    ax = fig.add_subplot(1, 3, 1)
    img0 = img.copy()
    if img0.shape[0] < 4:
        img0 = np.transpose(img0, (1, 2, 0))
    if img0.shape[-1] < 3 or img0.ndim < 3:
        img0 = image_to_rgb(img0, channels=channels)
    else:
        if img0.max() <= 50.0:
            img0 = np.uint8(np.clip(img0 * 255, 0, 1))
    ax.imshow(img0)
    ax.set_title('original image')
    ax.axis('off')

    outlines = masks_to_outlines(maski)
    overlay = mask_overlay(img0, maski)

    ax = fig.add_subplot(1, 3, 2)
    outX, outY = np.nonzero(outlines)
    imgout = img0.copy()
    imgout[outX, outY] = np.array([255, 75, 75])
    ax.imshow(imgout)
    # for o in outpix:
    #    ax.plot(o[:,0], o[:,1], color=[1,0,0], lw=1)
    ax.set_title('predicted outlines')
    ax.axis('off')

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(overlay)
    ax.set_title('predicted masks')
    ax.axis('off')

    # ax = fig.add_subplot(1, 4, 4)
    # ax.imshow(flowi)
    # ax.set_title('predicted cell pose')
    # ax.axis('off')

    if file_name is not None:
        save_path = os.path.splitext(file_name)[0]
        imsave(save_path + '_overlay.png', overlay)
        imsave(save_path + '_outlines.png', imgout)
        # imsave(save_path + '_flows.jpg', flowi)

def get_fast_aji(true, pred):
    """AJI version distributed by MoNuSeg, has no permutation problem but suffered from
    over-penalisation similar to DICE2.
    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no
    effect on the result.
    """
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    aji_score = overall_inter / overall_union
    return aji_score
def get_fast_pq(true, pred, match_iou=0.5):
    """`match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique
    (1 prediction instance to 1 GT instance mapping).
    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing.
    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.

    Fast computation requires instance IDs are in contiguous orderding
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand
    and `by_size` flag has no effect on the result.
    Returns:
        [dq, sq, pq]: measurement statistic
        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

    """
    assert match_iou >= 0.0, "Cant' be negative"

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    # try:
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    sq = paired_iou.sum() / (tp + 1.0e-6)
    tmp = dq * sq
    # except:
    #     dq = 'NA'
    #     sq = paired_iou.sum() / (tp + 1.0e-6)
    #     tmp = 'NA'
    # get the SQ, no paired has 0 iou so not impact
    # sq = paired_iou.sum() / (tp + 1.0e-6)
    

    return [dq, sq, tmp], [paired_true, paired_pred, unpaired_true, unpaired_pred]
def raw_anno_preprocess(annot):
    # annot = skimage.io.imread(config_vars["raw_annotations_dir"] + filename)
    # annot = skimage.io.imread(r'D:\ppp\spatial_proteomics\data\datasets\raw_annotations\Cell_line1_ROI1.png')

    # strip the first channel
    if len(annot.shape) == 3:
        annot = annot[:, :, 0]

    # filter small objects, e.g. micronulcei
    annot = skimage.morphology.remove_small_objects(annot, min_size=config_vars['cell_min_size'])

    # label the annotations nicely to prepare for future filtering operation
    annot = skimage.morphology.label(annot)
    # total_objects += len(np.unique(annot)) - 1


    return annot

def pred_postprocess(pred):

    # pred = ndimage.binary_fill_holes(pred-1).astype(int)

    #background =1 foreground=2
    # pred = skimage.morphology.remove_small_objects(pred.astype(int), min_size=config['MIN_SIZE'])
    # pred = skimage.morphology.label(pred,background=1)
    pred = skimage.morphology.label(pred)
    return pred

def get_fast_dice_2(true, pred):
    """Ensemble dice."""
    true = np.copy(true)
    pred = np.copy(pred)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))

    overall_total = 0
    overall_inter = 0

    true_masks = [np.zeros(true.shape)]
    for t in true_id[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [np.zeros(true.shape)]
    for p in pred_id[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    for true_idx in range(1, len(true_id)):
        t_mask = true_masks[true_idx]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        try:  # blinly remove background
            pred_true_overlap_id.remove(0)
        except ValueError:
            pass  # just mean no background
        for pred_idx in pred_true_overlap_id:
            p_mask = pred_masks[pred_idx]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            overall_total += total
            overall_inter += inter

    return 2 * overall_inter / overall_total
def get_dice_1(true, pred):
    """Traditional dice."""
    # cast to binary 1st
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = true * pred
    denom = true + pred
    return 2.0 * np.sum(inter) / np.sum(denom)


def proposed_seg_evaluation(experiment_name,partition):
    data_partitions = utils.dirtools.read_data_partitions(config_vars)
    utils.dirtools.setup_experiment(config_vars,experiment_name)

    gt_mask = [os.path.join(config_vars["raw_annotations_dir"], f) for f in data_partitions[partition]]
    raw_img_list = [os.path.join(config_vars["normalized_images_dir"], f) for f in data_partitions[partition]]
    pred_mask = [os.path.join(config_vars["labels_out_dir"], f[:-4]+'.npy') for f in data_partitions[partition]]
    pred_watershed_mask = [os.path.join(config_vars["watershed_labels_out_dir"], f[:-4]+'.npy') for f in data_partitions[partition]]

    metrics = [[], [], [], [], [], []]
    metrics_watershed = [[], [], [], [], [], []]
    for i in range(0,len(gt_mask)):
        # load gt
        print(i)
        gt = skimage.io.imread(gt_mask[i])
        gt = raw_anno_preprocess(gt)
        raw_img = skimage.io.imread(raw_img_list[i])

        pred = np.load(pred_mask[i])
        pred = skimage.morphology.label(pred)

        pred_watershed = np.load(pred_watershed_mask[i])
        pred_watershed = skimage.morphology.label(pred_watershed)

        show_segmentation(plt.figure(figsize=(12, 5)), raw_img, pred, channels=[0, 0])

        pq_info = get_fast_pq(gt, pred, match_iou=0.5)[0]
        metrics[0].append(get_dice_1(gt, pred))
        metrics[1].append(get_fast_dice_2(gt, pred))
        metrics[2].append(get_fast_aji(gt, pred))

        metrics[3].append(pq_info[0])  # dq
        metrics[4].append(pq_info[1])  # sq
        metrics[5].append(pq_info[2])  # pq

        pq_info_watershed = get_fast_pq(gt, pred_watershed, match_iou=0.5)[0]
        metrics_watershed[0].append(get_dice_1(gt, pred_watershed))
        metrics_watershed[1].append(get_fast_dice_2(gt, pred_watershed))
        metrics_watershed[2].append(get_fast_aji(gt, pred_watershed))

        metrics_watershed[3].append(pq_info_watershed[0])  # dq
        metrics_watershed[4].append(pq_info_watershed[1])  # sq
        metrics_watershed[5].append(pq_info_watershed[2])  # pq

    eval = pd.DataFrame({"Data subset": data_partitions[partition], "DICE1": metrics[0],
                         "DICE2": metrics[1], "AJI": metrics[2],
                         "DQ": metrics[3], "SQ": metrics[4], "PQ": metrics[5]})
    eval_watershed = pd.DataFrame({"Data subset": data_partitions[partition], "DICE1": metrics_watershed[0],
                         "DICE2": metrics_watershed[1], "AJI": metrics_watershed[2],
                         "DQ": metrics_watershed[3], "SQ": metrics_watershed[4], "PQ": metrics_watershed[5]})                        
    # return eval
    return eval,eval_watershed
# ilastik_pred_dir = r'D:\ppp\spatial_proteomics\data\ilastik_segmentation\ilastik_simple_segmentation_5%\segmentation'
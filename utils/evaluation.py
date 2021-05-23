import numpy as np
import pandas as pd


def intersection_over_union(ground_truth, prediction):
    
    # Count objects
    true_objects = len(np.unique(ground_truth))
    pred_objects = len(np.unique(prediction))
    
    # Compute intersection
    h = np.histogram2d(ground_truth.flatten(), prediction.flatten(), bins=(true_objects,pred_objects))
    intersection = h[0]
    
    # Area of objects
    area_true = np.histogram(ground_truth, bins=true_objects)[0]
    area_pred = np.histogram(prediction, bins=pred_objects)[0]
    
    # Calculate union
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection
    
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    
    # Compute Intersection over Union
    union[union == 0] = 1e-9
    IOU = intersection/union
    
    return IOU
    


def measures_at(threshold, IOU):
    
    matches = IOU > threshold
    
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Extra objects
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects
    
    assert np.all(np.less_equal(true_positives, 1))
    assert np.all(np.less_equal(false_positives, 1))
    assert np.all(np.less_equal(false_negatives, 1))
    
    TP, FP, FN = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    
    f1 = 2*TP / (2*TP + FP + FN + 1e-9)
    
    return f1, TP, FP, FN

# Compute Average Precision for all IoU thresholds

def compute_af1_results(ground_truth, prediction, results, image_name):

    # Compute IoU
    IOU = intersection_over_union(ground_truth, prediction)
    if IOU.shape[0] > 0:
        jaccard = np.max(IOU, axis=0).mean()
    else:
        jaccard = 0.0
    
    # Calculate F1 score at all thresholds
    for t in np.arange(0.5, 1.0, 0.05):
        f1, tp, fp, fn = measures_at(t, IOU)
        res = {"Image": image_name, "Threshold": t, "F1": f1, "Jaccard": jaccard, "TP": tp, "FP": fp, "FN": fn}
        row = len(results)
        results.loc[row] = res
        
    return results

# Count number of False Negatives at 0.7 IoU

def get_false_negatives(ground_truth, prediction, results, image_name, threshold=0.7):

    # Compute IoU
    IOU = intersection_over_union(ground_truth, prediction)
    
    true_objects = len(np.unique(ground_truth))
    if true_objects <= 1:
        return results
        
    area_true = np.histogram(ground_truth, bins=true_objects)[0][1:]
    true_objects -= 1
    
    # Identify False Negatives
    matches = IOU > threshold
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects

    data = np.asarray([ 
        area_true.copy(), 
        np.array(false_negatives, dtype=np.int32)
    ])

    results = pd.concat([results, pd.DataFrame(data=data.T, columns=["Area", "False_Negative"])])
        
    return results

# Count the number of splits and merges

def get_splits_and_merges(ground_truth, prediction, results, image_name):

    # Compute IoU
    IOU = intersection_over_union(ground_truth, prediction)
    
    matches = IOU > 0.1
    merges = np.sum(matches, axis=0) > 1
    splits = np.sum(matches, axis=1) > 1
    r = {"Image_Name":image_name, "Merges":np.sum(merges), "Splits":np.sum(splits)}
    results.loc[len(results)+1] = r
    return results

def proposed_seg_evaluation(gt_dir,pred_dir):
    file_list = os.listdir(gt_dir)
    metrics = [[], [], [], [], [], []]
    for file in tqdm(file_list):
        # load gt
        gt = skimage.io.imread(os.path.join(gt_dir, file))
        gt = raw_anno_preprocess(gt)
        raw_img = skimage.io.imread(os.path.join(r'D:\ppp\spatial_proteomics\data\datasets\norm_images', file))

        pred = skimage.io.imread(os.path.join(pred_dir, file[:-4] + '.png'))
        pred = skimage.morphology.label(pred)

        show_segmentation(plt.figure(figsize=(12, 5)), raw_img, pred, channels=[0, 0],
                          file_name=os.path.join(
                              r'D:\ppp\spatial_proteomics\data\proposed_segmentation\proposed_plot',
                              file[:-4]))

        pq_info = get_fast_pq(gt, pred, match_iou=0.5)[0]
        metrics[0].append(get_dice_1(gt, pred))
        metrics[1].append(get_fast_dice_2(gt, pred))
        metrics[2].append(get_fast_aji(gt, pred))

        metrics[3].append(pq_info[0])  # dq
        metrics[4].append(pq_info[1])  # sq
        metrics[5].append(pq_info[2])  # pq

    eval = pd.DataFrame({"Data subset": file_list, "DICE1": metrics[0],
                         "DICE2": metrics[1], "AJI": metrics[2],
                         "DQ": metrics[3], "SQ": metrics[4], "PQ": metrics[5]})
    # eval = pd.DataFrame({"Data subset":file_list,"DICE1":metrics[0],
    #               "DICE2":metrics[1],"PQ":metrics[2],
    #               "AJI":metrics[3]})
    return eval
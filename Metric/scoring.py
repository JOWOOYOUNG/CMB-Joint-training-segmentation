import numpy as np
import scipy.ndimage

def dice_coef(truth, prediction):
    # Reshape the input to reduce computation to a dot product
    prediction = np.reshape(prediction, (1,np.prod(prediction.shape)))
    truth = np.reshape(truth, (1,np.prod(truth.shape)))

    # Prevent values >1 from inflating the score
    np.clip(truth, 0, 1, out=truth)
    np.clip(prediction, 0, 1, out=prediction)

    # Compute dice coef
    coef_denom = np.sum(prediction) + np.sum(truth)
    if(coef_denom == 0):  # If there are no non-zero labels in either the truth or the prediction
        coef = (1.0)  # "Perfect" score
        
    coef = (prediction * truth).sum()
    coef = 2*coef / coef_denom

    return float(coef)

def volume_difference(truth, prediction):
    pred_vol = int(np.sum(prediction))
    truth_vol = int(np.sum(truth))
    return np.abs(pred_vol - truth_vol)

def simple_lesion_count_difference(truth, prediction):
    _, pred_count = scipy.ndimage.label(prediction)
    _, truth_count = scipy.ndimage.label(truth)
    return np.abs(pred_count - truth_count)

def _lesion_f1_score(truth, prediction, empty_value=1.0):
    tp, fp, fn = 0, 0, 0
    f1_score = empty_value

    labeled_ground_truth, num_lesions = scipy.ndimage.label(truth.astype(bool))

    # For each true lesion, check if there is at least one overlapping voxel.
    # This determines true positives (TP) and false negatives (FN).
    for idx_lesion in range(1, num_lesions + 1):
        lesion = labeled_ground_truth == idx_lesion
        if np.any(lesion & prediction):
            tp += 1
        else:
            fn += 1

    # For each predicted lesion, check if there is at least one overlapping voxel in the ground truth.
    # This determines false positives (FP).
    labeled_prediction, num_pred_lesions = scipy.ndimage.label(prediction.astype(bool))
    for idx_lesion in range(1, num_pred_lesions + 1):
        lesion = labeled_prediction == idx_lesion
        if not np.any(lesion & truth):
            fp += 1

    # Compute F1 score
    denom = tp + 0.5 * (fp + fn)
    if denom != 0:
        f1_score = tp / denom

    return f1_score, fp

def lesion_f1_score(truth, prediction):
    f1_score, fp = _lesion_f1_score(truth, prediction)
    return f1_score, fp

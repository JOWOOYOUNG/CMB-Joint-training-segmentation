import numpy as np
import pandas as pd
import os
from glob import glob
import nibabel as nib
import SimpleITK as sitk
import scipy.ndimage
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from tqdm import tqdm
import openpyxl

def dice_coef(truth, prediction):
    prediction = np.reshape(prediction, (1,np.prod(prediction.shape)))
    truth = np.reshape(truth, (1,np.prod(truth.shape)))
    np.clip(truth, 0, 1, out=truth)
    np.clip(prediction, 0, 1, out=prediction)
    coef_denom = np.sum(prediction) + np.sum(truth)
    if coef_denom == 0:
        return 1.0
    coef = (prediction * truth).sum()
    coef = 2*coef / coef_denom
    return float(coef)

def volume_difference(truth_path, prediction_path):
    truth_img = nib.load(truth_path)
    pred_img = nib.load(prediction_path)
    voxel_spacing = truth_img.header.get_zooms()
    voxel_volume = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
    truth_data = truth_img.get_fdata()
    pred_data = pred_img.get_fdata()
    truth_voxel_count = np.sum(truth_data == 1)
    pred_voxel_count = np.sum(pred_data == 1)
    truth_volume = truth_voxel_count * voxel_volume
    pred_volume = pred_voxel_count * voxel_volume
    return np.float32(pred_volume - truth_volume)

def simple_lesion_count_difference(truth, prediction):
    _, pred_count = scipy.ndimage.label(prediction)
    _, truth_count = scipy.ndimage.label(truth)
    return np.int32(pred_count - truth_count)

def lesion_f1_score(truth, prediction):
    tp, fp, fn = 0, 0, 0
    labeled_ground_truth, num_lesions = scipy.ndimage.label(truth.astype(bool))
    for idx_lesion in range(1, num_lesions + 1):
        lesion = labeled_ground_truth == idx_lesion
        if np.any(lesion & prediction.astype(bool)): # boolean으로!
            tp += 1
        else:
            fn += 1
    labeled_prediction, num_pred_lesions = scipy.ndimage.label(prediction.astype(bool))
    for idx_lesion in range(1, num_pred_lesions + 1):
        lesion = labeled_prediction == idx_lesion
        if not np.any(lesion & truth.astype(bool)):
            fp += 1
    denom = tp + 0.5 * (fp + fn)
    f1_score = tp / denom if denom != 0 else 1.0
    return f1_score, tp, fp, fn

def compute_froc_pos(df_pos, thresholds):
    froc_scores = []
    for th in thresholds:
        sensitivities = [
            row["TP"] / (row["TP"] + row["FN"])
            for _, row in df_pos.iterrows()
            if row["FP"] <= th and (row["TP"] + row["FN"]) > 0
        ]
        score = np.mean(sensitivities) if sensitivities else 0.0
        froc_scores.append(score)
    return froc_scores

def compute_froc_neg(df_neg, thresholds):
    if len(df_neg) == 0:
        return [0.0] * len(thresholds)
    return [(df_neg["FP"] <= th).sum() / len(df_neg) for th in thresholds]

def compute_froc_all(df_pos, df_neg, thresholds):
    df_all = pd.concat([df_pos, df_neg], ignore_index=True)
    froc_scores = []
    for th in thresholds:
        sensitivities = [
            row["TP"] / (row["TP"] + row["FN"])
            for _, row in df_all.iterrows()
            if row["CMB_Status"] == "CMB_Pos" and row["FP"] <= th and (row["TP"] + row["FN"]) > 0
        ]
        score = np.mean(sensitivities) if sensitivities else 0.0
        froc_scores.append(score)
    return froc_scores

def evaluate(truth_path, prediction_path, id_to_folder, froc_img_name, xlsx_name, save_dir):
    t_ = sorted(glob(f'{truth_path}/*.nii.gz'))
    p_ = sorted(glob(f'{prediction_path}/*.nii.gz'))

    wb = openpyxl.Workbook()
    sheet1 = wb['Sheet']
    sheet1.title = 'Evaluation'
    sheet1.append(['id', 'Dice', 'F1', 'Volume_GT', 'Volume_PRED', 'VolumeDiff', 'LesionCount_GT', 'LesionCount_PRED', 'LesionCountDiff', 'FP', 'CMB_Status'])

    results = []
    print(f"총 {len(t_)}개 파일에 대해 평가를 시작합니다...")

    for idx in tqdm(range(len(t_))):
        truth_img = nib.load(t_[idx])
        pred_img = nib.load(p_[idx])
        spacing = truth_img.header.get_zooms()
        voxel_vol = spacing[0] * spacing[1] * spacing[2]
        truth_ = truth_img.get_fdata()
        pred_ = pred_img.get_fdata()
        volume_gt = np.sum(truth_ == 1) * voxel_vol
        volume_pred = np.sum(pred_ == 1) * voxel_vol
        _, count_gt = scipy.ndimage.label(truth_)
        _, count_pred = scipy.ndimage.label(pred_)
        case_id = t_[idx].split('/')[-1]
        truth_ = sitk.GetArrayFromImage(sitk.ReadImage(str(t_[idx])))
        pred_ = sitk.GetArrayFromImage(sitk.ReadImage(str(p_[idx])))

        dice_ = dice_coef(truth=truth_, prediction=pred_)
        volume_ = volume_difference(t_[idx], p_[idx])
        count_ = simple_lesion_count_difference(truth=truth_, prediction=pred_)

        folder = id_to_folder.get(case_id, 'Unknown')
        if folder == 'CMB_Neg':
            _, fp = scipy.ndimage.label(pred_.astype(bool))
            f1_, tp, fn = 1.0, 0, 0
        else:
            f1_, tp, fp, fn = lesion_f1_score(truth=truth_, prediction=pred_)

        sheet1.append([case_id[:-7], round(dice_, 3), round(f1_, 3), round(volume_gt, 3), round(volume_pred, 3), round(volume_, 3), count_gt, count_pred, count_, fp, folder])
        results.append([case_id[:-7], round(dice_, 3), round(f1_, 3), round(volume_gt, 3), round(volume_pred, 3), round(volume_, 3), count_gt, count_pred, count_, fp, tp, fn, folder])

    columns = ['id', 'Dice', 'F1', 'Volume_GT', 'Volume_PRED', 'VolumeDiff', 'LesionCount_GT', 'LesionCount_PRED', 'LesionCountDiff', 'FP', 'TP', 'FN', 'CMB_Status']
    df_eval = pd.DataFrame(results, columns=columns)

    pos_df = df_eval[df_eval["CMB_Status"] == "CMB_Pos"]
    neg_df = df_eval[df_eval["CMB_Status"] == "CMB_Neg"]

    thresholds = [0, 1, 2, 3, 5, 8, 10]
    froc_pos = compute_froc_pos(pos_df, thresholds)
    print("\n📊 [FROC_Pos 디버깅] - False Positives per Image vs Sensitivity")
    for th, score in zip(thresholds, froc_pos):
        print(f"FP ≤ {th}  →  Sensitivity = {score:.4f}")

    froc_neg = compute_froc_neg(neg_df, thresholds)
    froc_all = compute_froc_all(pos_df, neg_df, thresholds)

    sensitivity_for_CMB_pos = pos_df['TP'].sum() / (pos_df['TP'].sum() + pos_df['FN'].sum()) if (pos_df['TP'].sum() + pos_df['FN'].sum()) > 0 else 0
    precision_for_CMB_pos = pos_df['TP'].sum() / (pos_df['TP'].sum() + pos_df['FP'].sum()) if (pos_df['TP'].sum() + pos_df['FP'].sum()) > 0 else 0
    efp_for_CMB_pos = pos_df['FP'].sum() / len(pos_df) if len(pos_df) > 0 else 0
    efp_for_CMB_neg = neg_df['FP'].sum() / len(neg_df) if len(neg_df) > 0 else 0

    summary_items = [
        ("avg_dice_for_CMB_pos", pos_df["Dice"].mean()),
        ("count_diff_for_CMB_pos", int(pos_df["LesionCountDiff"].abs().sum())),
        ("avg_count_diff_for_CMB_pos", float(pos_df["LesionCountDiff"].abs().mean())),
        ("vol_diff_for_CMB_pos", float(pos_df["VolumeDiff"].abs().sum())),
        ("avg_vol_diff_for_CMB_pos", float(pos_df["VolumeDiff"].abs().mean())),
        ("Lesion_F1_for_CMB_pos", pos_df["F1"].mean()),
        ("", ""),
        ("Specificity_for_CMB_Neg", neg_df.apply(lambda row: row["FP"] == 0, axis=1).sum() / len(neg_df) if len(neg_df) > 0 else 0),
        ("vol_diff_for_CMB_neg", float(neg_df["VolumeDiff"].abs().sum())),
        ("avg_vol_diff_for_CMB_neg", float(neg_df["VolumeDiff"].abs().mean())),
        ("count_diff_for_CMB_neg", int(neg_df["LesionCountDiff"].abs().sum())),
        ("avg_count_diff_for_CMB_neg", float(neg_df["LesionCountDiff"].abs().mean())),
        ("", ""),
        ("fROC_for_CMB_Pos", froc_pos[2]),
        ("fROC_for_CMB_Neg", froc_neg[2]),
        ("fROC_for_CMB_All", froc_all[2]),
        ("Sensitivity_for_CMB_Pos (%)", round(sensitivity_for_CMB_pos * 100, 3)),
        ("Precision_for_CMB_Pos (%)", round(precision_for_CMB_pos * 100, 3)),
        ("EFP_for_CMB_Pos", efp_for_CMB_pos),
        ("EFP_for_CMB_Neg", efp_for_CMB_neg),
    ]

    start_row = len(t_) + 5
    for i, (k, v) in enumerate(summary_items):
        sheet1[f"A{start_row + i}"] = k
        sheet1[f"B{start_row + i}"] = "" if k == "" else round(v, 3)

    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, froc_pos, label='CMB_Pos', marker='o', linestyle='-', markersize=8, linewidth=2)
    plt.plot(thresholds, froc_neg, label='CMB_Neg', marker='o', linestyle='-', markersize=8, linewidth=2)
    plt.plot(thresholds, froc_all, label='CMB_All', marker='o', linestyle='-', markersize=3, linewidth=1, color='red')
    plt.xlabel("False Positives per Image")
    plt.ylabel("Sensitivity (Patient-level)")
    plt.title("FROC Curves")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    froc_img_path = os.path.join(save_dir, froc_img_name)
    plt.savefig(froc_img_path)
    plt.close()

    froc_img = XLImage(froc_img_path)
    sheet1.add_image(froc_img, f"A{start_row + len(summary_items) + 2}")

    save_path = os.path.join(save_dir, xlsx_name)
    wb.save(save_path)
    print(f"결과 저장 완료: {save_path}")
    return save_path

# 실행
save_dir = '/workspace/7_QSM_CMB/nnUNet/nnUNet_RESULTS_FOLDER/results_chwise_swi_1ch_external_0411'
csv_path = "/workspace/7_QSM_CMB/nnUNet/data/swi_external_resampled_val.csv"
truth_path = '/workspace/7_QSM_CMB/nnUNet/nnUNet_raw/Dataset515_CMB_swi_External_resampled/labelsTs'
df = pd.read_csv(csv_path)
id_to_folder = dict(zip(df["CMB_ID"], df["folder"]))

prediction_path = '/workspace/7_QSM_CMB/nnUNet/nnUNet_pred2amcext_swi_chwise/Dataset515_CMB_swi_External_resampled_predictions_scrfinal'
froc_img_name = 'froc_curve_scrfinal.png'
xlsx_name = 'swi_1ch_scrfinal.xlsx'
evaluate(truth_path, prediction_path, id_to_folder, froc_img_name, xlsx_name, save_dir)

prediction_path = '/workspace/7_QSM_CMB/nnUNet/nnUNet_pred2amcext_swi_chwise/Dataset515_CMB_swi_External_resampled_predictions_anaftfinal'
froc_img_name = 'froc_curve_anaftfinal.png'
xlsx_name = 'swi_1ch_anaftfinal.xlsx'
evaluate(truth_path, prediction_path, id_to_folder, froc_img_name, xlsx_name, save_dir)

prediction_path = '/workspace/7_QSM_CMB/nnUNet/nnUNet_pred2amcext_swi_chwise/Dataset515_CMB_swi_External_resampled_predictions_bratsflairftfinal'
froc_img_name = 'froc_curve_bratsflairftfinal.png'
xlsx_name = 'swi_1ch_bratsflairftfinal.xlsx'
evaluate(truth_path, prediction_path, id_to_folder, froc_img_name, xlsx_name, save_dir)

prediction_path = '/workspace/7_QSM_CMB/nnUNet/nnUNet_pred2amcext_swi_chwise/Dataset515_CMB_swi_External_resampled_predictions_bratst2ftfinal'
froc_img_name = 'froc_curve_bratst2ftfinal.png'
xlsx_name = 'swi_1ch_bratst2ftfinal.xlsx'
evaluate(truth_path, prediction_path, id_to_folder, froc_img_name, xlsx_name, save_dir)

prediction_path = '/workspace/7_QSM_CMB/nnUNet/nnUNet_pred2amcext_swi_chwise/Dataset515_CMB_swi_External_resampled_predictions_strokeftfinal'
froc_img_name = 'froc_curve_strokeftfinal.png'
xlsx_name = 'swi_1ch_strokeftfinal.xlsx'
evaluate(truth_path, prediction_path, id_to_folder, froc_img_name, xlsx_name, save_dir)
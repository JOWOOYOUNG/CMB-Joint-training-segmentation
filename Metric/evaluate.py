from glob import glob
import SimpleITK as sitk
from isles.scoring import dice_coef, volume_difference, simple_lesion_count_difference, lesion_f1_score
import openpyxl
from tqdm import tqdm
import numpy as np

# 경로 설정
truth_path = 'workspace/1_nnUNet/nnUNet_raw/Dataset400_CMB/labelsTs'
prediction_path = 'workspace/1_nnUNet/nnUNet_pred/Dataset400_CMB_predictions'

def evaluate(truth_path, prediction_path, id_to_folder):
    t_ = sorted(glob(f'{truth_path}/*.nii.gz'))
    p_ = sorted(glob(f'{prediction_path}/*.nii.gz'))

    wb = openpyxl.Workbook()
    sheet1 = wb['Sheet']
    sheet1.title = 'Evaluation'
    sheet1['A1'] = 'id'
    sheet1['B1'] = 'Dice'
    sheet1['C1'] = 'F1'
    sheet1['D1'] = 'VolumeDiff'
    sheet1['E1'] = 'LesionCount'
    sheet1['F1'] = 'FP'
    sheet1['G1'] = 'CMB_Status'

    for idx in tqdm(range(len(t_))):
        case_id = t_[idx].split('/')[-1]  # e.g., CMB_test_1234.nii.gz
        truth_ = sitk.GetArrayFromImage(sitk.ReadImage(str(t_[idx])))
        pred_ = sitk.GetArrayFromImage(sitk.ReadImage(str(p_[idx])))

        dice_ = dice_coef(truth=truth_, prediction=pred_)
        volume_ = volume_difference(truth=truth_, prediction=pred_)
        count_ = simple_lesion_count_difference(truth=truth_, prediction=pred_)

        folder = id_to_folder.get(case_id, 'Unknown')
        if folder == 'CMB_Neg':
            _, fp = scipy.ndimage.label(pred_.astype(bool))
            f1_ = 1.0  # 완전 음성인 경우 F1은 1로 간주
        else:
            f1_, fp = lesion_f1_score(truth=truth_, prediction=pred_)

        sheet1.append([case_id[:-7], dice_, f1_, volume_, count_, fp, folder])

    save_path = 'workspace/1_nnUNet/nnUNet_RESULTS_FOLDER/Dataset400_CMB_validation'
    wb.save(save_path)
    print(f"📁 결과 저장 완료: {save_path}")


#test
# p = 'workspace/1_nnUNet/nnUNet_pred/Dataset400_CMB_predictions' 
# t = 'workspace/1_nnUNet/nnUNet_raw/Dataset400_CMB/labelsTs' 
# evaluate_(t, p)
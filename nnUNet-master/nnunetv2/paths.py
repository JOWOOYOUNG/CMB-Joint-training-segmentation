import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

# 환경 변수에서 경로 읽기
nnUNet_raw = os.environ.get('nnUNet_raw')
nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
nnUNet_results = os.environ.get('nnUNet_results')

# ❗ 추가: V1 방식처럼 서브폴더 포함한 경로 만들기
if nnUNet_raw is not None:
    nnUNet_raw_data = join(nnUNet_raw, "nnUNet_raw_data")
    nnUNet_cropped_data = join(nnUNet_raw, "nnUNet_cropped_data")
    maybe_mkdir_p(nnUNet_raw_data)
    maybe_mkdir_p(nnUNet_cropped_data)
else:
    nnUNet_raw_data = nnUNet_cropped_data = None
    print("nnUNet_raw is not defined. Please set this env variable.")

if nnUNet_preprocessed is not None:
    preprocessing_output_dir = nnUNet_preprocessed
    maybe_mkdir_p(preprocessing_output_dir)
else:
    preprocessing_output_dir = None
    print("nnUNet_preprocessed is not defined. Please set this env variable.")

if nnUNet_results is not None:
    network_training_output_dir = join(nnUNet_results, "nnUNet")
    maybe_mkdir_p(network_training_output_dir)
else:
    network_training_output_dir = None
    print("nnUNet_results is not defined. Please set this env variable.")

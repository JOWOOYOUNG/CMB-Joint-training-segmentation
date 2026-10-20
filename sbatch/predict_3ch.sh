#!/bin/bash
#SBATCH --job-name nnUNET_predict_ensemble_3ch
#SBATCH -t 7-00:00:00 
#SBATCH -o ./logs/nnUNET_predict_ensemble_3ch/%x_%A_%N_.out
#SBATCH --mail-type BEGIN,END,TIME_LIMIT_90,REQUEUE,INVALID_DEPEND
#SBATCH --mail-user chowooyoung0513@gmail.com
#SBATCH --nodes=1                # 2개의 노드를 요청
#SBATCH -w gpu16
#SBATCH --gres=gpu:1                # 총 GPU 2개
#SBATCH -p TitanRTX               # V100 GPU 풀에서 실행
#SBATCH --exclude gpu10



# Define Variables
DOCKER_IMAGE="localhost/$(echo "$SLURM_JOB_NAME" | tr '[:upper:]' '[:lower:]')"

# container 생성
# Start Container
docker build . -t $DOCKER_IMAGE -f ./vscode3.Dockerfile
docker run -i --rm --device nvidia.com/gpu=all --shm-size 128GB \
    -p 9300:9300 \
    -v /mnt/nas203/forGPU2/Wooyoung/0_project/4_QSM/:/workspace/ \
    $DOCKER_IMAGE \
        bash -c " 
        pwd && ls -l &&\
        pip install --upgrade tensorboardX monai==1.1.0 nibabel einops natsort SimpleITK tqdm==4.59.0 openpyxl && \
        pip install hiddenlayer && \
        cd /workspace/2_nnUNet/nnUNet-master && \
        pip install -e . && \
        export nnUNet_raw="/workspace/2_nnUNet/nnUNet_raw" && \
        export nnUNet_preprocessed="/workspace/2_nnUNet/nnUNet_preprocessed" && \
        export nnUNet_results="/workspace/2_nnUNet/nnUNet_results_ensemble" && \
        export JOINT_DATASET_DIRS="/workspace/2_nnUNet/nnUNet_preprocessed/Dataset302_1ch_qsm,/workspace/2_nnUNet/nnUNet_preprocessed/Dataset301_1ch_swi,/workspace/2_nnUNet/nnUNet_preprocessed/Dataset300_1ch_phase" && \
        export JOINT_PLANS_FILES="/workspace/2_nnUNet/nnUNet_preprocessed/Dataset302_1ch_qsm/nnUNetResEncUNetLPlans.json,/workspace/2_nnUNet/nnUNet_preprocessed/Dataset301_1ch_swi/nnUNetResEncUNetLPlans.json,/workspace/2_nnUNet/nnUNet_preprocessed/Dataset300_1ch_phase/nnUNetResEncUNetLPlans.json" && \
        cd /workspace/2_nnUNet/nnUNet-master/nnunetv2/inference && \
        export OMP_NUM_THREADS=1 && \
        export MKL_NUM_THREADS=1 && \
        export OPENBLAS_NUM_THREADS=1 && \
        export NUMEXPR_NUM_THREADS=1 && \
        python joint_predictor_entry.py \
        -i /workspace/2_nnUNet/nnUNet_raw/Dataset301_1ch_swi/imagesTs \
        -o /workspace/2_nnUNet/nnUNet_pred_AMC_INT/fold01234/JointPred_3ch_phase_swi_qsm_2opti_final \
        -m /workspace/2_nnUNet/nnUNet_results_ensemble/JointDataset__phase__swi__qsm/JointTrainer__nnUNetResEncUNetLPlans__3d_fullres_2opti \
        -chk checkpoint_final.pth \
        -tr JointTrainer \
        -p nnUNetResEncUNetLPlans \
        --ensemble \
        --modality_idx 1 && \
        python joint_predictor_entry.py \
        -i /workspace/1_nnUNet/previous_nnUNet/previous_nnUNet_raw/Asan_weak_external/imagesTs \
        -o /workspace/2_nnUNet/nnUNet_pred_AMC_weak_EXT/fold01234/JointPred_3ch_phase_swi_qsm_2opti_final \
        -m /workspace/2_nnUNet/nnUNet_results_ensemble/JointDataset__phase__swi__qsm/JointTrainer__nnUNetResEncUNetLPlans__3d_fullres_2opti \
        -chk checkpoint_final.pth \
        -tr JointTrainer \
        -p nnUNetResEncUNetLPlans \
        --ensemble \
        --modality_idx 1 && \
        python joint_predictor_entry.py \
        -i /workspace/data/External_resampled/imagesTs_swi \
        -o /workspace/2_nnUNet/nnUNet_pred_EXT/fold01234/JointPred_3ch_phase_swi_qsm_2opti_final \
        -m /workspace/2_nnUNet/nnUNet_results_ensemble/JointDataset__phase__swi__qsm/JointTrainer__nnUNetResEncUNetLPlans__3d_fullres_2opti \
        -chk checkpoint_final.pth \
        -tr JointTrainer \
        -p nnUNetResEncUNetLPlans \
        --ensemble \
        --modality_idx 1       
        "

#!/bin/bash
#SBATCH --job-name nnUNET_QSM
#SBATCH -t 7-00:00:00 
#SBATCH -o ./logs/nnUNET_QSM/%x_%A_%N_.out
#SBATCH --mail-type BEGIN,END,TIME_LIMIT_90,REQUEUE,INVALID_DEPEND
#SBATCH --mail-user chowooyoung0513@gmail.com
#SBATCH -w gpu33
#SBATCH --exclude gpu102
#SBATCH -p RTX3090
#SBATCH --gpus 1

# Define Variables
DOCKER_IMAGE="localhost/$(echo "$SLURM_JOB_NAME" | tr '[:upper:]' '[:lower:]')"

# container 생성
# Start Container
docker build . -t $DOCKER_IMAGE -f ./vscode3.Dockerfile
docker run -i --rm --device nvidia.com/gpu=all --shm-size 128GB \
    -p 9500:9500 \
    -v /mnt/nas203/forGPU2/Wooyoung/0_project/4_QSM/:/workspace/ \
    $DOCKER_IMAGE \
        bash -c " 
        pwd && ls -l &&\
        pip install --upgrade tensorboardX monai==1.1.0 nibabel einops natsort SimpleITK tqdm==4.59.0 openpyxl && \
        cd /workspace/1_nnUNet/nnUNet-master && \
        pip install -e . &&\
        export nnUNet_raw="/workspace/1_nnUNet/nnUNet_raw" && \
        export nnUNet_preprocessed="/workspace/1_nnUNet/nnUNet_preprocessed" && \
        export nnUNet_results="/workspace/1_nnUNet/nnUNet_results_scr" && \
        export OPENBLAS_NUM_THREADS=16 && \
        export TORCHINDUCTOR_COMPILE_THREADS=1 && \
        nnUNetv2_train 400 3d_fullres all -p nnUNetResEncUNetLPlans --c --npz
        "

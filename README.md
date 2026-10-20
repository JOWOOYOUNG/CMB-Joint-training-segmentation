# Joint training for SWI-only cerebral microbleed segmentation

Official code for **"Enhancing Cerebral Microbleeds Detection via Jointly Incorporating Multiple MR Sequences Validated in Multicenter SWI-only Inference"** (submitted to *Artificial Intelligence in Medicine*).

> **TL;DR.** We train cerebral microbleed (CMB) segmentation models with SWI, filtered phase and QSM, and deploy them with **SWI alone**. Channel-wise multi-sequence nnU-Nets collapse when only SWI is available at test time (Dice 0.448 / 0.367 on two external cohorts). Joint training avoids this collapse (Dice 0.630 / 0.536, both p < 0.001) and matches or modestly exceeds an SWI-only baseline, including on a 48-institution external cohort.

**Contents:**
[Overview](#overview) ·
[Built on nnU-Net](#built-on-nnu-net-what-we-changed) ·
[Core files](#core-files) ·
[How joint training is implemented](#how-joint-training-is-implemented) ·
[Quick start](#quick-start) ·
[Results](#key-results) ·
[Citation](#citation)

---

## Overview

- **CMB counts gate anti-amyloid therapy.** Appropriate use recommendations for lecanemab exclude patients with more than four CMBs.
- **QSM helps but is rarely acquired.** It separates hemorrhage from veins and calcification better than SWI, yet most hospitals only have SWI.
- **Multi-sequence models usually assume every input at test time.** We test what happens when only SWI is available, and provide a training scheme that is deployed with SWI alone.

| Strategy | Training input | Inference input | Code |
|---|---|---|---|
| SWI-only baseline | SWI | SWI | standard `nnUNetTrainer` |
| Channel-wise | SWI, phase, QSM concatenated as channels | SWI | standard `nnUNetTrainer`, multi-channel dataset |
| **Joint (ours)** | SWI, phase, QSM, one network each | **SWI** | `JointTrainer` + `JointPredictor` (this repository) |

---

## Built on nnU-Net: what we changed

This repository is a fork of **[nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) (v2.6)** with the residual-encoder preset (**ResEnc-L**). Dataset format, experiment planning, preprocessing, network architecture, data augmentation, loss, sliding-window inference and test-time mirroring are nnU-Net's own. If you know nnU-Net, you already know how to run this code.

We added one trainer, one predictor and a few command-line options:

---

## Core files

Only these files are specific to this project. Everything else under `nnUNet-master/` comes from nnU-Net v2.

| File | Role |
|---|---|
| [`nnunetv2/training/nnUNetTrainer/Joint_nnUNetTrainer.py`](nnUNet-master/nnunetv2/training/nnUNetTrainer/Joint_nnUNetTrainer.py) | **`JointTrainer`**: builds one network, optimizer and data loader per sequence and trains them together |
| [`nnunetv2/run/run_training.py`](nnUNet-master/nnunetv2/run/run_training.py#L57) | `nnUNetv2_train` entry point; adds `--joint_dataset_dirs`, `--joint_plans_files`, `--use_meanloss_for_multitask`, `--primary_network_index` and constructs `JointTrainer` |
| [`nnunetv2/inference/joint_predictor_entry.py`](nnUNet-master/nnunetv2/inference/joint_predictor_entry.py#L126) | Command-line prediction for joint models (`--modality_idx` selects the network) |
| [`nnunetv2/inference/predictor_classes/joint_predictorV2.py`](nnUNet-master/nnunetv2/inference/predictor_classes/joint_predictorV2.py#L69) | **`JointPredictor`**: loads one network from a joint checkpoint; inherits nnU-Net's `nnUNetPredictor` |
| [`Metric/`](Metric) | Evaluation: Dice, lesion-level F1, lesion count and volume differences |
| [`sbatch/`](sbatch) | SLURM scripts used to generate the paper's predictions |

---

## How joint training is implemented

All references point to [`Joint_nnUNetTrainer.py`](nnUNet-master/nnunetv2/training/nnUNetTrainer/Joint_nnUNetTrainer.py). `JointTrainer` subclasses nnU-Net's `nnUNetTrainer` and overrides the methods below.

| Step | Method | What it does |
|---|---|---|
| 1. Setup | [`__init__`](nnUNet-master/nnunetv2/training/nnUNetTrainer/Joint_nnUNetTrainer.py#L75) | Reads the per-sequence dataset folders and plans files; sets the initial LR to 1e-4 and the primary (SWI) network index to 1 |
| 2. Networks | [`initialize`](nnUNet-master/nnunetv2/training/nnUNetTrainer/Joint_nnUNetTrainer.py#L131) | Builds one single-channel ResEnc-L network and one SGD optimizer (Nesterov, momentum 0.99) per sequence |
| 3. Split | [`do_split`](nnUNet-master/nnunetv2/training/nnUNetTrainer/Joint_nnUNetTrainer.py#L448) | Uses one 5-fold split (seed 12345) for all sequences |
| 4. Data | [`get_dataloaders`](nnUNet-master/nnunetv2/training/nnUNetTrainer/Joint_nnUNetTrainer.py#L534) | Creates one nnU-Net data loader with standard augmentation per sequence |
| 5. Training step | [`train_step`](nnUNet-master/nnunetv2/training/nnUNetTrainer/Joint_nnUNetTrainer.py#L911) | Each network predicts on its own sequence; Dice + CE per network; losses averaged; one backward pass; gradient clipping (12) per network; every optimizer steps |
| 6. Validation | [`validation_step`](nnUNet-master/nnunetv2/training/nnUNetTrainer/Joint_nnUNetTrainer.py#L990) | Online pseudo-Dice and per-sequence validation losses |
| 7. Checkpoints | [`on_epoch_end`](nnUNet-master/nnunetv2/training/nnUNetTrainer/Joint_nnUNetTrainer.py#L1125), [`save_checkpoint`](nnUNet-master/nnunetv2/training/nnUNetTrainer/Joint_nnUNetTrainer.py#L1227) | Saves all networks in one file; besides `checkpoint_best.pth` and `checkpoint_final.pth`, also keeps `checkpoint_best_meanloss.pth` and `checkpoint_best_swi_loss.pth` |
| 8. Final validation | [`perform_actual_validation`](nnUNet-master/nnunetv2/training/nnUNetTrainer/Joint_nnUNetTrainer.py#L1317) | Runs nnU-Net's sliding-window prediction with the SWI network only |

At inference, [`JointPredictor.initialize_from_trained_model_folder`](nnUNet-master/nnunetv2/inference/predictor_classes/joint_predictorV2.py#L69) reads `network_weights[modality_idx]` from the checkpoint, rebuilds that network from the saved plans and hands it to nnU-Net's standard predictor (Gaussian-weighted sliding window, mirroring test-time augmentation).

---

## Quick start

The workflow is the standard nnU-Net v2 workflow. Only two steps differ: training uses `-tr JointTrainer` with one dataset per sequence, and prediction uses `joint_predictor_entry.py`.

| Step | Command | Same as nnU-Net? |
|---|---|---|
| 1. Install | `pip install -e nnUNet-master` | yes |
| 2. Prepare data | one nnU-Net dataset per sequence | dataset format yes; one dataset per sequence is ours |
| 3. Plan and preprocess | `nnUNetv2_plan_and_preprocess` | **identical** |
| 4. Train | `nnUNetv2_train … -tr JointTrainer` | **joint training** |
| 5. Predict (SWI only) | `python -m nnunetv2.inference.joint_predictor_entry …` | **joint prediction** |
| 6. Evaluate | `Metric/` | ours |

### 1. Install

We ran all experiments in a Docker container based on `pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel` (Python 3.10, PyTorch 2.5.1, CUDA 11.8).

```bash
git clone https://github.com/JOWOOYOUNG/CMB-Joint-training-segmentation.git
cd CMB-Joint-training-segmentation/nnUNet-master
pip install -e .

# extra packages installed in our container
pip install tensorboardX monai==1.1.0 nibabel einops natsort SimpleITK tqdm==4.59.0 openpyxl hiddenlayer
```

Set the standard nnU-Net paths:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

### 2. Prepare data

Create **one nnU-Net dataset per sequence**. All three datasets contain the same case IDs and the same binary CMB label maps (0 = background, 1 = CMB). See nnU-Net's [dataset format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).

```
nnUNet_raw/
├── Dataset300_1ch_phase/   imagesTr/  labelsTr/  dataset.json
├── Dataset301_1ch_swi/     imagesTr/  labelsTr/  imagesTs/  dataset.json
└── Dataset302_1ch_qsm/     imagesTr/  labelsTr/  dataset.json
```

Each image is a single channel named `CASE_0000.nii.gz`. Test images (`imagesTs/`) are needed for SWI only.

Before nnU-Net, images were preprocessed as described in paper Sec. 2.3: DICOM to NIfTI conversion, brain extraction with [HD-BET](https://github.com/MIC-DKFZ/HD-BET), N4 bias-field correction, and QSM reconstruction with the χ-separation toolbox followed by z-score normalization.

### 3. Plan and preprocess

This step is unchanged from nnU-Net. Run it once for the three datasets with the ResEnc-L preset:

```bash
nnUNetv2_plan_and_preprocess -d 300 301 302 -pl nnUNetPlannerResEncL --verify_dataset_integrity
```

All sequences must share one cross-validation split. `JointTrainer` reads `splits_final.json` from the first dataset's preprocessed folder (`Dataset300_1ch_phase`); keep the same file in the other two folders.

### 4. Train (joint training)

Give one preprocessed dataset folder and one plans file per sequence, in the same order. **SWI must be the second entry (index 1).** The paper's three-sequence model uses the order phase, SWI, QSM.

```bash
P=$nnUNet_preprocessed
export JOINT_DATASET_DIRS="$P/Dataset300_1ch_phase,$P/Dataset301_1ch_swi,$P/Dataset302_1ch_qsm"
export JOINT_PLANS_FILES="$P/Dataset300_1ch_phase/nnUNetResEncUNetLPlans.json,$P/Dataset301_1ch_swi/nnUNetResEncUNetLPlans.json,$P/Dataset302_1ch_qsm/nnUNetResEncUNetLPlans.json"

for FOLD in 0 1 2 3 4; do
  nnUNetv2_train 301 3d_fullres $FOLD -tr JointTrainer -p nnUNetResEncUNetLPlans \
    --use_meanloss_for_multitask \
    --primary_network_index 1 \
    --npz
done
```

| Option | Meaning |
|---|---|
| `-tr JointTrainer` | Use the joint trainer instead of `nnUNetTrainer` |
| `JOINT_DATASET_DIRS` | Comma-separated preprocessed dataset folders, one per sequence (or `--joint_dataset_dirs`) |
| `JOINT_PLANS_FILES` | Comma-separated plans files in the same order (or `--joint_plans_files`) |
| `--use_meanloss_for_multitask` | Average the per-network losses; without it they are summed |
| `--primary_network_index 1` | Network used for validation and inference (SWI) |
| `--npz` | Save validation probabilities (standard nnU-Net option) |
| `--c` | Resume an interrupted run (standard nnU-Net option) |

For two-sequence models keep SWI at index 1, for example `phase,swi` or `qsm,swi`. Checkpoints are written to:

```
$nnUNet_results/JointDataset__phase__swi__qsm/JointTrainer__nnUNetResEncUNetLPlans__3d_fullres_2opti/fold_X/
```

### 5. Predict with SWI only

Only SWI images are needed. `--modality_idx 1` selects the SWI network from the joint checkpoint.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
MODEL=$nnUNet_results/JointDataset__phase__swi__qsm/JointTrainer__nnUNetResEncUNetLPlans__3d_fullres_2opti

python -m nnunetv2.inference.joint_predictor_entry \
  -i $nnUNet_raw/Dataset301_1ch_swi/imagesTs \
  -o /path/to/predictions/internal_test \
  -m $MODEL \
  -tr JointTrainer \
  -chk checkpoint_final.pth \
  --modality_idx 1
```

Run the same command with `-i` pointing at each external test set (SWI only). By default all five fold models are used, and each writes to `fold_0/` … `fold_4/` under the output folder; use `-f 0` for a single fold. Equivalently, run `python joint_predictor_entry.py` from `nnUNet-master/nnunetv2/inference/`.

| Option | Meaning |
|---|---|
| `-i` / `-o` | Input folder of SWI images (`CASE_0000.nii.gz`) / output folder |
| `-m` | Trained joint model folder (contains `fold_0` … `fold_4`) |
| `-tr JointTrainer` | Load the checkpoint as a joint model |
| `-chk` | Checkpoint to use (`checkpoint_final.pth` in the paper) |
| `--modality_idx 1` | Network to load from the joint checkpoint (SWI) |
| `-f` | Folds to use (default: 0 1 2 3 4) |
| `--disable_tta` | Turn off mirroring test-time augmentation |

### 6. Evaluate

| File | What it computes |
|---|---|
| `Metric/scoring.py` | Per-case Dice, lesion-level F1, lesion count difference and volume difference |
| `Metric/evaluate.py` | Runs `scoring.py` over a folder of reference labels and a folder of predictions and writes one row per case to an Excel sheet |
| `Metric/Metric.ipynb` | Cohort-level metrics reported in the paper |

### Baselines

Both baselines use nnU-Net's standard trainer and predictor.

```bash
# SWI-only baseline
nnUNetv2_train 301 3d_fullres $FOLD -p nnUNetResEncUNetLPlans --npz

# Channel-wise: one dataset whose imagesTr holds SWI, phase and QSM as _0000, _0001, _0002
nnUNetv2_train <multichannel_dataset_id> 3d_fullres $FOLD -p nnUNetResEncUNetLPlans --npz

# Prediction
nnUNetv2_predict -i <input_folder> -o <output_folder> -d <dataset_id> -c 3d_fullres -p nnUNetResEncUNetLPlans
```
Key results

Dice similarity coefficient (DSC) / lesion-level F1, mean [SD] across five cross-validation models. All models are evaluated with SWI only.

Model	Internal test (n=74)	Temporally independent (n=334)	Cross-institutional, 48 sites (n=65)
SWI-only baseline	0.839 [0.012] / 0.854 [0.014]	0.627 [0.007] / 0.732 [0.008]	0.514 [0.024] / 0.605 [0.038]
Channel-wise, SWI+phase+QSM	0.826 [0.011] / 0.860 [0.015]	0.448 [0.030] / 0.572 [0.029]	0.367 [0.060] / 0.460 [0.074]
Joint, SWI+phase+QSM (ours)	0.833 [0.029] / 0.860 [0.018]	0.630 [0.007] / 0.734 [0.009]	0.536 [0.011] / 0.617 [0.020]

In the cross-institutional cohort, joint training also improved three-class CMB burden classification (≤4 / 5–9 / ≥10 CMBs): balanced accuracy 0.545 vs. 0.444 for the baseline. The cohort covers Philips, Siemens and GE scanners at 1.5 T and 3 T with slice thickness from 0.8 to 5 mm.
---

## Repository structure

```
├── nnUNet-master/                              # nnU-Net v2.6 with joint-training support
│   └── nnunetv2/
│       ├── training/nnUNetTrainer/
│       │   └── Joint_nnUNetTrainer.py          # JointTrainer
│       ├── run/
│       │   └── run_training.py                 # nnUNetv2_train with joint options
│       └── inference/
│           ├── joint_predictor_entry.py        # SWI-only prediction (command line)
│           └── predictor_classes/
│               └── joint_predictorV2.py        # JointPredictor
├── Metric/                                     # evaluation
└── sbatch/                                     # SLURM scripts used for the paper

## Data availability

The MRI data cannot be shared publicly because of patient privacy and institutional restrictions. They are available from the corresponding authors on reasonable request, subject to approval by the Institutional Review Board of Asan Medical Center.

## Citation

```bibtex
@article{jo2026cmbjoint,
  title   = {Enhancing Cerebral Microbleeds Detection via Jointly Incorporating Multiple MR Sequences Validated in Multicenter SWI-only Inference},
  author  = {Jo, Wooyoung and Oh, Joonseo and Won, Jongjun and Lee, Hyunju and Park, Hyungbin and Sung, Jimin and Won, Jonghun and Choi, Yangsean and Kim, Namkug},
  journal = {Artificial Intelligence in Medicine},
  note    = {Under review},
  year    = {2026}
}
```

Please also cite nnU-Net:

```bibtex
@article{isensee2021nnunet,
  title   = {nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author  = {Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and Petersen, Jens and Maier-Hein, Klaus H.},
  journal = {Nature Methods},
  volume  = {18},
  number  = {2},
  pages   = {203--211},
  year    = {2021}
}
```

## Acknowledgements

Built on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). Brain extraction uses [HD-BET](https://github.com/MIC-DKFZ/HD-BET); QSM uses χ-separation (Min et al., *NMR Biomed* 2024). The joint-training idea is adapted from Jeong et al., *J Imaging Inform Med* 2024. Supported by the Asan Institute for Life Sciences, Asan Medical Center (2025IP0049-1, 2025IP0062-1).

## License

The nnU-Net code in `nnUNet-master/` is distributed under the Apache License 2.0 (see [`nnUNet-master/LICENSE`](nnUNet-master/LICENSE)).

## Contact

- Wooyoung Jo, University of Ulsan College of Medicine (co-first author)
- Namkug Kim, PhD, and Yangsean Choi, PhD, Asan Medical Center (co-corresponding authors): namkugkim@gmail.com, Phillipchoi007@gmail.com

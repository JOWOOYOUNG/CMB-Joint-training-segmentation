import torch
import multiprocessing
from os.path import isdir
from nnunetv2.utilities.file_path_utilities import maybe_mkdir_p
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.predictor_classes.joint_predictorV2 import JointPredictor
import inspect
import itertools
import multiprocessing
import os
from copy import deepcopy
from queue import Queue
from threading import Thread
from time import sleep
from typing import Tuple, Union, List, Optional

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy, preprocessing_iterator_fromfiles, \
    preprocessing_iterator_fromnpy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
from nnunetv2.ensembling.ensemble import ensemble_crossvalidations

def predict_entry_point_modelfolder():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-m', type=str, required=True,
                        help='Folder in which the trained model is. Must have subfolders fold_X for the different '
                             'folds you trained')
    parser.add_argument('-f', nargs='+', type=int, default=[0, 1, 2, 3, 4],
                        help='Specify the folds of the trained model that should be used for prediction. Default: all (0-4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', '--c', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')
    parser.add_argument('-tr', '--trainer_class_name', type=str, default='nnUNetTrainer', help='Trainer class name')
    
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble mode for JointPredictor')
    
    parser.add_argument('--modality_idx', type=int, default=0, help='Modality index for JointPredictor')


    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    if args.trainer_class_name in ['JointTrainer', 'JointTrainer2']:
        from nnunetv2.inference.predictor_classes.joint_predictorV2 import JointPredictor
        for f in args.f:
            # ✅ fold별 출력 폴더 생성
            output_folder_fold = os.path.join(args.o, f"fold_{f}")
            maybe_mkdir_p(output_folder_fold)

            predictor = JointPredictor(
                tile_step_size=args.step_size,
                use_gaussian=True,
                use_mirroring=not args.disable_tta,
                perform_everything_on_device=True,
                device=device,
                verbose=args.verbose,
                allow_tqdm=not args.disable_progress_bar,
                verbose_preprocessing=args.verbose
            )
            predictor.initialize_from_trained_model_folder(
                model_training_output_dir=args.m,
                use_folds=[f],
                checkpoint_name=args.chk,
                primary_network_index=args.modality_idx,
                use_ensemble=args.ensemble,
            )
            predictor.predict_from_files(
                args.i,
                output_folder_fold,
                save_probabilities=args.save_probabilities,
                overwrite=not args.continue_prediction,
                num_processes_preprocessing=args.npp,
                num_processes_segmentation_export=args.nps,
                folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                num_parts=1,
                part_id=0
            )
    else:
        predictor = nnUNetPredictor(
            tile_step_size=args.step_size,
            use_gaussian=True,
            use_mirroring=not args.disable_tta,
            perform_everything_on_device=True,
            device=device,
            verbose=args.verbose,
            allow_tqdm=not args.disable_progress_bar,
            verbose_preprocessing=args.verbose
        )
        predictor.initialize_from_trained_model_folder(args.m, args.f, args.chk)
        predictor.predict_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
                                overwrite=not args.continue_prediction,
                                num_processes_preprocessing=args.npp,
                                num_processes_segmentation_export=args.nps,
                                folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                num_parts=1, part_id=0)
                # --- 예측 완료 후 앙상블 수행 ---
    if args.ensemble:
        print("Starting ensemble from predicted folds...")
        ensemble_output_dir = os.path.join(args.o, "ensemble_result")
        maybe_mkdir_p(ensemble_output_dir)

        # 각 fold의 validation 폴더 경로 생성
        fold_dirs = [os.path.join(args.m, f"fold_{f}", "validation") for f in args.f if isinstance(f, int)]

        # 실제 모델 폴더 (fold_X 상위 폴더)를 전달
        ensemble_crossvalidations(
            list_of_trained_model_folders=[args.m],
            output_folder=ensemble_output_dir,
            folds=args.f,
            num_processes=args.nps
        )

        print(f"Ensemble segmentation saved to: {ensemble_output_dir}")

if __name__ == "__main__":
    predict_entry_point_modelfolder()

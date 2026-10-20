# ✅ 통합된 JointPredictor: 기존 nnUNetPredictor 기능을 최대한 활용하며 ensemble도 지원
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
from typing import Optional, Tuple
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import argparse
from os.path import join, basename
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from nnunetv2.paths import nnUNet_results


import glob
def collect_case_paths_from_folder(input_folder: str) -> List[List[str]]:
    """
    input_folder 내 *_0000.nii.gz 파일을 [[path1], [path2], ...] 형식으로 수집
    """
    image_files = sorted(glob.glob(os.path.join(input_folder, "*_0000.nii.gz")))
    if len(image_files) == 0:
        raise FileNotFoundError(f"No *_0000.nii.gz files found in {input_folder}")
    return [[f] for f in image_files]

import os

def collect_case_paths_from_folder(input_folder: str) -> List[List[str]]:
    image_files = sorted(glob.glob(os.path.join(input_folder, "*_0000.nii.gz")))
    if len(image_files) == 0:
        raise FileNotFoundError(f"No *_0000.nii.gz files found in {input_folder}")
    return [[f] for f in image_files]

class JointPredictor(nnUNetPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.networks = []  # 현재는 unused이긴 하지만 유지 가능

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                            use_folds: Union[Tuple[Union[int, str]], None],
                                            checkpoint_name: str = 'checkpoint_final.pth',
                                            primary_network_index: int = 1,
                                            prediction_base: Optional[str] = None,
                                            use_ensemble: bool = False):

        if use_folds is None:
            use_folds = self.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        checkpoint = torch.load(join(model_training_output_dir, f'fold_{use_folds[0]}', checkpoint_name),
                                map_location=torch.device('cpu'), weights_only=False)

        assert 'network_weights' in checkpoint and isinstance(checkpoint['network_weights'], list), \
            "Expected joint training checkpoint with 'network_weights' as list"

        trainer_name = checkpoint['trainer_name']
        configuration_name = checkpoint['init_args']['configuration']
        inference_allowed_mirroring_axes = checkpoint.get('inference_allowed_mirroring_axes', None)

        configuration_manager = plans_manager.get_configuration(configuration_name)
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        num_output_channels = plans_manager.get_label_manager(dataset_json).num_segmentation_heads

        trainer_class = recursive_find_python_class(
            join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
            trainer_name, 'nnunetv2.training.nnUNetTrainer'
        )
        assert trainer_class is not None, f"Trainer class {trainer_name} not found"

        # ✅ Only load SWI network
        weights = checkpoint['network_weights'][primary_network_index]
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=False
        )
        network.load_state_dict(weights)
        network.to(self.device)  # ✅ 반드시!

        if ('nnUNet_compile' in os.environ and os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) and \
                not isinstance(network, OptimizedModule):
            print('Using torch.compile')
            network = torch.compile(network)

        self.network = network
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        self.list_of_parameters = None
        self.primary_network_index = primary_network_index
        print(f"[JointPredictor] Loaded network index {self.primary_network_index} (e.g., SWI)")
        # ✅ 결과 저장 경로 자동 지정
        if prediction_base is None:
            results_root = os.environ.get('nnUNet_results', None)
            if results_root is None:
                raise EnvironmentError("nnUNet_results environment variable is not set. Please set it or provide prediction_base.")
            pred_root = results_root.replace('nnUNet_results_scr', 'nnUNet_pred_scr')
            relative_model_path = os.path.relpath(model_training_output_dir, results_root)
            fold_name = f'fold_{use_folds[0]}' if isinstance(use_folds, (list, tuple)) else f'fold_{use_folds}'
            self.prediction_output_folder = join(pred_root, relative_model_path, fold_name)
        else:
            # ✅ prediction_base가 명시된 경우: 환경변수 접근 없이 단순 경로 조합
            fold_name = f'fold_{use_folds[0]}' if isinstance(use_folds, (list, tuple)) else f'fold_{use_folds}'
            self.prediction_output_folder = join(prediction_base, fold_name)

        maybe_mkdir_p(self.prediction_output_folder)
        print(f"[✔] Inference results will be saved to:\n    {self.prediction_output_folder}")

    # def manual_initialization(self,
    #                         network: Union[nn.Module, List[nn.Module]],
    #                         plans_manager: PlansManager,
    #                         configuration_manager: ConfigurationManager,
    #                         parameters: Optional[List[dict]],
    #                         dataset_json: dict,
    #                         trainer_name: str,
    #                         inference_allowed_mirroring_axes: Optional[Tuple[int, ...]],
    #                         primary_network_index: int = 1):
    #     """
    #     Modified for JointTrainer inference:
    #     Allows selection of a specific sub-network from joint training
    #     """
    #     self.plans_manager = plans_manager
    #     self.configuration_manager = configuration_manager
    #     self.dataset_json = dataset_json
    #     self.trainer_name = trainer_name
    #     self.allowed_mirroring_axes = inference_allowed_mirroring_axes
    #     self.label_manager = plans_manager.get_label_manager(dataset_json)
    #     self.primary_network_index = primary_network_index

    #     # 🌟 [1] primary network만 선택
    #     if isinstance(network, (list, tuple)):
    #         self.network = network[primary_network_index]
    #     else:
    #         self.network = network  # fallback (non-joint)

    #     if parameters is not None:
    #         self.list_of_parameters = [parameters[primary_network_index]]
    #         self.network.load_state_dict(parameters[primary_network_index])
    #     else:
    #         self.list_of_parameters = None  # optional

    #     # 🌟 [2] torch.compile 조건 체크
    #     allow_compile = ('nnUNet_compile' in os.environ and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'))
    #     allow_compile = allow_compile and not isinstance(self.network, OptimizedModule)
    #     if isinstance(self.network, DistributedDataParallel):
    #         allow_compile = allow_compile and isinstance(self.network.module, OptimizedModule)
    #     if allow_compile:
    #         print('Using torch.compile')
    #         self.network = torch.compile(self.network)


    @staticmethod
    def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
        print('use_folds is None, attempting to auto detect available folds')
        fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
        fold_folders = [i for i in fold_folders if i != 'fold_all']
        fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
        use_folds = [int(i.split('_')[-1]) for i in fold_folders]
        print(f'found the following folds: {use_folds}')
        return use_folds

    def _manage_input_and_output_lists(self,
                                        list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                        output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
                                        folder_with_segs_from_prev_stage: str = None,
                                        overwrite: bool = True,
                                        part_id: int = 0,
                                        num_parts: int = 1,
                                        save_probabilities: bool = False):
        """
        Handles splitting the prediction input list and output file paths for the current process.
        This function ensures that each DDP process only works on its assigned subset of cases.
        """

        # Case 1: Input is a folder (e.g., imagesTs), parse case lists from dataset.json
        if isinstance(list_of_lists_or_source_folder, str):
            list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(
                list_of_lists_or_source_folder,
                self.dataset_json['file_ending']
            )

        print(f"[JointPredictor] Total {len(list_of_lists_or_source_folder)} cases found.")

        # Distribute cases across processes
        list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
        caseids = [os.path.basename(i[0])[:-(len(self.dataset_json['file_ending']) + 5)]
                for i in list_of_lists_or_source_folder]

        print(f"[JointPredictor] I am process {part_id} out of {num_parts}.")
        print(f"[JointPredictor] Assigned {len(caseids)} cases for prediction.")

        # Determine output file names (without file ending)
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_filename_truncated = [join(output_folder_or_list_of_truncated_output_files, i) for i in caseids]
        else:
            output_filename_truncated = output_folder_or_list_of_truncated_output_files[part_id::num_parts]

        # Handle previous stage segmentation input if cascaded
        seg_from_prev_stage_files = [
            join(folder_with_segs_from_prev_stage, i + self.dataset_json['file_ending'])
            if folder_with_segs_from_prev_stage is not None else None
            for i in caseids
        ]

        # Skip already predicted outputs if overwrite is False
        if not overwrite and output_filename_truncated is not None:
            predicted_flags = [isfile(i + self.dataset_json['file_ending']) for i in output_filename_truncated]

            if save_probabilities:
                prob_flags = [isfile(i + '.npz') for i in output_filename_truncated]
                predicted_flags = [a and b for a, b in zip(predicted_flags, prob_flags)]

            not_done_indices = [i for i, done in enumerate(predicted_flags) if not done]

            # Filter only cases not yet predicted
            output_filename_truncated = [output_filename_truncated[i] for i in not_done_indices]
            list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_done_indices]
            seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_done_indices]

            print(f"[JointPredictor] Skipping already predicted cases. {len(not_done_indices)} remaining.")

        return list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files


    def predict_from_files(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = default_num_processes,
                           num_processes_segmentation_export: int = default_num_processes,
                           folder_with_segs_from_prev_stage: str = None,
                           num_parts: int = 1,
                           part_id: int = 0):
        """
        JointPredictor version of nnUNet's default prediction function.
        Allows prediction using only the primary network from a joint-trained model.
        """
        assert part_id <= num_parts, (
            "Part ID must be smaller than num_parts. Remember that we start counting with 0. "
            "So if there are 3 parts then valid part IDs are 0, 1, 2")

        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(my_init_kwargs)
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)

        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, (
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous '
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where '
                f'they are located via folder_with_segs_from_prev_stage')

        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(
                list_of_lists_or_source_folder,
                output_folder_or_list_of_truncated_output_files,
                folder_with_segs_from_prev_stage,
                overwrite,
                part_id,
                num_parts,
                save_probabilities
            )
            
        # 🔥 핵심: cascaded network가 아니라면 seg_from_prev_stage_files 무시
        if self.configuration_manager.previous_stage_name is None:
            seg_from_prev_stage_files = [None] * len(list_of_lists_or_source_folder)


        if len(list_of_lists_or_source_folder) == 0:
            print("[JointPredictor] No cases to predict.")
            return

        print(f"[JointPredictor] Starting prediction for {len(list_of_lists_or_source_folder)} cases...")

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(
            list_of_lists_or_source_folder,
            seg_from_prev_stage_files,
            output_filename_truncated,
            num_processes_preprocessing
        )

        return self.predict_from_data_iterator(data_iterator, save_probabilities, num_processes_segmentation_export)

    def _internal_get_data_iterator_from_lists_of_filenames(self,
                                                        input_list_of_lists: List[List[str]],
                                                        seg_from_prev_stage_files: Union[List[str], None],
                                                        output_filenames_truncated: Union[List[str], None],
                                                        num_processes: int):
        return preprocessing_iterator_fromfiles(
            input_list_of_lists,
            seg_from_prev_stage_files,
            output_filenames_truncated,
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_processes,
            self.device.type == 'cuda',
            self.verbose_preprocessing
        )

    def get_data_iterator_from_raw_npy_data(self,
                                            image_or_list_of_images,
                                            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
                                            properties_or_list_of_properties,
                                            truncated_ofname,
                                            num_processes: int = 3):
        list_of_images = [image_or_list_of_images] if not isinstance(image_or_list_of_images, list) else image_or_list_of_images

        if isinstance(segs_from_prev_stage_or_list_of_segs_from_prev_stage, np.ndarray):
            segs_from_prev_stage_or_list_of_segs_from_prev_stage = [segs_from_prev_stage_or_list_of_segs_from_prev_stage]
        if isinstance(truncated_ofname, str):
            truncated_ofname = [truncated_ofname]
        if isinstance(properties_or_list_of_properties, dict):
            properties_or_list_of_properties = [properties_or_list_of_properties]

        num_processes = min(num_processes, len(list_of_images))

        return preprocessing_iterator_fromnpy(
            list_of_images,
            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
            properties_or_list_of_properties,
            truncated_ofname,
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_processes,
            self.device.type == 'cuda',
            self.verbose_preprocessing
        )

    def predict_from_list_of_npy_arrays(self,
                                        image_or_list_of_images,
                                        segs_from_prev_stage_or_list_of_segs_from_prev_stage,
                                        properties_or_list_of_properties,
                                        truncated_ofname,
                                        num_processes: int = 3,
                                        save_probabilities: bool = False,
                                        num_processes_segmentation_export: int = default_num_processes):
        iterator = self.get_data_iterator_from_raw_npy_data(
            image_or_list_of_images,
            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
            properties_or_list_of_properties,
            truncated_ofname,
            num_processes
        )
        return self.predict_from_data_iterator(iterator, save_probabilities, num_processes_segmentation_export)

    def predict_from_data_iterator(self,
                                    data_iterator,
                                    save_probabilities: bool = False,
                                    num_processes_segmentation_export: int = default_num_processes):
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                if ofile is not None:
                    print(f"\nPredicting {os.path.basename(ofile)}:")
                else:
                    print(f"\nPredicting image of shape {data.shape}:")

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                properties = preprocessed['data_properties']

                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                prediction = self.predict_logits_from_preprocessed_data(data).cpu()

                if ofile is not None:
                    print('sending off prediction to background worker for resampling and export')
                    r.append(export_pool.starmap_async(
                        export_prediction_from_logits,
                        ((prediction, properties, self.configuration_manager, self.plans_manager,
                        self.dataset_json, ofile, save_probabilities),)
                    ))
                else:
                    print('sending off prediction to background worker for resampling')
                    r.append(export_pool.starmap_async(
                        convert_predicted_logits_to_segmentation_with_correct_shape,
                        ((prediction, self.plans_manager,
                        self.configuration_manager, self.label_manager,
                        properties, save_probabilities),)
                    ))

                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f'Done with image of shape {data.shape}')

            ret = [i.get()[0] for i in r]

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        compute_gaussian.cache_clear()
        empty_cache(self.device)
        return ret

    def predict_single_npy_array(self, input_image: np.ndarray, image_properties: dict,
                                 segmentation_previous_stage: np.ndarray = None,
                                 output_file_truncated: str = None,
                                 save_or_return_probabilities: bool = False):
        """
        Predicts a single image np.ndarray using joint-trained model (SWI by default).
        """
        ppa = PreprocessAdapterFromNpy([input_image], [segmentation_previous_stage], [image_properties],
                                       [output_file_truncated],
                                       self.plans_manager, self.dataset_json, self.configuration_manager,
                                       num_threads_in_multithreaded=1, verbose=self.verbose)

        if self.verbose:
            print('Preprocessing...')
        dct = next(ppa)

        if self.verbose:
            print('Predicting...')
        predicted_logits = self.predict_logits_from_preprocessed_data(dct['data']).cpu()

        if self.verbose:
            print('Resampling to original shape...')
        if output_file_truncated is not None:
            export_prediction_from_logits(predicted_logits, dct['data_properties'], self.configuration_manager,
                                          self.plans_manager, self.dataset_json, output_file_truncated,
                                          save_or_return_probabilities)
        else:
            ret = convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits, self.plans_manager,
                                                                              self.configuration_manager,
                                                                              self.label_manager,
                                                                              dct['data_properties'],
                                                                              return_probabilities=
                                                                              save_or_return_probabilities)
            return ret if not save_or_return_probabilities else ret[0], ret[1]

    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        n_threads = torch.get_num_threads()
        torch.set_num_threads(min(default_num_processes, n_threads))

        prediction = None
            # ✅ 단일 네트워크 예측
        if self.list_of_parameters is None:
            self.network.eval()
            prediction = self.predict_sliding_window_return_logits(data).to('cpu')
            if self.verbose:
                print('Prediction done (single network)')
            torch.set_num_threads(n_threads)
            return prediction
    
        for params in self.list_of_parameters:
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)

            current_prediction = self.predict_sliding_window_return_logits(data).to('cpu')
            prediction = current_prediction if prediction is None else prediction + current_prediction

        if len(self.list_of_parameters) > 1:
            prediction /= len(self.list_of_parameters)

        if self.verbose:
            print('Prediction done')

        torch.set_num_threads(n_threads)
        return prediction

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        slicers = []
        patch_size = self.configuration_manager.patch_size
        tile_step = self.tile_step_size

        if len(patch_size) < len(image_size):
            assert len(patch_size) == len(image_size) - 1, (
                "tile_size must be one shorter than image_size if they're different lengths"
            )
            steps = compute_steps_for_sliding_window(image_size[1:], patch_size, tile_step)
            if self.verbose:
                print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size: {image_size}, tile_size: {patch_size}, steps: {steps}')

            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            (slice(None), d, slice(sx, sx + patch_size[0]), slice(sy, sy + patch_size[1]))
                        )
        else:
            steps = compute_steps_for_sliding_window(image_size, patch_size, tile_step)
            if self.verbose:
                print(f'n_steps {np.prod([len(i) for i in steps])}, image size: {image_size}, tile_size: {patch_size}, steps: {steps}')

            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(tuple([slice(None)] + [slice(s, s + p) for s, p in zip((sx, sy, sz), patch_size)]))

        return slicers

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self.network(x)

        if mirror_axes is not None:
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'
            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
            prediction /= (len(axes_combinations) + 1)
        return prediction

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(self,
                                                    data: torch.Tensor,
                                                    slicers,
                                                    do_on_device: bool = True):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        def producer(d, slh, q):
            for s in slh:
                q.put((torch.clone(d[s][None], memory_format=torch.contiguous_format).to(self.device), s))
            q.put('end')

        try:
            empty_cache(self.device)
            data = data.to(results_device)
            queue = Queue(maxsize=2)
            t = Thread(target=producer, args=(data, slicers, queue))
            t.start()

            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                        dtype=torch.half, device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10, device=results_device)
            else:
                gaussian = 1

            with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                while True:
                    item = queue.get()
                    if item == 'end':
                        queue.task_done()
                        break
                    workon, sl = item
                    prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)

                    if self.use_gaussian:
                        prediction *= gaussian
                    predicted_logits[sl] += prediction
                    n_predictions[sl[1:]] += gaussian
                    queue.task_done()
                    pbar.update()
            queue.join()
            torch.div(predicted_logits, n_predictions, out=predicted_logits)

            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting...')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) -> torch.Tensor:
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()
        empty_cache(self.device)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            assert input_image.ndim == 4
            data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                    'constant', {'value': 0}, True, None)
            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

            try:
                predicted_logits = self._internal_predict_sliding_window_return_logits(
                    data, slicers, self.perform_everything_on_device)
            except RuntimeError:
                print('OOM on GPU, falling back to CPU')
                empty_cache(self.device)
                predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)

            empty_cache(self.device)
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits

    def predict_from_files_sequential(self,
                                    list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                    output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                                    save_probabilities: bool = False,
                                    overwrite: bool = True,
                                    folder_with_segs_from_prev_stage: str = None):
        """
        Just like predict_from_files but doesn't use any multiprocessing. Slow, but sometimes necessary.
        Compatible with JointPredictor (only primary network used).
        """

        # determine output folder
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
            if len(output_folder) == 0:
                output_folder = os.path.curdir
        else:
            output_folder = None

        # log arguments and save metadata
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files_sequential).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(my_init_kwargs)
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)

        # cascaded model check
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, (
                f'Cascaded network detected. Previous stage ({self.configuration_manager.previous_stage_name}) '
                f'segmentations are required via folder_with_segs_from_prev_stage.')

        # get filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage,
                                                overwrite, part_id=0, num_parts=1,
                                                save_probabilities=save_probabilities)

        if len(list_of_lists_or_source_folder) == 0:
            return

        label_manager = self.plans_manager.get_label_manager(self.dataset_json)
        preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose_preprocessing)

        if output_filename_truncated is None:
            output_filename_truncated = [None] * len(list_of_lists_or_source_folder)
        if seg_from_prev_stage_files is None:
            seg_from_prev_stage_files = [None] * len(list_of_lists_or_source_folder)

        ret = []
        for li, of, sps in zip(list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files):
            data, seg, data_properties = preprocessor.run_case(
                li,
                sps,
                self.plans_manager,
                self.configuration_manager,
                self.dataset_json
            )

            print(f'[JointPredictor] perform_everything_on_device: {self.perform_everything_on_device}')
            prediction = self.predict_logits_from_preprocessed_data(torch.from_numpy(data)).cpu()

            if of is not None:
                export_prediction_from_logits(prediction, data_properties, self.configuration_manager,
                                            self.plans_manager, self.dataset_json, of, save_probabilities)
            else:
                ret.append(convert_predicted_logits_to_segmentation_with_correct_shape(prediction, self.plans_manager,
                                                                                    self.configuration_manager,
                                                                                    label_manager,
                                                                                    data_properties,
                                                                                    save_probabilities))
        # cleanup
        compute_gaussian.cache_clear()
        empty_cache(self.device)

        return ret

    def predict_entry_point_modelfolder():
        import argparse
        parser = argparse.ArgumentParser(description='Run inference with JointPredictor or nnUNetPredictor')
        parser.add_argument('-i', type=str, required=True, help='Input folder (e.g., imagesTs)')
        parser.add_argument('-o', type=str, required=True, help='Output folder')
        parser.add_argument('-m', type=str, required=True, help='Trained model folder path (must contain fold_X)')
        parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4), help='Folds to use')
        parser.add_argument('-step_size', type=float, default=0.5)
        parser.add_argument('--disable_tta', action='store_true')
        parser.add_argument('--verbose', action='store_true')
        parser.add_argument('--save_probabilities', action='store_true')
        parser.add_argument('--continue_prediction', '--c', action='store_true')
        parser.add_argument('-chk', type=str, default='checkpoint_final.pth')
        parser.add_argument('-npp', type=int, default=3)
        parser.add_argument('-nps', type=int, default=3)
        parser.add_argument('-prev_stage_predictions', type=str, default=None)
        parser.add_argument('-device', type=str, default='cuda')
        parser.add_argument('--disable_progress_bar', action='store_true')
        parser.add_argument('--trainer_class_name', type=str, default='nnUNetTrainer')
        parser.add_argument('--ensemble', action='store_true')
        parser.add_argument('--modality_idx', type=int, default=1)

        print("\n######## nnU-Net Inference ########\n")
        args = parser.parse_args()
        args.f = [i if i == 'all' else int(i) for i in args.f]

        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.utilities.file_path_utilities import maybe_mkdir_p
        from os.path import isdir
        import torch

        if not isdir(args.o):
            maybe_mkdir_p(args.o)

        if args.device not in ['cpu', 'cuda', 'mps']:
            raise ValueError(f"Unsupported device: {args.device}")
        device = torch.device(args.device)

        if args.device == 'cpu':
            import multiprocessing
            torch.set_num_threads(multiprocessing.cpu_count())
        elif args.device == 'cuda':
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)

        if args.trainer_class_name in ['JointTrainer', 'JointTrainer2']:
            from nnunetv2.inference.predictor_classes.joint_predictor import JointPredictor
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
                use_folds=args.f,
                checkpoint_name=args.chk,
                primary_network_index=args.modality_idx
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
            predictor.initialize_from_trained_model_folder(
                model_training_output_dir=args.m,
                use_folds=args.f,
                checkpoint_name=args.chk
            )

        predictor.predict_from_files(
            list_of_lists_or_source_folder=args.i,
            output_folder_or_list_of_truncated_output_files=args.o,
            save_probabilities=args.save_probabilities,
            overwrite=not args.continue_prediction,
            num_processes_preprocessing=args.npp,
            num_processes_segmentation_export=args.nps,
            folder_with_segs_from_prev_stage=args.prev_stage_predictions,
            num_parts=1,
            part_id=0
        )
    def predict_entry_point():
        import argparse
        parser = argparse.ArgumentParser(description='Run inference with nnU-Net or JointPredictor.')

        # Required
        parser.add_argument('-i', type=str, required=True, help='Input folder')
        parser.add_argument('-o', type=str, required=True, help='Output folder')
        parser.add_argument('-d', type=str, required=True, help='Dataset name or ID')
        parser.add_argument('-c', type=str, required=True, help='Configuration name')

        # Optional
        parser.add_argument('-p', type=str, default='nnUNetPlans', help='Plans identifier')
        parser.add_argument('-tr', type=str, default='nnUNetTrainer', help='Trainer class name')
        parser.add_argument('-f', nargs='+', type=str, default=(0, 1, 2, 3, 4), help='Folds to use')
        parser.add_argument('-chk', type=str, default='checkpoint_final.pth', help='Checkpoint name')
        parser.add_argument('--modality_idx', type=int, default=0, help='Primary network index for JointPredictor')
        parser.add_argument('--ensemble', action='store_true', help='Enable JointPredictor ensemble mode')

        parser.add_argument('-step_size', type=float, default=0.5, help='Sliding window step size')
        parser.add_argument('--disable_tta', action='store_true', default=False, help='Disable test-time augmentation')
        parser.add_argument('--save_probabilities', action='store_true', help='Save softmax probabilities')
        parser.add_argument('--continue_prediction', action='store_true', help='Skip already predicted cases')
        parser.add_argument('-npp', type=int, default=3, help='Num preprocessing workers')
        parser.add_argument('-nps', type=int, default=3, help='Num export workers')
        parser.add_argument('-prev_stage_predictions', type=str, default=None, help='Path to previous stage segs')
        parser.add_argument('-num_parts', type=int, default=1, help='Total parts in distributed inference')
        parser.add_argument('-part_id', type=int, default=0, help='ID of this inference part')
        parser.add_argument('-device', type=str, default='cuda', help='Device for inference')
        parser.add_argument('--disable_progress_bar', action='store_true', default=False, help='Disable tqdm')
        parser.add_argument('--verbose', action='store_true', help='Verbose output')

        args = parser.parse_args()
        args.f = [i if i == 'all' else int(i) for i in args.f]

        from nnunetv2.paths import get_output_folder, maybe_mkdir_p
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        model_folder = get_output_folder(args.d, args.tr, args.p, args.c)
        if not os.path.isdir(args.o):
            maybe_mkdir_p(args.o)

        import torch
        if args.device == 'cpu':
            import multiprocessing
            torch.set_num_threads(multiprocessing.cpu_count())
            device = torch.device('cpu')
        elif args.device == 'cuda':
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            device = torch.device('cuda')
        else:
            device = torch.device('mps')

        if args.tr in ['JointTrainer', 'JointTrainer2']:
            from nnunetv2.inference.predictor_classes.joint_predictor import JointPredictor
            predictor = JointPredictor(
                tile_step_size=args.step_size,
                use_gaussian=True,
                use_mirroring=not args.disable_tta,
                perform_everything_on_device=True,
                device=device,
                verbose=args.verbose,
                verbose_preprocessing=args.verbose,
                allow_tqdm=not args.disable_progress_bar
            )
            predictor.initialize_from_trained_model_folder(
                model_training_output_dir=model_folder,
                primary_network_index=args.modality_idx,
                use_folds=args.f if not args.ensemble else None,
                checkpoint_name=args.chk
            )
        else:
            predictor = nnUNetPredictor(
                tile_step_size=args.step_size,
                use_gaussian=True,
                use_mirroring=not args.disable_tta,
                perform_everything_on_device=True,
                device=device,
                verbose=args.verbose,
                verbose_preprocessing=args.verbose,
                allow_tqdm=not args.disable_progress_bar
            )
            predictor.initialize_from_trained_model_folder(
                model_training_output_dir=model_folder,
                use_folds=args.f,
                checkpoint_name=args.chk
            )

        predictor.predict_from_files(
            args.i,
            args.o,
            save_probabilities=args.save_probabilities,
            overwrite=not args.continue_prediction,
            num_processes_preprocessing=args.npp,
            num_processes_segmentation_export=args.nps,
            folder_with_segs_from_prev_stage=args.prev_stage_predictions,
            num_parts=args.num_parts,
            part_id=args.part_id
        )
    if __name__ == '__main__':
        predict_entry_point()
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


class JointPredictor(nnUNetPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.networks = []
        self.ensemble = False
        self.current_network_id = 0

    def enable_ensemble(self):
        self.ensemble = True

    def disable_ensemble(self):
        self.ensemble = False

    def set_current_network(self, idx: int):
        assert 0 <= idx < len(self.networks), f"Invalid network index: {idx}"
        self.current_network_id = idx

    def manual_initialization(self, networks: list, plans_manager, configuration_manager, parameters: list,
                              dataset_json: dict, trainer_name: str,
                              inference_allowed_mirroring_axes: Optional[Tuple[int, ...]]):
        self.networks = networks
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)

    def initialize_from_trained_model_folder(
    self,
    folder: str,
    modality_idx: Optional[int] = 0,
    use_ensemble: bool = False,
    use_folds: Optional[List[int]] = None,
    checkpoint_name: str = 'checkpoint_final.pth'):
        self.networks = []
        self.ensemble = use_ensemble
        self.current_network_id = modality_idx

        if use_folds is None:
            use_folds = [0]

        plans_path = os.path.join(folder, 'plans.json')
        plans_manager = PlansManager(plans_path)
        configuration = os.path.basename(folder).split('__')[-1]
        configuration_manager = plans_manager.get_configuration(configuration)
        dataset_json_path = os.path.join(folder, 'dataset.json')
        dataset_json = load_json(dataset_json_path)

        num_input_channels = configuration_manager.num_input_channels
        num_output_channels = len(dataset_json['labels']) - 1

        for f in use_folds:
            ckpt_path = os.path.join(folder, f"fold_{f}", checkpoint_name)
            assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"
            checkpoint = torch.load(ckpt_path, map_location='cpu')

            network = get_network_from_plans(
                configuration_manager.network_arch_class_name,
                configuration_manager.network_arch_init_kwargs,
                configuration_manager.network_arch_init_kwargs_req_import,
                num_input_channels,
                num_output_channels,
                deep_supervision=False
            )
            network.load_state_dict(checkpoint['network_weights'])
            network.eval()
            network.inference_apply_nonlin = softmax_helper
            if torch.cuda.is_available():
                network = network.cuda()
            self.networks.append(network)

        # 첫 번째 network 기준으로 initialization
        self.network = self.networks[self.current_network_id]
        self.manual_initialization(
            self.network,
            plans_manager,
            configuration_manager,
            [0.5],  # dummy
            dataset_json,
            "nnUNetTrainer",
            self.inference_allowed_mirroring_axes
        )

    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        torch.set_num_threads(min(torch.get_num_threads(), 8))

        if self.ensemble:
            logits_list = []
            for net in self.networks:
                net.eval()
                net = net.to(self.device)
                logits = self.predict_sliding_window_return_logits(data, net)
                logits_list.append(logits)
            mean_logits = torch.stack(logits_list, dim=0).mean(0)
            return mean_logits
        else:
            net = self.networks[self.current_network_id]
            net.eval()
            net = net.to(self.device)
            return self.predict_sliding_window_return_logits(data, net)

    def predict_sliding_window_return_logits(self, input_image: torch.Tensor, network: torch.nn.Module) -> torch.Tensor:
        self.network = network  # for internal calls using self.network
        return super().predict_sliding_window_return_logits(input_image, network)
    
    def set_current_network(self, idx: int):
        assert 0 <= idx < len(self.networks), "Invalid network index"
        self.current_network_id = idx

    # def enable_ensemble(self):
    #     raise NotImplementedError("Ensemble inference not yet implemented. Use set_current_network(idx) for now.")

    def predict_from_files(self, input_folder: str, output_folder: str,
                       save_probabilities: bool = False,
                       overwrite: bool = True,
                       num_processes_preprocessing: int = 2,
                       num_processes_segmentation_export: int = 2,
                       folder_with_segs_from_prev_stage: Optional[str] = None,
                       num_parts: int = 1,
                       part_id: int = 0): maybe_mkdir_p(output_folder)

    preprocessing = preprocessing_iterator_fromfiles(
        input_folder,
        self.plans_manager,
        self.configuration_manager,
        self.dataset_json,
        folder_with_segs_from_prev_stage,
        num_processes_preprocessing,
        self.verbose_preprocessing,
        num_parts,
        part_id
    )

    for preprocessed in tqdm(preprocessing, disable=not self.allow_tqdm):
        case_id = preprocessed['data_identifier']
        data = torch.from_numpy(preprocessed['data'][None]).to(self.device)

        # ✅ ensemble mode
        pred_logits = self.predict_logits_from_preprocessed_data(data)

        export_prediction_from_logits(
            pred_logits[0],  # remove batch dim
            preprocessed,
            self.plans_manager,
            self.configuration_manager,
            output_folder,
            self.label_manager,
            save_probabilities=save_probabilities,
            overwrite=overwrite,
            segmentation_postprocessing_fn=self.segmentation_export_kwargs.get('segmentation_postprocessing_fn', None),
            save_probabilities_kwargs=self.segmentation_export_kwargs.get('save_probabilities_kwargs', None),
            verbose=self.verbose
        )

        if self.verbose:
            print(f"✅ Inference done for {case_id}")

 # ✅ CLI 사용 시 modality index 또는 ensemble 지정 가능하도록 argparse 예시 추가
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', type=str, required=True, help='Path to trained model folder')
    parser.add_argument('--modality_idx', type=int, default=0, help='Index of the modality to use')
    parser.add_argument('--ensemble', action='store_true', help='Enable ensemble mode')
    parser.add_argument('--trainer_class_name', default='JointTrainer', help='Custom trainer class name (default: JointTrainer)')
    args = parser.parse_args()

    predictor = JointPredictor(trainer_name=args.trainer_class_name)
    predictor.initialize_from_trained_model_folder(
        args.model_folder,
        modality_idx=args.modality_idx,
        use_ensemble=args.ensemble
    )

    print(f"✅ Loaded predictor with {'ensemble' if args.ensemble else f'modality {args.modality_idx}'} using trainer {args.trainer_class_name}")

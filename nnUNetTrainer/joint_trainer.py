import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Tuple, Union, List

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from torch import autocast, nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
from torch import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

#여기서부터 다름
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context, softmax_helper_dim1
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.utils_resunet import build_residual_encoder_unet_from_plans
from nnunetv2.utilities.to_torch import maybe_to_torch, to_cuda
from nnunetv2.training.learning_rate.poly_lr import poly_lr
from nnunetv2.training.loss_functions.deep_supervision import MultipleOutputLoss2

# ✅ 커스텀 DataLoader 사용
from nnunetv2.training.dataloading.JointDataLoader import get_joint_dataloaders_from_env, load_dataset
from nnunetv2.training.data_augmentation.default_data_augmentation import (
    default_2D_augmentation_params,
    default_3D_augmentation_params,
    get_patch_size,
)
from nnunetv2.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunetv2.training.nnUNetTrainer.Joint_nnUNetTrainer import JointTrainer


class JointTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
             device: torch.device = torch.device('cuda'), **kwargs):
        super().__init__(plans, configuration, fold, dataset_json, device)
        #Joint training kwargs
        self.plans_files = kwargs.get("plans_files", os.environ.get("JOINT_PLANS_FILES", "").split(","))
        self.dataset_directories = kwargs.get("dataset_directories", os.environ.get("JOINT_DATASET_DIRS", "").split(","))
        self.max_num_epochs = kwargs.get("max_num_epochs", 1000)
        self.use_meanloss_for_multitask = kwargs.get("use_meanloss_for_multitask", False)
        self.fixed_validation_set = kwargs.get("fixed_validation_set", None)
        self.epoch = 0
        self.output_folders = [
        join(self.output_folder, f"modality_{i}") for i in range(len(self.dataset_directories))
        ]
        self.unpack_data = True
        self.networks = []
        self.optimizers = []
        self.datasets = []
        
        # 이식한 부분
        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.local_rank = 0 if not self.is_ddp else dist.get_rank()

        self.device = device

        # print what device we are using
        if self.is_ddp:  # implicitly it's clear that we use cuda in this case
            print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
                  f"{dist.get_world_size()}."
                  f"Setting device to {self.device}")
            self.device = torch.device(type='cuda', index=self.local_rank)
        else:
            if self.device.type == 'cuda':
                # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
                self.device = torch.device(type='cuda', index=0)
            print(f"Using device: {self.device}")
            
    def initialize(self, training=True, force_load_plans=False):
        if self.was_initialized:
            self.print_to_log_file('Already initialized.')
            return

        # 출력 폴더 생성
        maybe_mkdir_p(self.output_folder)

        # plans 파일(JSON)을 불러옴
        self.planses = [load_json(p) for p in self.plans_files]
        self.plans_managers = [PlansManager(p) for p in self.planses]
        self.configuration_managers = [pm.get_configuration(self.configuration_name) for pm in self.plans_managers]
        # stage 정보 설정 (!!! 추가됨)
        self.stage_plans = self.planses[0]['configurations'][self.configuration_name]
        
        # 기본 설정값들 초기화
        self.net_num_pool_op_kernel_sizes = self.configuration_managers[0].pool_op_kernel_sizes
        self.patch_size = self.configuration_managers[0].patch_size
        self.use_mask_for_norm = self.configuration_managers[0].use_mask_for_norm
        self.num_input_channels = 1
        self.num_classes = len(self.dataset_json['labels'])

        # Data Augmentation 파라미터 설정
        self.setup_DA_params()

        # Deep Supervision Loss 설정
        net_numpool = len(self.net_num_pool_op_kernel_sizes)
        weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
        mask = np.array([True if i < net_numpool - 1 else False for i in range(net_numpool)])
        weights[~mask] = 0
        weights = weights / weights.sum()
        self.loss = MultipleOutputLoss2(self.loss, weights)
        
        # 전처리 데이터 폴더 경로 설정
        self.folder_with_preprocessed_datas = [
            join(dataset_dir, self.configuration_managers[i].data_identifier)
            for i, dataset_dir in enumerate(self.dataset_directories)
        ]

        # ✅ 필요시 데이터 압축 해제
        if training:
            for folder in self.folder_with_preprocessed_datas:
                if self.unpack_data:
                    unpack_dataset(folder)

        # ✅ dataset.json 로딩
        self.datasets = [load_dataset(join(folder)) for folder in self.folder_with_preprocessed_datas]
        self.dataset = self.datasets[0]  # SWI 기준

        # ✅ 데이터 스플릿
        self.do_split()

        # ✅ DataLoader 생성
        self.dl_trs, self.dl_vals = self.get_basic_generators()
        self.tr_gens, self.val_gens = zip(*[
            get_moreDA_augmentation(
                self.dl_trs[i], self.dl_vals[i],
                self.data_aug_params['patch_size_for_spatialtransform'],
                self.data_aug_params,
                deep_supervision_scales=self.deep_supervision_scales,
                pin_memory=self.pin_memory
            )
            for i in range(len(self.dl_trs))
        ])
    
        # ✅ 네트워크, 옵티마이저 초기화
        self.initialize_network()
        self.initialize_optimizer_and_scheduler()

        # ✅ inference 시 SWI 네트워크만 사용
        self.network = self.networks[1]

        # ✅ 완료
        self.was_initialized = True

    def get_basic_generators(self):
        plans = self.planses[0]
        configuration_manager = self.configuration_managers[0]

        patch_size = configuration_manager.patch_size
        final_patch_size = patch_size
        batch_size = plans['configurations'][self.configuration_name]['batch_size']

        # mirror_axes 기본값 처리 (nnUNet V2와 호환)
        # 수정 후
        is_3d = len(configuration_manager.patch_size) == 3
        mirror_axes = getattr(configuration_manager, 'mirror_axes', (0, 1, 2) if is_3d else (0, 1))

        # transforms는 기존 nnUNetTrainer의 static method 그대로 사용
        tr_transforms = nnUNetTrainer.get_training_transforms(
            patch_size=patch_size,
            rotation_for_DA=self.rotation_for_DA,
            deep_supervision_scales=self.deep_supervision_scales,
            mirror_axes=mirror_axes,
            do_dummy_2d_data_aug=self.do_dummy_2d_data_aug,
            use_mask_for_norm=self.use_mask_for_norm,
            is_cascaded=False,
            foreground_labels=None,
            regions=None,
            ignore_label=None
        )

        val_transforms = nnUNetTrainer.get_validation_transforms(
            deep_supervision_scales=self.deep_supervision_scales,
            is_cascaded=False,
            foreground_labels=None,
            regions=None,
            ignore_label=None
        )

        tr_loader = get_joint_dataloaders_from_env(
            patch_size=patch_size,
            final_patch_size=final_patch_size,
            batch_size=batch_size,
            oversample_foreground_percent=0.33,
            transforms=tr_transforms
        )

        val_loader = get_joint_dataloaders_from_env(
            patch_size=patch_size,
            final_patch_size=final_patch_size,
            batch_size=1,
            oversample_foreground_percent=0.0,
            transforms=val_transforms
        )

        return tr_loader, val_loader

    def initialize_network(self):
        self.networks = []
        for plans in self.planses:
            net = build_residual_encoder_unet_from_plans(
                plans=plans,
                num_input_channels=self.num_input_channels,
                num_classes=self.num_classes
            )
            net.inference_apply_nonlin = softmax_helper_dim1
            if torch.cuda.is_available():
                net = net.cuda()
            self.networks.append(net)


    def initialize_optimizer_and_scheduler(self):
        self.optimizers = []
        for net in self.networks:
            self.optimizers.append(torch.optim.SGD(
                net.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay,
                momentum=0.99, nesterov=True
            ))
        self.lr_scheduler = None

    def run_iteration(self, data_generators, do_backprop=True, run_online_evaluation=False):
        data_dicts = [next(gen) for gen in data_generators]
        datas = [to_cuda(maybe_to_torch(d['data'])) for d in data_dicts]
        targets = [to_cuda(maybe_to_torch(d['target'])) for d in data_dicts]

        [opt.zero_grad() for opt in self.optimizers]

        with autocast():
            outputs = [net(data) for net, data in zip(self.networks, datas)]
            losses = [self.loss(output, target) for output, target in zip(outputs, targets)]
            loss_sum = sum(losses)
            loss_mean = loss_sum / len(losses)

        if do_backprop:
            self.amp_grad_scaler.scale(loss_mean if self.use_meanloss_for_multitask else loss_sum).backward()
            for net, opt in zip(self.networks, self.optimizers):
                self.amp_grad_scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
                self.amp_grad_scaler.step(opt)
            self.amp_grad_scaler.update()

        if run_online_evaluation:
            self.run_online_evaluation(outputs[1], targets[1])

        return loss_mean.detach().cpu().numpy() if self.use_meanloss_for_multitask else loss_sum.detach().cpu().numpy()

    def on_epoch_end(self):
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs
        return continue_training

    def run_training(self):
        if not self.was_initialized:
            self.initialize()

        self.maybe_update_lr(self.epoch)
        while self.epoch < self.max_num_epochs:
            self.print_to_log_file(f"Epoch {self.epoch + 1}/{self.max_num_epochs}")
            epoch_losses = []
            for _ in range(self.num_iterations_per_epoch):
                loss = self.run_iteration(self.tr_gens, do_backprop=True, run_online_evaluation=self.run_online_evaluation)
                epoch_losses.append(loss)
            self.print_to_log_file(f"Average loss: {np.mean(epoch_losses):.4f}")
            self.on_epoch_end()
            self.epoch += 1
        self.save_checkpoint(join(self.output_folder, 'model_final.pth'))
        self.print_to_log_file("Training finished")

    def validate(self, *args, **kwargs):
        self.network = self.networks[1]  # SWI only
        return super().validate(*args, **kwargs)

    def predict_preprocessed_data_return_seg_and_softmax(self, *args, **kwargs):
        self.network = self.networks[1]  # SWI only
        return super().predict_preprocessed_data_return_seg_and_softmax(*args, **kwargs)

    def maybe_update_lr(self, epoch=None):
        """
        Updates learning rate using poly_lr for each optimizer in JointTrainer.

        If epoch is None, we assume it's being called before the next epoch starts, so we do self.epoch + 1.
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch

        new_lr = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)

        for opt in self.optimizers:
            opt.param_groups[0]['lr'] = new_lr

        self.print_to_log_file("lr:", np.round(new_lr, decimals=6))
        
    def setup_DA_params(self):
        """
        Simplified setup_DA_params without using self.threeD, fixed to 3D settings.
        """
        # Deep supervision scales 계산
        self.deep_supervision_scales = [[1 for _ in range(len(self.net_num_pool_op_kernel_sizes[0]))]] + \
            [[int(i) for i in s] for s in 1 / np.cumprod(np.vstack(self.net_num_pool_op_kernel_sizes), axis=0)][:-1]

        # 기본 3D augmentation 파라미터 사용
        self.data_aug_params = default_3D_augmentation_params.copy()
        self.data_aug_params["patch_size_for_spatialtransform"] = self.patch_size
        self.data_aug_params["rotation_x"] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        self.data_aug_params["rotation_y"] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        self.data_aug_params["rotation_z"] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params["selected_seg_channels"] = [0]
        self.data_aug_params["num_cached_per_thread"] = 2
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm
        self.data_aug_params["ignore_axes"] = None

        # Augmentation을 고려한 patch size 계산
        self.basic_generator_patch_size = get_patch_size(
            self.patch_size,
            self.data_aug_params["rotation_x"],
            self.data_aug_params["rotation_y"],
            self.data_aug_params["rotation_z"],
            self.data_aug_params["scale_range"]
        )

    def _build_loss(self):
        # 일반적으로 첫 번째 label_manager 사용 (두 modality가 동일하다는 가정)
        label_manager = self.configuration_managers[0].label_manager

        if label_manager.has_regions:
            loss = DC_and_BCE_loss(
                {},
                {
                    'batch_dice': self.configuration_managers[0].batch_dice,
                    'do_bg': True,
                    'smooth': 1e-5,
                    'ddp': self.is_ddp
                },
                use_ignore_label=label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss
            )
        else:
            loss = DC_and_CE_loss(
                {
                    'batch_dice': self.configuration_managers[0].batch_dice,
                    'smooth': 1e-5,
                    'do_bg': False,
                    'ddp': self.is_ddp
                },
                {},
                weight_ce=1,
                weight_dice=1,
                ignore_label=label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss
            )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self.deep_supervision_scales
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        patch_size = self.patch_size
        dim = len(patch_size)

        if dim == 2:
            do_dummy_2d_data_aug = False
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            else:
                rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            mirror_axes = (0, 1)
        elif dim == 3:
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            else:
                rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()

        initial_patch_size = get_patch_size(
            patch_size[-dim:],
            rotation_for_DA,
            rotation_for_DA,
            rotation_for_DA,
            (0.85, 1.25)
        )
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        self.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')
        self.inference_allowed_mirroring_axes = mirror_axes

        # 저장
        self.rotation_for_DA = rotation_for_DA
        self.initial_patch_size = initial_patch_size
        self.mirror_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


    def do_split(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        if self.fold == "all":
            case_identifiers = self.dataset_class.get_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            dataset = self.dataset_class(self.preprocessed_dataset_folder, identifiers=None,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)

            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.identifiers)))
                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)
            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                    % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits contain only %d folds. I am now creating a random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(dataset.identifiers))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                    % (len(tr_keys), len(val_keys)))
            if any([i in val_keys for i in tr_keys]):
                self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the splits.json or ignore if this is intentional.')

        return tr_keys, val_keys


    def get_tr_and_val_datasets(self):
        tr_keys, val_keys = self.do_split()

        dataset_tr = self.dataset_class(self.preprocessed_dataset_folder, tr_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        return dataset_tr, dataset_val


    def get_dataloaders(self):
        patch_size = self.patch_size
        deep_supervision_scales = self.deep_supervision_scales

        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_managers[0].use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.configuration_managers[0].label_manager.foreground_labels,
            regions=self.configuration_managers[0].label_manager.foreground_regions if self.configuration_managers[0].label_manager.has_regions else None,
            ignore_label=self.configuration_managers[0].label_manager.ignore_label)

        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.configuration_managers[0].label_manager.foreground_labels,
            regions=self.configuration_managers[0].label_manager.foreground_regions if self.configuration_managers[0].label_manager.has_regions else None,
            ignore_label=self.configuration_managers[0].label_manager.ignore_label)

        dl_tr = JointDataLoader(dataset_tr, self.batch_size, initial_patch_size, patch_size,
                                self.configuration_managers[0].label_manager,
                                oversample_foreground_percent=self.oversample_foreground_percent,
                                sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                probabilistic_oversampling=self.probabilistic_oversampling)

        dl_val = JointDataLoader(dataset_val, self.batch_size, patch_size, patch_size,
                                self.configuration_managers[0].label_manager,
                                oversample_foreground_percent=self.oversample_foreground_percent,
                                sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                    transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                    num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                    pin_memory=self.device.type == 'cuda',
                                                    wait_time=0.002)
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def get_basic_generators(self):
        # use configuration from the primary modality (e.g., SWI)
        patch_size = self.patch_size
        deep_supervision_scales = self.deep_supervision_scales

        # get DA-related configs
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # perform split and prepare datasets
        tr_keys, val_keys = self.do_split()

        dataset_tr = self.dataset_class(self.preprocessed_dataset_folder, tr_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)

        # training transforms
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_managers[0].use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.configuration_managers[0].label_manager.foreground_labels,
            regions=self.configuration_managers[0].label_manager.foreground_regions if self.configuration_managers[0].label_manager.has_regions else None,
            ignore_label=self.configuration_managers[0].label_manager.ignore_label)

        # validation transforms
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.configuration_managers[0].label_manager.foreground_labels,
            regions=self.configuration_managers[0].label_manager.foreground_regions if self.configuration_managers[0].label_manager.has_regions else None,
            ignore_label=self.configuration_managers[0].label_manager.ignore_label)

        # create joint dataloaders (custom wrapper for multiple modalities)
        dl_tr = JointDataLoader(dataset_tr, self.batch_size, initial_patch_size, patch_size,
                                self.configuration_managers[0].label_manager,
                                oversample_foreground_percent=self.oversample_foreground_percent,
                                sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                probabilistic_oversampling=self.probabilistic_oversampling)

        dl_val = JointDataLoader(dataset_val, self.batch_size, patch_size, patch_size,
                                self.configuration_managers[0].label_manager,
                                oversample_foreground_percent=self.oversample_foreground_percent,
                                sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                probabilistic_oversampling=self.probabilistic_oversampling)

        # wrap with augmenter
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                    transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                    num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                    pin_memory=self.device.type == 'cuda',
                                                    wait_time=0.002)
        # prime the pipelines
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

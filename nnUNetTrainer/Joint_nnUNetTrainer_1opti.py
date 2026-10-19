import inspect
import multiprocessing
import os
import subprocess
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
from torch import GradScaler
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
from torch import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from os.path import join
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.f1_lesion_loss import LesionF1Loss
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
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from datetime import datetime

class JointTrainer2(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), **kwargs):
        self.dataset_directories = kwargs.get("dataset_directories", os.environ.get("JOINT_DATASET_DIRS", "").split(","))
        self.plans_files = kwargs.get("plans_files", os.environ.get("JOINT_PLANS_FILES", "").split(","))
        if any(p == '' for p in self.plans_files): self.plans_files = []
        if any(d == '' for d in self.dataset_directories): self.dataset_directories = []
        assert len(self.plans_files) == len(self.dataset_directories), "plans_files와 dataset_directories의 길이는 같아야 합니다."
        self.planses = [load_json(p) for p in self.plans_files]
        self.plans_managers = [PlansManager(p) for p in self.planses]
        
        super().__init__(plans, configuration, fold, dataset_json, device)
        # ✅ 바로 여기! output_folder 재정의는 super 다음에!
        now_str = datetime.now().strftime('%Y%m%d_%H%M%S')

        modality_names = [d.split("_")[-1] for d in self.dataset_directories]
        modalities_str = "__".join(modality_names)

        self.output_folder_base = join(
            nnUNet_results,
            f"JointDataset__{modalities_str}",
            f"{self.__class__.__name__}__{self.plans_manager.plans_name}__{self.configuration_name}_1opti"
        )
        self.output_folder = join(self.output_folder_base, f'fold_{self.fold}')
        
        self.initial_lr = 1e-4
        # Joint training 전용 변수 (V2 CLI 호환)
        self.configuration_managers = [pm.get_configuration(self.configuration_name) for pm in self.plans_managers]
        self.use_meanloss_for_multitask = kwargs.get("use_meanloss_for_multitask", False)
        self.primary_network_index = 1
        # Multi-modality용 리스트들
        self.networks = []
        self.optimizers = None
        self.datasets = []
        self.dataloaders_train = []
        self.dataloaders_val = []
        self.output_folders = [
            join(self.output_folder, f"modality_{i}") for i in range(len(self.dataset_directories))
        ]

        # self.primary_network_index = 1  # inference 시 사용하는 네트워크
        self.loss = None
        self.current_epoch = 0
        from nnunetv2.training.loss.f1_lesion_loss import LesionF1Loss  # 만들 파일
        self.use_f1_loss = kwargs.get("use_f1_loss", True)  # CLI 인자 등으로 제어 가능
        self.f1_loss_computer = LesionF1Loss(threshold=0.5) if self.use_f1_loss else None

        # 로그
        self.print_to_log_file("JointTrainer initialized with:")
        self.print_to_log_file(f"  - plans_files: {self.plans_files}")
        self.print_to_log_file(f"  - dataset_directories: {self.dataset_directories}")
        self.print_to_log_file(f"  - use_meanloss_for_multitask: {self.use_meanloss_for_multitask}")
        self.print_to_log_file(f"  - use_f1_loss: {self.use_f1_loss}")
        self.print_to_log_file(f"✔ F1 loss integration is {'enabled' if self.use_f1_loss else 'disabled'}")
        self.print_to_log_file(f"✔ use_meanloss_for_multitask is set to: {self.use_meanloss_for_multitask}")
        print(f"*** JointTrainer loaded modalities: {self.dataset_directories}")
        self.print_to_log_file(f"✔ [JointTrainer] output_folder is set to: {self.output_folder}")
        print(f"[INFO] Using modality index {self.primary_network_index} for validation (e.g., SWI)")
        print(f"[INFO] Number of GPUS {torch.cuda.device_count()} for training")
        from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
        maybe_mkdir_p(self.output_folder_base)
        maybe_mkdir_p(self.output_folder)
        self.print_to_log_file(f"✔ [JointTrainer] output_folder is set to: {self.output_folder}")

    def initialize(self):
        if self.was_initialized:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized.")

        self._set_batch_size_and_oversample()
        self.num_input_channels = 1
        self.networks = []
        all_params = []
        self._best_mean_loss = None
        self._best_swi_loss = None
        self.planses = [load_json(p) for p in self.plans_files]
        self.plans_managers = [PlansManager(p) for p in self.planses]
        self.configuration_managers = [pm.get_configuration(self.configuration_name) for pm in self.plans_managers]
        self.print_to_log_file("🔍 Verifying JointTrainer Modality Setup:")
        self.print_to_log_file(f"  > Number of modalities: {len(self.dataset_directories)}")
        for i, (plan_file, data_dir) in enumerate(zip(self.plans_files, self.dataset_directories)):
            plan_name = os.path.basename(plan_file)
            dataset_name = os.path.basename(data_dir)
            self.print_to_log_file(f"- Modality {i}: dataset = {dataset_name}, plan = {plan_name}")

        self.folder_with_preprocessed_datas = [
            join(self.dataset_directories[i], self.configuration_managers[i].data_identifier)
            for i in range(len(self.dataset_directories))
        ]

        for i, plans_file in enumerate(self.plans_files):
            plans = PlansManager(load_json(plans_file))
            config = plans.get_configuration(self.configuration_name)

            net = self.build_network_architecture(
                config.network_arch_class_name,
                config.network_arch_init_kwargs,
                config.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)

            if self._do_i_compile():
                self.print_to_log_file(f"[modality {i}] Using torch.compile...")
                net = torch.compile(net)

            if self.is_ddp:
                net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
                net = DDP(net, device_ids=[self.local_rank])

            self.networks.append(net)
            all_params += list(net.parameters())  # 💡 모든 네트워크의 파라미터를 모음

        # 💡 하나의 공통 optimizer를 생성
        self.optimizer, _ = self.configure_optimizers(all_params)

        # 💡 공통 learning rate scheduler
        self.lr_scheduler = PolyLRScheduler(self.optimizer, self.initial_lr, self.num_epochs)

        # 나머지 초기화
        self.loss = self._build_loss()
        self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        self.was_initialized = True
        self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None

    def _save_debug_information(self):
        if self.local_rank == 0:
            dct = {}
            for k in self.__dir__():
                if not k.startswith("__"):
                    if not callable(getattr(self, k)) or k in ['loss']:
                        dct[k] = str(getattr(self, k))
                    elif k == 'network':
                        if hasattr(self, 'primary_network_index'):
                            dct[k] = str(self.networks[self.primary_network_index].__class__.__name__)
                        else:
                            dct[k] = str(getattr(self, k).__class__.__name__)
            dct['networks'] = [str(n.__class__.__name__) for n in self.networks]

            for dl_name in ['dataloaders_train', 'dataloaders_val']:
                loaders = getattr(self, dl_name, None)
                if loaders is not None:
                    for i, loader in enumerate(loaders):
                        if hasattr(loader, 'generator'):
                            dct[f'{dl_name}[{i}].generator'] = str(loader.generator)
                        if hasattr(loader, 'num_processes'):
                            dct[f'{dl_name}[{i}].num_processes'] = str(loader.num_processes)
                        if hasattr(loader, 'transform'):
                            dct[f'{dl_name}[{i}].transform'] = str(loader.transform)

            hostname = subprocess.getoutput(['hostname'])
            dct['hostname'] = hostname
            dct['device'] = str(self.device)
            dct['torch_version'] = torch.__version__
            dct['cudnn_version'] = torch.backends.cudnn.version() if self.device.type == 'cuda' else 'None'
            if self.device.type == 'cuda':
                dct['gpu_name'] = torch.cuda.get_device_name()

            save_json(dct, join(self.output_folder, "debug.json"))

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                arch_init_kwargs: dict,
                                arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                num_input_channels: int,
                                num_output_channels: int,
                                enable_deep_supervision: bool = True) -> nn.Module:
        """
        Returns a single network instance given all required information.
        In JointTrainer, this function will be called multiple times during initialize() — once per modality.
        """
        return get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
                self.configuration_managers[0].pool_op_kernel_sizes), axis=0))[:-1]
        else:
            scales = None
        return scales

    def _set_batch_size_and_oversample(self):
        config = self.configuration_managers[0]  # 대표 modality 기준
        if not self.is_ddp:
            self.batch_size = config.batch_size
        else:
            world_size = dist.get_world_size()
            my_rank = dist.get_rank()
            global_batch_size = config.batch_size
            assert global_batch_size >= world_size

            batch_size_per_GPU = [global_batch_size // world_size] * world_size
            batch_size_per_GPU = [batch_size_per_GPU[i] + 1
                                if (batch_size_per_GPU[i] * world_size + i) < global_batch_size
                                else batch_size_per_GPU[i]
                                for i in range(len(batch_size_per_GPU))]
            assert sum(batch_size_per_GPU) == global_batch_size

            sample_id_low = 0 if my_rank == 0 else np.sum(batch_size_per_GPU[:my_rank])
            sample_id_high = np.sum(batch_size_per_GPU[:my_rank + 1])

            oversample = [True if not i < round(global_batch_size * (1 - self.oversample_foreground_percent)) else False
                        for i in range(global_batch_size)]

            if sample_id_high / global_batch_size < (1 - self.oversample_foreground_percent):
                oversample_percent = 0.0
            elif sample_id_low / global_batch_size > (1 - self.oversample_foreground_percent):
                oversample_percent = 1.0
            else:
                oversample_percent = sum(oversample[sample_id_low:sample_id_high]) / batch_size_per_GPU[my_rank]

            print("worker", my_rank, "oversample", oversample_percent)
            print("worker", my_rank, "batch_size", batch_size_per_GPU[my_rank])

            self.batch_size = batch_size_per_GPU[my_rank]
            self.oversample_foreground_percent = oversample_percent

    def _build_loss(self):
        config = self.configuration_managers[0]
        label_manager = self.label_manager  # 대표 modality 기준

        if label_manager.has_regions:
            loss = DC_and_BCE_loss(
                {},
                {
                    'batch_dice': config.batch_dice,
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
                    'batch_dice': config.batch_dice,
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
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        config = self.configuration_managers[0]  # 대표 modality 기준
        patch_size = config.patch_size
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
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        if self.local_rank == 0:
            timestamp = time()
            dt_object = datetime.fromtimestamp(timestamp)

            if add_timestamp:
                args = (f"{dt_object}:", *args)

            successful = False
            max_attempts = 5
            ctr = 0
            while not successful and ctr < max_attempts:
                try:
                    with open(self.log_file, 'a+') as f:
                        for a in args:
                            f.write(str(a))
                            f.write(" ")
                        f.write("\n")
                    successful = True
                except IOError:
                    print(f"{datetime.fromtimestamp(timestamp)}: failed to log: ", sys.exc_info())
                    sleep(0.5)
                    ctr += 1
            if also_print_to_console:
                print(*args)
        elif also_print_to_console:
            print(*args)
            
    def print_plans(self):
        if self.local_rank == 0:
            dct = deepcopy(self.plans_managers[0].plans)
            if 'configurations' in dct:
                del dct['configurations']
            self.print_to_log_file(
                f"\nThis is the configuration used by this training (modality 0):\nConfiguration name: {self.configuration_name}\n",
                self.configuration_managers[0], '\\n', add_timestamp=False
            )
            self.print_to_log_file('These are the global plan.json settings:\n', dct, '\n', add_timestamp=False)
            
    def configure_optimizers(self, all_params):
        """
        JointTrainer용 configure_optimizers:
        여러 네트워크의 파라미터를 리스트(all_params)로 받아,
        하나의 공통 optimizer를 반환.
        """
        optimizer = torch.optim.SGD(
            all_params,
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True
        )
        return optimizer, None  # scheduler는 initialize()에서 따로 설정

    def plot_network_architecture(self):
        if self._do_i_compile():
            self.print_to_log_file("Unable to plot network architecture: nnUNet_compile is enabled!")
            return

        if self.local_rank == 0:
            try:
                import hiddenlayer as hl
                primary_net = self.networks[self.primary_network_index]
                patch_size = self.configuration_managers[self.primary_network_index].patch_size

                g = hl.build_graph(primary_net,
                                torch.rand((1, self.num_input_channels, *patch_size), device=self.device),
                                transforms=None)
                g.save(join(self.output_folder, "network_architecture.pdf"))
                del g
            except Exception as e:
                self.print_to_log_file("Unable to plot network architecture:")
                self.print_to_log_file(e)
            finally:
                empty_cache(self.device)


    def do_split(self):
        """
        JointTrainer에서도 대표 modality 기준으로 동일 split 사용.
        모든 modality가 동일한 case_identifier를 공유한다고 가정.
        """
        representative_folder = self.folder_with_preprocessed_datas[self.primary_network_index]
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(representative_folder)

        if self.fold == "all":
            case_identifiers = self.dataset_class.get_identifiers(representative_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            dataset = self.dataset_class(
                representative_folder,
                identifiers=None,
                folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
            )

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
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                    "contain only %d folds. I am now creating a "
                                    "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(dataset.identifiers))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                    % (len(tr_keys), len(val_keys)))

            if any([i in val_keys for i in tr_keys]):
                self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                    'splits.json or ignore if this is intentional.')

        return tr_keys, val_keys

    def get_tr_and_val_datasets(self):
        """
        JointTrainer에서 modality별로 개별 dataset_tr, dataset_val 구성.
        split은 대표 modality 기준으로 한 번만 수행.
        """

        tr_keys, val_keys = self.do_split()

        datasets_tr = []
        datasets_val = []

        for i in range(len(self.folder_with_preprocessed_datas)):
            dataset_tr = self.dataset_class(
                self.folder_with_preprocessed_datas[i], tr_keys,
                folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
            )
            dataset_val = self.dataset_class(
                self.folder_with_preprocessed_datas[i], val_keys,
                folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
            )
            datasets_tr.append(dataset_tr)
            datasets_val.append(dataset_val)

        # 대표 modality(SWI 등)를 기준으로 하나만 할당 (예: inference용)
        self.dataset = datasets_tr[self.primary_network_index]
        self.datasets_train = datasets_tr
        self.datasets_val = datasets_val

        return datasets_tr, datasets_val

    def get_dataloaders(self):
        """
        JointTrainer에서 각 modality에 대해 개별 DataLoader + transforms + augmenters를 구성.
        nnUNet V2의 Dataloader 구조를 그대로 사용함.
        """

        # 결과 리스트 (각 modality별)
        dataloaders_train = []
        dataloaders_val = []

        # split은 대표 modality 기준으로 한 번만 실행
        tr_keys, val_keys = self.do_split()

        for i in range(len(self.folder_with_preprocessed_datas)):
            config = self.configuration_managers[i]
            patch_size = config.patch_size
            deep_supervision_scales = self._get_deep_supervision_scales()

            rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
                self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

            # transforms
            tr_transforms = self.get_training_transforms(
                patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
                use_mask_for_norm=config.use_mask_for_norm,
                is_cascaded=self.is_cascaded,
                foreground_labels=self.label_manager.foreground_labels,
                regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
                ignore_label=self.label_manager.ignore_label)

            val_transforms = self.get_validation_transforms(
                deep_supervision_scales,
                is_cascaded=self.is_cascaded,
                foreground_labels=self.label_manager.foreground_labels,
                regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
                ignore_label=self.label_manager.ignore_label)

            # 각 modality에 맞는 dataset 로딩
            dataset_tr = self.dataset_class(self.folder_with_preprocessed_datas[i], tr_keys,
                                            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            dataset_val = self.dataset_class(self.folder_with_preprocessed_datas[i], val_keys,
                                            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)

            # DataLoader 구성 (V2 방식)
            dl_tr = nnUNetDataLoader(
                dataset_tr, self.batch_size, initial_patch_size, patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, pad_sides=None,
                transforms=tr_transforms,
                probabilistic_oversampling=self.probabilistic_oversampling
            )
            dl_val = nnUNetDataLoader(
                dataset_val, self.batch_size, patch_size, patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, pad_sides=None,
                transforms=val_transforms,
                probabilistic_oversampling=self.probabilistic_oversampling
            )

            # Data Augmenter 구성
            allowed_num_processes = min(get_allowed_n_proc_DA(), 2)
            if allowed_num_processes == 0:
                mt_gen_tr = SingleThreadedAugmenter(dl_tr, None)
                mt_gen_val = SingleThreadedAugmenter(dl_val, None)
            else:
                mt_gen_tr = NonDetMultiThreadedAugmenter(dl_tr, None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2),
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
                mt_gen_val = NonDetMultiThreadedAugmenter(dl_val, None,
                                                        num_processes=max(1, allowed_num_processes // 2),
                                                        num_cached=max(3, allowed_num_processes // 4),
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)

            # pipe prime
            _ = next(mt_gen_tr)
            _ = next(mt_gen_val)

            # dataloaders_train과 dataloaders_val에 추가
            dataloaders_train.append(iter(mt_gen_tr))
            dataloaders_val.append(iter(mt_gen_val))

        # 각 modality에 대해 iterator로 변환
        self.dataloader_train = dataloaders_train
        self.dataloader_val = dataloaders_val

        return dataloaders_train, dataloaders_val

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        synchronize_channels=False,
        ignore_label: int = None,
    ) -> BasicTransform:
        """
        JointTrainer 전용 get_training_transforms 함수.
        nnUNetTrainer.get_training_transforms의 구조를 그대로 유지하면서, modality-wise 설정이 가능하게 구성.
        """
        transforms = []

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        # Spatial transform
        transforms.append(SpatialTransform(
            patch_size_spatial,
            patch_center_dist_from_border=0,
            random_crop=False,
            p_elastic_deform=0,
            p_rotation=0.2,
            rotation=rotation_for_DA,
            p_scaling=0.2,
            scaling=(0.7, 1.4),
            p_synchronize_scaling_across_axes=1,
            bg_style_seg_sampling=False
        ))

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

                # Intensity transforms (noise, blur, brightness, contrast, gamma)
        transforms += [
            RandomTransform(
                GaussianNoiseTransform(
                    noise_variance=(0, 0.1),
                    p_per_channel=1,
                    synchronize_channels=True
                ),
                apply_probability=0.1
            ),
            RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.5, 1.),
                    synchronize_channels=False,
                    synchronize_axes=False,
                    p_per_channel=0.5,
                    benchmark=True
                ),
                apply_probability=0.2
            ),
            RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast((0.75, 1.25)),
                    synchronize_channels=False,
                    p_per_channel=1
                ),
                apply_probability=0.15
            ),
            RandomTransform(
                ContrastTransform(
                    contrast_range=BGContrast((0.75, 1.25)),
                    preserve_range=True,
                    synchronize_channels=False,
                    p_per_channel=1
                ),
                apply_probability=0.15
            ),
            RandomTransform(
                SimulateLowResolutionTransform(
                    scale=(0.5, 1),
                    synchronize_channels=False,
                    synchronize_axes=True,
                    ignore_axes=ignore_axes,
                    allowed_channels=None,
                    p_per_channel=0.5
                ),
                apply_probability=0.25
            ),
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=1,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1
                ),
                apply_probability=0.1
            ),
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=0,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1
                ),
                apply_probability=0.3
            )
        ]
        # Mirror transform
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))

        # Mask-based normalization
        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i, v in enumerate(use_mask_for_norm) if v],
                channel_idx_in_seg=0,
                set_outside_to=0
            ))

        transforms.append(RemoveLabelTansform(-1, 0))

        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms += [
                MoveSegAsOneHotToDataTransform(1, foreground_labels, True),
                RandomTransform(ApplyRandomBinaryOperatorTransform(
                    channel_idx=list(range(-len(foreground_labels), 0)), strel_size=(1, 8), p_per_label=1), 0.4),
                RandomTransform(RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    list(range(-len(foreground_labels), 0)), 0, 0.15, 1), 0.2)
            ]

        if regions is not None:
            transforms.append(ConvertSegmentationToRegionsTransform(
                regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                channel_in_seg=0
            ))

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(deep_supervision_scales))

        return ComposeTransforms(transforms)


    @staticmethod
    def get_validation_transforms(
        deep_supervision_scales: Union[List, Tuple, None],
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        """
        JointTrainer 전용 get_validation_transforms 함수.
        nnUNetTrainer 구조를 그대로 유지하면서, modality-wise로 반복 사용 가능한 형태로 정의.
        """
        transforms = [RemoveLabelTansform(-1, 0)]

        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade validation'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )

        if regions is not None:
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        JointTrainer용 deep supervision 설정 함수.
        각 modality의 네트워크마다 deep_supervision 속성을 설정.
        """
        for i, net in enumerate(self.networks):
            if self.is_ddp:
                mod = net.module
            else:
                mod = net

            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod

            # nnUNet V2의 기본 구조를 따른다는 전제
            if hasattr(mod, 'decoder') and hasattr(mod.decoder, 'deep_supervision'):
                mod.decoder.deep_supervision = enabled
                self.print_to_log_file(f"Set deep_supervision={enabled} for network {i}")
            else:
                self.print_to_log_file(f"[WARNING] Network {i} has no attribute decoder.deep_supervision")

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()

        # DataLoader 구성 (각 modality별로 구성됨)
        self.dataloaders_train, self.dataloaders_val = self.get_dataloaders()

        maybe_mkdir_p(self.output_folder)

        # deep supervision 활성화
        self.set_deep_supervision_enabled(self.enable_deep_supervision)

        self.print_plans()
        empty_cache(self.device)

        # ✅ dataset 압축 해제: 대표 modality 기준
        if self.local_rank == 0:
            self.dataset_class.unpack_dataset(
                self.preprocessed_dataset_folder,  # 기존 nnUNetTrainer에서 사용된 변수
                overwrite_existing=False,
                num_processes=max(1, round(get_allowed_n_proc_DA() // 2)),
                verify=True
            )

        if self.is_ddp:
            dist.barrier()

        # ✅ 대표 modality 기준 plans 및 dataset 메타데이터 저장
        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)
        shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                    join(self.output_folder_base, 'dataset_fingerprint.json'))

        # ✅ 네트워크 구조 시각화 (대표 modality만)
        self.plot_network_architecture()

        # ✅ 디버그 정보 저장
        self._save_debug_information()

    def on_train_end(self):
        # dirty hack because on_epoch_end increments the epoch counter and this is executed afterwards.
        # This will lead to the wrong current epoch to be stored
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.current_epoch += 1

        # now we can delete latest
        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        # shut down dataloaders for each modality
        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if isinstance(self.dataloader_train, list):
                for dl in self.dataloader_train:
                    if isinstance(dl, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                        dl._finish()
            if isinstance(self.dataloader_val, list):
                for dl in self.dataloader_val:
                    if isinstance(dl, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                        dl._finish()
            sys.stdout = old_stdout

        empty_cache(self.device)
        self.print_to_log_file("Training done.")

    def on_train_epoch_start(self):
        for net in self.networks:
            net.train()

        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        
        current_lr = self.optimizer.param_groups[0]['lr']
        self.print_to_log_file(f"Current learning rate: {np.round(current_lr, decimals=5)}")
        self.logger.log('lrs', current_lr, self.current_epoch)


    def train_step(self, batch_list: list[dict]) -> dict:
        """
        batch_list: [batch_modality_0, batch_modality_1, ...]
        각 batch는 {'data': ..., 'target': ...} 형태의 dict
        """
        outputs = []
        losses = []

        for i, batch in enumerate(batch_list):
            data = batch['data'].to(self.device, non_blocking=True)
            target = batch['target']

            if isinstance(target, list):
                target = [t.to(self.device, non_blocking=True) for t in target]
            else:
                target = target.to(self.device, non_blocking=True)


            # forward pass
            with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                output = self.networks[i](data)
                loss = self.loss(output, target)

            outputs.append(output)
            # sigmoid 적용 후 lesion-level F1 loss 추가
            if self.use_f1_loss:
                pred_soft = torch.sigmoid(output[0] if isinstance(output, (list, tuple)) else output)
                target_bin = target[0] if isinstance(target, (list, tuple)) else target
                lesion_f1_loss, f1_score_val, tp, fp, fn = self.f1_loss_computer(pred_soft, target_bin)
                loss = 0.5 * loss + 0.5 * lesion_f1_loss # ✅ 여기서 바로 보완된 loss로 덮어쓰기

                if self.local_rank == 0:
                    self.print_to_log_file(
                        f"[Train] Epoch {self.current_epoch} | Modality {i} | F1: {f1_score_val:.4f} | TP: {tp} | FP: {fp} | FN: {fn} | F1_loss: {lesion_f1_loss.item():.4f}")

            losses.append(loss)  # ✅ 이제 이 줄은 보완된 loss를 넣게 됨

        # ✅ 총 loss 계산
        total_loss = sum(losses) / len(losses) if self.use_meanloss_for_multitask else sum(losses)

        # ✅ optimizer 하나만
        self.optimizer.zero_grad(set_to_none=True)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()

            # ✅ gradient clipping은 여전히 network별로 수행
            self.grad_scaler.unscale_(self.optimizer)
            for net in self.networks:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 12)

            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            for net in self.networks:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
            self.optimizer.step()

        return {
            'loss': total_loss.detach().cpu().numpy(),
            'loss_per_modality': [l.detach().cpu().numpy() for l in losses]
        }


    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            modality_losses_tr = [None for _ in range(dist.get_world_size())]
            
            dist.all_gather_object(losses_tr, outputs['loss'])
            dist.all_gather_object(modality_losses_tr, outputs['loss_per_modality'])
            
            loss_here = np.vstack(losses_tr).mean()
            modality_loss_here = np.vstack(modality_losses_tr).mean(axis=0)
            
        else:
            loss_here = np.mean(outputs['loss'])
            modality_loss_here = np.mean(np.vstack(outputs['loss_per_modality']), axis=0)

        self.logger.log('train_losses', loss_here, self.current_epoch)
        if 'train_losses_per_modality' not in self.logger.my_fantastic_logging:
            self.logger.my_fantastic_logging['train_losses_per_modality'] = []
        self.logger.log('train_losses_per_modality', modality_loss_here.tolist(), self.current_epoch)


    def on_validation_epoch_start(self):
        for net in self.networks:
            net.eval()

    def validation_step(self, val_batch_list: list) -> dict:
        swi_batch = val_batch_list[self.primary_network_index]  # SWI만 사용

        data = swi_batch['data'].to(self.device, non_blocking=True)
        target = swi_batch['target']
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        data_list = [batch['data'].to(self.device, non_blocking=True) for batch in val_batch_list]
        target_list = [batch['target'] for batch in val_batch_list]

        target_list = [
            [t.to(self.device, non_blocking=True) for t in tgt] if isinstance(tgt, list)
            else tgt.to(self.device, non_blocking=True)
            for tgt in target_list
        ]

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            if self.use_meanloss_for_multitask:
                outputs = [net(data) for net, data in zip(self.networks, data_list)]

                if self.enable_deep_supervision:
                    outputs = [out[0] if isinstance(out, (list, tuple)) else out for out in outputs]
                    target_list = [t[0] if isinstance(t, (list, tuple)) else t for t in target_list]

                total_output = sum(outputs) / len(outputs)

                losses = [self.loss([output], [tgt]) for output, tgt in zip(outputs, target_list)]
                loss = sum(losses) / len(losses)
                eval_target = target_list[self.primary_network_index]
            else:
                data = data_list[self.primary_network_index]
                target = target_list[self.primary_network_index]
                total_output = self.networks[self.primary_network_index](data)
                loss = self.loss(total_output, target)
                eval_target = target
                losses = [loss]  # 💡 안전하게 하나짜리 리스트로 만들어줌

        # ✅ deep supervision 후처리
        if self.enable_deep_supervision:
            if isinstance(total_output, (list, tuple)):
                total_output = total_output[0]
            if isinstance(eval_target, (list, tuple)):
                eval_target = eval_target[0]

        axes = [0] + list(range(2, total_output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(total_output) > 0.5).long()
        else:
            output_seg = total_output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros_like(total_output, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (eval_target != self.label_manager.ignore_label).float()
                eval_target[eval_target == self.label_manager.ignore_label] = 0
            else:
                if eval_target.dtype == torch.bool:
                    mask = ~eval_target[:, -1:]
                else:
                    mask = 1 - eval_target[:, -1:]
                eval_target = eval_target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, eval_target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
        # ✅ Lesion-level F1 평가용
        f1_score_val, lesion_f1_loss_val, tp_f1, fp_f1, fn_f1 = None, None, None, None, None
        lesion_f1s = []
        lesion_f1_losses = []

        if self.use_f1_loss:
            for output, tgt in zip(outputs, target_list):
                pred_soft = torch.sigmoid(output)
                lesion_f1_loss, f1_score, tp, fp, fn = self.f1_loss_computer(pred_soft, tgt)
                lesion_f1s.append(f1_score)
                lesion_f1_losses.append(lesion_f1_loss.item())

            avg_f1 = float(np.mean(lesion_f1s))
            avg_f1_loss = float(np.mean(lesion_f1_losses))

            if self.local_rank == 0:
                self.print_to_log_file(f"[Val] Epoch {self.current_epoch} | Avg Lesion F1 (all modalities): {avg_f1:.4f} | F1 Loss: {avg_f1_loss:.4f}")

        return {
            'loss': loss.detach().cpu().numpy(),
            'loss_per_modality': [l.detach().cpu().numpy() for l in losses] if self.use_meanloss_for_multitask else [loss.detach().cpu().numpy()],
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
            'lesion_f1': avg_f1 if self.use_f1_loss else None,
            'lesion_f1_loss': avg_f1_loss if self.use_f1_loss else None
        }

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        if self.use_f1_loss and 'lesion_f1' in outputs_collated:
            lesion_f1_scores = [f for f in outputs_collated['lesion_f1'] if f is not None]
            lesion_f1_losses = [l for l in outputs_collated['lesion_f1_loss'] if l is not None]
            if lesion_f1_scores:
                avg_f1 = np.mean(lesion_f1_scores)
                avg_f1_loss = np.mean(lesion_f1_losses)
                self.logger.log('val_lesion_f1', avg_f1, self.current_epoch)
                self.logger.log('val_lesion_f1_loss', avg_f1_loss, self.current_epoch)
                self.print_to_log_file(f"Val Lesion F1 score: {np.round(avg_f1, 4)}, Loss: {np.round(avg_f1_loss, 4)}")
                
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            modality_losses_val = [None for _ in range(world_size)]
            
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            dist.all_gather_object(modality_losses_val, outputs_collated['loss_per_modality'])
            
            loss_here = np.vstack(losses_val).mean()
            modality_loss_here = np.vstack(modality_losses_val).mean(axis=0)
        else:
            loss_here = np.mean(outputs_collated['loss'])
            modality_loss_here = np.mean(np.vstack(outputs_collated['loss_per_modality']), axis=0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
        if 'val_losses_per_modality' not in self.logger.my_fantastic_logging:
            self.logger.my_fantastic_logging['val_losses_per_modality'] = []
        self.logger.log('val_losses_per_modality', modality_loss_here.tolist(), self.current_epoch)
         # ✅ Lesion-level F1 최종 출력
        if 'val_lesion_f1' in self.logger.my_fantastic_logging:
            f1_val = self.logger.my_fantastic_logging['val_lesion_f1'][-1]
            f1_loss_val = self.logger.my_fantastic_logging.get('val_lesion_f1_loss', [None])[-1]
            self.print_to_log_file(f"Lesion-level F1 (avg across modalities): {np.round(f1_val, 4)}")
            if f1_loss_val is not None:
                self.print_to_log_file(f"Lesion-level F1 loss: {np.round(f1_loss_val, 4)}")

    def on_epoch_start(self):
        self.logger.log('epoch_start_timestamps', time(), self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file(
            'train_loss', 
            np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4)
        )
        self.print_to_log_file(
            'val_loss', 
            np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4)
        )
        self.print_to_log_file(
            'Pseudo dice',
            [np.round(i, decimals=4) for i in self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]]
        )
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s"
        )
        
        if 'val_lesion_f1' in self.logger.my_fantastic_logging:
            self.print_to_log_file(f"Lesion-level F1: {np.round(self.logger.my_fantastic_logging['val_lesion_f1'][-1], decimals=4)}")
       # This Python code snippet is handling the calculation and logging of losses for different
       # modalities during training and validation phases of a machine learning model. Here is a
       # breakdown of what the code is doing:
        # 🎯 각 modality별 loss
        train_losses_per_modality = self.logger.my_fantastic_logging.get('train_losses_per_modality', None)
        val_losses_per_modality = self.logger.my_fantastic_logging.get('val_losses_per_modality', None)

        # 🛠 중복 출력 방지: 2차원이면 mean 처리
        if train_losses_per_modality and len(np.array(train_losses_per_modality[-1]).shape) > 1:
            train_losses_per_modality[-1] = np.mean(np.vstack(train_losses_per_modality[-1]), axis=0).tolist()

        if val_losses_per_modality and len(np.array(val_losses_per_modality[-1]).shape) > 1:
            val_losses_per_modality[-1] = np.mean(np.vstack(val_losses_per_modality[-1]), axis=0).tolist()

        modality_names = [d.split("_")[-1] for d in self.dataset_directories]
        val_losses_per_modality = self.logger.my_fantastic_logging.get('val_losses_per_modality', None)

        # 🎯 train loss 출력
        if train_losses_per_modality:
            for i, loss in enumerate(train_losses_per_modality[-1]):
                if i >= len(modality_names):
                    break
                name = modality_names[i]
                self.print_to_log_file(f" - train loss ({name}): {np.round(loss, 4)}")

        # 🎯 val loss 출력 및 로깅
        if val_losses_per_modality:
            val_losses_last = val_losses_per_modality[-1]
            for i, val_loss in enumerate(val_losses_last):
                if i >= len(modality_names):
                    continue
                name = modality_names[i]

                self.print_to_log_file(f"val_{name}_loss", np.round(val_loss, 4))

                # 로그 키 없으면 생성
                val_key = f'val_{name}_loss'
                if val_key not in self.logger.my_fantastic_logging:
                    self.logger.my_fantastic_logging[val_key] = []

                self.logger.log(val_key, val_loss, self.current_epoch)

            
        # checkpoint 저장 조건
        if (self.current_epoch + 1) % self.save_every == 0 and self.current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # best dice model 저장
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        # 🧠 Best lesion-level F1 기반 모델 저장
        val_f1_log = self.logger.my_fantastic_logging.get('val_lesion_f1', None)
        if val_f1_log is not None:
            val_lesion_f1 = val_f1_log[-1]
            
            if not hasattr(self, '_best_lesion_f1'):
                self._best_lesion_f1 = None

            if self._best_lesion_f1 is None or val_lesion_f1 > self._best_lesion_f1:
                self._best_lesion_f1 = val_lesion_f1
                self.print_to_log_file(f"New best lesion-level F1: {np.round(val_lesion_f1, 4)}")
                self.save_checkpoint(join(self.output_folder, 'checkpoint_best_lesionF1.pth'))
                
        val_losses_per_modality = self.logger.my_fantastic_logging.get('val_losses_per_modality', None)
        mean_val_loss = self.logger.my_fantastic_logging['val_losses'][-1]  # 전체 평균
        swi_loss = None
        if val_losses_per_modality is not None:
            swi_loss = val_losses_per_modality[-1][self.primary_network_index]

        if not hasattr(self, '_best_mean_loss'):
            self._best_mean_loss = None

        if self._best_mean_loss is None or mean_val_loss < self._best_mean_loss:
            self._best_mean_loss = mean_val_loss
            self.print_to_log_file(f"New best mean val loss: {np.round(mean_val_loss, 4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best_meanloss.pth'))

        if swi_loss is not None:
            if not hasattr(self, '_best_swi_loss'):
                self._best_swi_loss = None

            if self._best_swi_loss is None or swi_loss < self._best_swi_loss:
                self._best_swi_loss = swi_loss
                self.print_to_log_file(f"New best SWI val loss: {np.round(swi_loss, 4)}")
                self.save_checkpoint(join(self.output_folder, 'checkpoint_best_swi_loss.pth'))

        # rank 0만 로그 plot
        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                try:
                    network_weights = []
                    for net in self.networks:
                        mod = net.module if self.is_ddp else net
                        if isinstance(mod, OptimizedModule):
                            mod = mod._orig_mod
                        network_weights.append(mod.state_dict())

                    # ✅ optimizer 하나만 저장
                    optimizer_state = self.optimizer.state_dict()  # 🔥 수정

                    checkpoint = {
                        'network_weights': network_weights,
                        'optimizer_state': optimizer_state,  # 🔥 수정
                        'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                        'current_epoch': self.current_epoch + 1,
                        '_best_ema': self._best_ema,
                        '_best_lesion_f1': getattr(self, '_best_lesion_f1', None),
                        'trainer_name': self.__class__.__name__,
                        'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes
                    }

                    try:
                        checkpoint['init_args'] = {
                            k: v for k, v in self.my_init_kwargs.items()
                            if isinstance(v, (str, int, float, bool, type(None)))
                        }
                    except Exception as e:
                        self.print_to_log_file(f"⚠️ Skipping init_args due to serialization error: {e}")

                    try:
                        checkpoint['logging'] = self.logger.get_checkpoint()
                    except Exception as e:
                        self.print_to_log_file(f"⚠️ Skipping logging due to serialization error: {e}")

                    torch.save(checkpoint, filename)
                except Exception as e:
                    self.print_to_log_file(f"❌ Failed to save checkpoint: {e}")
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        else:
            checkpoint = filename_or_checkpoint

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint.get(
            'inference_allowed_mirroring_axes', self.inference_allowed_mirroring_axes)

        for i, net in enumerate(self.networks):
            net_weights = checkpoint['network_weights'][i]
            new_state_dict = {}
            for k, value in net_weights.items():
                key = k
                if key not in net.state_dict().keys() and key.startswith('module.'):
                    key = key[7:]
                new_state_dict[key] = value

            mod = net.module if self.is_ddp else net
            if isinstance(mod, OptimizedModule):
                mod._orig_mod.load_state_dict(new_state_dict)
            else:
                mod.load_state_dict(new_state_dict)

        # ✅ optimizer 하나만 로딩
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])  # 🔥 수정

        if self.grad_scaler is not None and checkpoint['grad_scaler_state'] is not None:
            self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        net = self.networks[self.primary_network_index]  # ✅ 대표 네트워크만 사용 (예: SWI)
        net.eval()

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. "
                                "If you encounter crashes in validation then rerun with --val")

        predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
            perform_everything_on_device=True, device=self.device, verbose=False,
            verbose_preprocessing=False, allow_tqdm=False
        )
        predictor.manual_initialization(
            net,
            self.plans_manager,
            self.configuration_manager,
            None,
            self.dataset_json,
            self.__class__.__name__,
            self.inference_allowed_mirroring_axes
        )

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1
                val_keys = val_keys[self.local_rank::dist.get_world_size()]

            dataset_val = self.dataset_class(
                self.preprocessed_dataset_folder,
                val_keys,
                folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
            )

            next_stages = self.configuration_manager.next_stage_names
            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []
            for i, k in enumerate(dataset_val.identifiers):
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                        allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                            allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, _, seg_prev, properties = dataset_val.load_case(k)
                data = data[:]
                if self.is_cascaded:
                    seg_prev = seg_prev[:]
                    data = np.vstack((
                        data,
                        convert_labelmap_to_one_hot(seg_prev, self.label_manager.foreground_labels, output_dtype=data.dtype)
                    ))

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)

                prediction = predictor.predict_sliding_window_return_logits(data)
                prediction = prediction.cpu()

                results.append(segmentation_export_pool.starmap_async(
                    export_prediction_from_logits, (
                        (prediction, properties, self.configuration_manager, self.plans_manager,
                        self.dataset_json, output_filename_truncated, save_probabilities),
                    )
                ))

                # if next stage exists, export for it too
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)
                        dataset_class = infer_dataset_class(expected_preprocessed_folder)
                        try:
                            tmp = dataset_class(expected_preprocessed_folder, [k])
                            d, _, _, _ = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(f"Next stage {n} skipped for {k}, missing preprocessed file.")
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file_truncated = join(output_folder, k)

                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                (prediction, target_shape, output_file_truncated, self.plans_manager,
                                self.configuration_manager, properties, self.dataset_json,
                                default_num_processes, dataset_class),
                            )
                        ))

                if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    dist.barrier()

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(
                join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                validation_output_folder,
                join(validation_output_folder, 'summary.json'),
                self.plans_manager.image_reader_writer_class(),
                self.dataset_json["file_ending"],
                self.label_manager.foreground_regions if self.label_manager.has_regions else self.label_manager.foreground_labels,
                self.label_manager.ignore_label,
                chill=True,
                num_processes=default_num_processes * dist.get_world_size() if self.is_ddp else default_num_processes
            )
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", metrics['foreground_mean']["Dice"], also_print_to_console=True)
            # ✅ F1 설정 로그
            self.print_to_log_file(f"✔ F1 loss integration is {'enabled' if self.use_f1_loss else 'disabled'}")

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()

    def run_training(self):
            self.on_train_start()

            for epoch in range(self.current_epoch, self.num_epochs):
                self.on_epoch_start()

                self.on_train_epoch_start()
                train_outputs = []
                for batch_id in range(self.num_iterations_per_epoch):
                    # 각 modality의 dataloader에서 batch 추출
                    batch_list = [next(dl) for dl in self.dataloader_train]
                    # train_step이 여러 배치를 동시에 받아 처리하도록 구현되어 있어야 함
                    train_outputs.append(self.train_step(batch_list))
                self.on_train_epoch_end(train_outputs)

                with torch.no_grad():
                    self.on_validation_epoch_start()
                    val_outputs = []
                    for batch_id in range(self.num_val_iterations_per_epoch):
                        val_batch_list = [next(dl) for dl in self.dataloader_val]
                        val_outputs.append(self.validation_step(val_batch_list))  # 수정된 부분
                    self.on_validation_epoch_end(val_outputs)

            # ✅ F1 loss 로그 (epoch 루프 직전)
            if self.local_rank == 0:
                self.print_to_log_file(f"▶ Training started with F1 loss = {'ON' if self.use_f1_loss else 'OFF'}")

                self.on_epoch_end()

            self.on_train_end()

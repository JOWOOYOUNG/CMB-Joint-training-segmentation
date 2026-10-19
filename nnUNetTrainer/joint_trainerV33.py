#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.data_augmentation.default_data_augmentation import get_default_augmentation
from nnunetv2.training.loss.deep_supervision import MultipleOutputLoss2
from nnunetv2.utilities.helpers import softmax_helper
from nnunetv2.utilities.utils_resunet import build_residual_encoder_unet_from_plans
from nnunetv2.training.dataloading.dataset_loading import unpack_dataset
from joint_dataloader import get_joint_dataloaders_from_env
from nnunetv2.training.dataloading.dataset_loading import load_dataset



class JointTrainerV3(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_files, fold, output_folder_names=None, dataset_directories=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, max_num_epochs=1000, use_meanloss_for_multitask=False, fixed_validation_set=None):
        super().__init__(plans_files, fold, output_folder_names, dataset_directories, batch_dice, stage, unpack_data,
                         deterministic, fp16)
         # ✅ 여기에 dataset 및 plan 파일 경로를 명시적으로 설정
        self.dataset_directories = [
            'nnUNet_preprocessed/Dataset513_phase',
            'nnUNet_preprocessed/Dataset514_swi'
        ]
        self.plans_files = [
            'nnUNet_preprocessed/Dataset513_phase/plans.json',
            'nnUNet_preprocessed/Dataset514_swi/plans.json'
        ]
        
        self.max_num_epochs = max_num_epochs
        self.initial_lr = 1e-2
        self.deep_supervision_scales = self.compute_deep_supervision_scales()
        self.ds_loss_weights = None

        self.pin_memory = True

        self.use_meanloss_for_multitask = use_meanloss_for_multitask
        self.fixed_validation_set = fixed_validation_set

    def initialize(self, training=True, force_load_plans=False):
        if not self.was_initialized:
            for output_folder in self.output_folders:
                maybe_mkdir_p(output_folder)

            ###########################
            # ✅ load planses from self.plans_files
            ###########################
            if force_load_plans or (self.planses is None):
                self.planses = []
                for p in self.plans_files:
                    self.planses.append(load_pickle(p))

            self.num_input_channels = 1  # 각 modality는 1채널씩
            self.num_classes = len(self.planses[0]['labels'])  # 보통 background + foreground = 2

            self.print_to_log_file(f"Loaded {len(self.planses)} modality plans")
            self.print_to_log_file(f"Input channels: {self.num_input_channels}, Classes: {self.num_classes}")

            for plans in self.planses:
                self.process_plans(plans)

            self.setup_DA_params()

            ################# Wrap loss for deep supervision #################
            net_numpool = len(self.net_num_pool_op_kernel_sizes)
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END #################

            self.folder_with_preprocessed_datas = []
            for dataset_directory, plans in zip(self.dataset_directories, self.planses):
                self.folder_with_preprocessed_datas.append(
                    join(dataset_directory, plans['data_identifier'] + f"_stage{self.stage}")
                )

            if training:
                self.dl_trs, self.dl_vals = self.get_basic_generators()

                if self.unpack_data:
                    print("unpacking dataset")
                    for folder_with_preprocessed_data in self.folder_with_preprocessed_datas:
                        unpack_dataset(folder_with_preprocessed_data)
                    print("done")
                else:
                    print("INFO: Not unpacking data! Training may be slow due to that.")

                self.tr_gens = []
                self.val_gens = []
                for i in range(len(self.dl_trs)):
                    tr_gen, val_gen = get_moreDA_augmentation(
                        self.dl_trs[i], self.dl_vals[i],
                        self.data_aug_params['patch_size_for_spatialtransform'],
                        self.data_aug_params,
                        deep_supervision_scales=self.deep_supervision_scales,
                        pin_memory=self.pin_memory,
                        use_nondetMultiThreadedAugmenter=False
                    )
                    self.tr_gens.append(tr_gen)
                    self.val_gens.append(val_gen)

                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_trs[0].keys())), also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_vals[0].keys())), also_print_to_console=False)

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            for network in self.networks:
                assert isinstance(network, (SegmentationNetwork, nn.DataParallel))

            # ✅ Set self.network to default for inference compatibility
            self.default_modality_idx = 1  # 0: phase, 1: SWI
            self.network = self.networks[self.default_modality_idx]
            self.was_initialized = True
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
            
    def load_joint_datasets(self):
        # 각 modality에 해당하는 dataset을 로딩함
        self.datasets = []
        for dataset_dir in self.dataset_directories:
            dataset = load_dataset(join(dataset_dir, 'dataset.json'))  # 또는 load_dataset_from_json(dataset.json)
            self.datasets.append(dataset)

        # fold="all" 일 때 self.dataset은 첫 번째 dataset 기준으로 설정
        self.dataset = self.datasets[0]


    def get_basic_generators(self):
        plans = self.planses[0]
        patch_size = plans['patch_size']
        batch_size = plans['batch_size']
        final_patch_size = plans['patch_size']

        tr_loader = get_joint_dataloaders_from_env(
            patch_size=patch_size,
            final_patch_size=final_patch_size,
            batch_size=batch_size,
            oversample_foreground_percent=0.33,
            transforms=self.get_train_transforms()
        )
        val_loader = get_joint_dataloaders_from_env(
            patch_size=patch_size,
            final_patch_size=final_patch_size,
            batch_size=1,
            oversample_foreground_percent=0.0,
            transforms=self.get_validation_transforms()
        )

        return tr_loader, val_loader
    
    def initialize_network(self):
        from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
        from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm

        self.networks = []

        for plans in self.planses:
            spacing = plans['spacing']
            architecture_kwargs = plans['architecture']['arch_kwargs']
            conv_op = convert_dim_to_conv_op(len(spacing))
            norm_op = get_matching_instancenorm(conv_op)

            network = ResidualEncoderUNet(
                input_channels=self.num_input_channels,
                num_classes=self.num_classes,
                conv_op=conv_op,
                norm_op=norm_op,
                norm_op_kwargs=architecture_kwargs.get('norm_op_kwargs', {'eps': 1e-5, 'affine': True}),
                dropout_op=None,
                dropout_op_kwargs=None,
                nonlin=nn.LeakyReLU,
                nonlin_kwargs=architecture_kwargs.get('nonlin_kwargs', {'inplace': True}),
                deep_supervision=True,
                **{k: architecture_kwargs[k] for k in [
                    'n_stages', 'features_per_stage', 'kernel_sizes', 'strides',
                    'n_blocks_per_stage', 'n_conv_per_stage_decoder', 'conv_bias'
                ]}
            )
            network.inference_apply_nonlin = softmax_helper
            if torch.cuda.is_available():
                network = network.cuda()
            self.networks.append(network)



    # ✅ joint training을 위한 initialize_optimizer_and_scheduler 및 run_training() 구현

    def initialize_optimizer_and_scheduler(self):
        """
        각 네트워크에 대해 SGD optimizer 설정
        """
        self.optimizers = []
        for i, network in enumerate(self.networks):
            assert network is not None, f"Network {i} is None. initialize_network must be called first."
            optimizer = torch.optim.SGD(
                network.parameters(),
                lr=self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=0.99,
                nesterov=True
            )
            self.optimizers.append(optimizer)
        self.lr_scheduler = None  # poly_lr 사용하므로 scheduler 없음

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        target = target[0]
        output = output[0]
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
             step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
             validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
             segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        Validation 시 SWI encoder만 사용하여 단일 모델로 예측 수행
        """
        for net in self.networks:
            net.do_ds = False

        self.current_network_id = 0  # 기본 SWI encoder를 사용한다고 가정
        ret = super().validate(
            do_mirroring=do_mirroring,
            use_sliding_window=use_sliding_window,
            step_size=step_size,
            save_softmax=save_softmax,
            use_gaussian=use_gaussian,
            overwrite=overwrite,
            validation_folder_name=validation_folder_name,
            debug=debug,
            all_in_gpu=all_in_gpu,
            segmentation_export_kwargs=segmentation_export_kwargs,
            run_postprocessing_on_folds=run_postprocessing_on_folds
        )

        for net in self.networks:
            net.do_ds = True

        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                          mirror_axes: Tuple[int] = None,
                                                          use_sliding_window: bool = True, step_size: float = 0.5,
                                                          use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                          pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                          verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        ensemble_outputs = []
        for network in self.networks:
            ds = network.do_ds
            network.do_ds = False
            pred = self.inference_predictor.predict_preprocessed_data_return_seg_and_softmax(
                network, data, do_mirroring, mirror_axes, use_sliding_window, step_size,
                use_gaussian, pad_border_mode, pad_kwargs, all_in_gpu, verbose, mixed_precision
            )
            ensemble_outputs.append(pred[1])
            network.do_ds = ds

        avg_softmax = np.mean(np.stack(ensemble_outputs, axis=0), axis=0)
        seg = avg_softmax.argmax(0)
        return seg, avg_softmax

    def run_iteration(self, data_generators, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dicts = [next(data_generator) for data_generator in data_generators]
        datas = [data_dict['data'] for data_dict in data_dicts]
        targets = [data_dict['target'] for data_dict in data_dicts]

        # To check for each plane
        # import matplotlib.pyplot as plt
        # plt.imshow(data0[0, 0, :, :], cmap='gray') ; plt.savefig('data0.png')
        # plt.imshow(data1[0, 0, :, :], cmap='gray') ; plt.savefig('data1.png')
        # plt.imshow(data2[0, 0, :, :], cmap='gray') ; plt.savefig('data2.png')
        # exit()

        datas = [maybe_to_torch(data) for data in datas]
        targets = [maybe_to_torch(target) for target in targets]

        if torch.cuda.is_available():
            datas = [to_cuda(data) for data in datas]
            targets = [to_cuda(target) for target in targets]

        [optimizer.zero_grad() for optimizer in self.optimizers]

        if self.fp16:
            with autocast():
                outputs = [network(data) for network, data in zip(self.networks, datas)]
                datas = None
                del datas
                losses = [self.loss(output, target) for output, target in zip(outputs, targets)]
                loss_sum = sum(losses)
                loss_mean = loss_sum / len(losses)
                
                self.print_to_log_file(
            f'{"Training" if do_backprop else "Validation"} losses: \
            {", ".join([f"L{i}:{loss.data:8.04f}" for i, loss in enumerate(losses)])}\
            , L-All: {loss_sum.data:8.04f}\
            , L-Mean: {loss_mean.data:8.04f}\
            ', also_print_to_console=False)

            if do_backprop:
                self.amp_grad_scaler.scale(loss_mean if self.use_meanloss_for_multitask else loss_sum).backward()
                
                for network, optimizer in zip(self.networks, self.optimizers):
                    # self.amp_grad_scaler.scale(la).backward()
                    self.amp_grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 12)
                    self.amp_grad_scaler.step(optimizer)
                    # self.amp_grad_scaler.update()
                
                self.amp_grad_scaler.update()
        else:
            raise NotImplementedError
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(outputs[0], targets[0])

        targets = None
        del targets

        return loss_mean.detach().cpu().numpy() if self.use_meanloss_for_multitask else loss_sum.detach().cpu().numpy()

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fixed_validation_set:
            from nnunet.utilities.fixed_validation_sets import FIXED_VALIDATION_SETS

            val_keys = FIXED_VALIDATION_SETS[self.fixed_validation_set][self.fold]
            tr_keys = list(self.datasets[0].keys())
            for v in val_keys:
                tr_keys.remove(v)
            self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
        elif self.fold == "all":
        # if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

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
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
        tr_keys.sort()
        val_keys.sort()
        self.dataset_trs = [OrderedDict() for _ in range(len(self.datasets))]
        for tr_key in tr_keys:
            for dataset_trs_idx, dataset in enumerate(self.datasets):
                self.dataset_trs[dataset_trs_idx][tr_key] = dataset[tr_key]
        self.dataset_vals = [OrderedDict() for _ in range(len(self.datasets))]
        for val_key in val_keys:
            for dataset_vals_idx, dataset in enumerate(self.datasets):
                self.dataset_vals[dataset_vals_idx][val_key] = dataset[val_key]

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch

        for i in range(len(self.optimizers)):
            self.optimizers[i].param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
            self.print_to_log_file("lr:", np.round(self.optimizers[i].param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        if self.epoch == 100 and self.all_val_eval_metrics[-1] == 0:
            for optimizer in self.optimizers:
                optimizer.param_groups[0]["momentum"] = 0.95
            for network in self.networks:
                network.apply(InitWeights_He(1e-2))
            self.print_to_log_file("Momentum reduced to 0.95 and weights reinitialized at epoch 100")

        return continue_training

    def run_training(self):
        self.maybe_update_lr(self.epoch)
        self.print_to_log_file(f'Use Meanloss for Multitask: {self.use_meanloss_for_multitask}')
        [net.do_ds for net in self.networks]

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file(f"\nEpoch: {self.epoch + 1}/{self.max_num_epochs}")
            
            epoch_losses = []
            for _ in range(self.num_iterations_per_epoch):  # 예: 250 iterations
                loss = self.run_iteration(self.tr_gens, do_backprop=True, run_online_evaluation=self.run_online_evaluation)
                epoch_losses.append(loss)

            self.print_to_log_file(f"Avg loss: {np.mean(epoch_losses):.4f}")
            self.maybe_update_lr()
            self.epoch += 1

        self.save_checkpoint(self.output_folder)
        self.print_to_log_file("Training completed.")

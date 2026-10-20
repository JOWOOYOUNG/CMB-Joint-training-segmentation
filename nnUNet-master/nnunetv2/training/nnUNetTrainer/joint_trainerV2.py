from copy import deepcopy
from typing import List

import torch
from torch import nn
import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


class JointTrainerV2(nnUNetTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.networks: List[nn.Module] = []
        self.optimizers: List[torch.optim.Optimizer] = []
        self.tr_gens = []
        self.val_gens = []
        self.use_meanloss_for_multitask = True  # default

    def build_network_architecture(self, *args, **kwargs):
        raise NotImplementedError("Use initialize_networks instead of build_network_architecture")

    def initialize_networks(self):
        self.networks = []
        for i in range(len(self.plans_files)):
            net = super().build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels, self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)
            if self.is_ddp:
                net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
                net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[self.local_rank])
            self.networks.append(net)

    def initialize(self):
        if not self.was_initialized:
            self._set_batch_size_and_oversample()
            self.num_input_channels = self._get_num_input_channels()
            self.initialize_networks()
            self.initialize_optimizer_and_scheduler()
            self.loss = self._build_loss()
            self.dataset_class = self._get_dataset_class()
            self.was_initialized = True
        else:
            raise RuntimeError("Trainer already initialized.")

    def initialize_optimizer_and_scheduler(self):
        self.optimizers = []
        for net in self.networks:
            opt = torch.optim.SGD(net.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                  momentum=0.99, nesterov=True)
            self.optimizers.append(opt)
        self.lr_scheduler = None

    def get_dataloaders(self):
        dataloaders_train, dataloaders_val = [], []
        for _ in range(len(self.networks)):
            dataset_tr, dataset_val = self.get_tr_and_val_datasets()
            dl_tr = nnUNetDataLoader(dataset_tr, self.batch_size, self.configuration_manager.patch_size,
                                     self.configuration_manager.patch_size, self.label_manager,
                                     oversample_foreground_percent=self.oversample_foreground_percent,
                                     probabilistic_oversampling=self.probabilistic_oversampling)
            dl_val = nnUNetDataLoader(dataset_val, self.batch_size, self.configuration_manager.patch_size,
                                      self.configuration_manager.patch_size, self.label_manager,
                                      oversample_foreground_percent=self.oversample_foreground_percent,
                                      probabilistic_oversampling=self.probabilistic_oversampling)
            n_proc = get_allowed_n_proc_DA()
            gen_tr = NonDetMultiThreadedAugmenter(dl_tr, None, num_processes=n_proc,
                                                  num_cached=max(6, n_proc // 2), pin_memory=True)
            gen_val = NonDetMultiThreadedAugmenter(dl_val, None, num_processes=max(1, n_proc // 2),
                                                   num_cached=max(3, n_proc // 4), pin_memory=True)
            dataloaders_train.append(gen_tr)
            dataloaders_val.append(gen_val)
        return dataloaders_train, dataloaders_val

    def run_iteration(self, data_generators, do_backprop=True, run_online_evaluation=False):
        data_dicts = [next(gen) for gen in data_generators]
        datas = [d['data'].to(self.device) for d in data_dicts]
        targets = [d['target'] for d in data_dicts]
        targets = [t[0].to(self.device) if isinstance(t, list) else t.to(self.device) for t in targets]

        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, enabled=self.device.type == 'cuda'):
            outputs = [net(data) for net, data in zip(self.networks, datas)]
            losses = [self.loss(o, t) for o, t in zip(outputs, targets)]
            loss_sum = sum(losses)
            loss_mean = loss_sum / len(losses)

        if do_backprop:
            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss_mean if self.use_meanloss_for_multitask else loss_sum).backward()
                for net, opt in zip(self.networks, self.optimizers):
                    self.grad_scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
                    self.grad_scaler.step(opt)
                self.grad_scaler.update()
            else:
                loss_mean.backward()
                for net, opt in zip(self.networks, self.optimizers):
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
                    opt.step()

        if run_online_evaluation:
            self.run_online_evaluation(outputs[0], targets[0])

        return loss_mean.detach().cpu().numpy() if self.use_meanloss_for_multitask else loss_sum.detach().cpu().numpy()

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()
        self.tr_gens, self.val_gens = self.get_dataloaders()
        self.set_deep_supervision_enabled(self.enable_deep_supervision)
        empty_cache(self.device)

    def on_train_end(self):
        for loader in self.tr_gens + self.val_gens:
            if hasattr(loader, '_finish'):
                loader._finish()
        empty_cache(self.device)

    def set_deep_supervision_enabled(self, enabled: bool):
        for net in self.networks:
            mod = net.module if self.is_ddp else net
            if hasattr(mod, 'decoder'):
                mod.decoder.deep_supervision = enabled

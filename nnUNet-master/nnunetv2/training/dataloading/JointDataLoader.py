import os
import warnings
from typing import List, Optional

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, load_json

from nnunetv2.training.dataloading.dataset_loading import (
    load_dataset,
    unpack_dataset,
    DataLoader3D,
    DataLoader2D,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import LabelManager


class JointDataLoader:
    def __init__(
        self,
        dataset_dirs: List[str],
        plans_files: List[str],
        patch_size,
        final_patch_size,
        batch_size,
        oversample_foreground_percent=0.0,
        transforms=None,
        use_3d=True,
    ):
        assert len(dataset_dirs) == len(plans_files), "Each dataset must have a matching plans file"

        self.loaders = []

        for dataset_dir, plans_file in zip(dataset_dirs, plans_files):
            data_dir = join(dataset_dir, "nnUNetPlans_3d_fullres")  # assume 3d_fullres default
            if not os.path.isdir(data_dir):
                raise RuntimeError(f"Missing folder: {data_dir}")

            # unpack npz to npy for speed
            if any([f.endswith('.npz') for f in os.listdir(data_dir)]):
                print(f"Unpacking dataset in: {data_dir}")
                unpack_dataset(data_dir)

            dataset = load_dataset(data_dir)
            plans = load_json(plans_file)
            plans_manager = PlansManager(plans_file)
            dataset_json = load_json(join(dataset_dir, "dataset.json"))
            label_manager = plans_manager.get_label_manager(dataset_json)

            if use_3d:
                loader = DataLoader3D(
                    dataset,
                    patch_size,
                    final_patch_size,
                    batch_size,
                    oversample_foreground_percent=oversample_foreground_percent,
                )
            else:
                loader = DataLoader2D(
                    dataset,
                    patch_size,
                    final_patch_size,
                    batch_size,
                    oversample_foreground_percent=oversample_foreground_percent,
                )

            if transforms is not None:
                loader = transforms(loader)

            self.loaders.append(loader)

    def __iter__(self):
        self.iters = [iter(loader) for loader in self.loaders]
        return self

    def __next__(self):
        return [next(it) for it in self.iters]


# ✅ 클래스 밖으로 이동!
def get_joint_dataloaders_from_env(patch_size, final_patch_size, batch_size,
                                    oversample_foreground_percent=0.0, transforms=None):
    dataset_dirs = os.environ.get("JOINT_DATASET_DIRS", "").split(',')
    plans_files = os.environ.get("JOINT_PLANS_FILES", "").split(',')

    dataset_dirs = [d.strip() for d in dataset_dirs if d.strip()]
    plans_files = [p.strip() for p in plans_files if p.strip()]

    assert all(os.path.isdir(d) for d in dataset_dirs), f"Some dataset dirs do not exist: {dataset_dirs}"
    assert all(os.path.isfile(p) for p in plans_files), f"Some plans files do not exist: {plans_files}"

    return JointDataLoader(dataset_dirs, plans_files, patch_size, final_patch_size, batch_size,
                           oversample_foreground_percent, transforms)

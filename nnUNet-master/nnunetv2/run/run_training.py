import multiprocessing
import os
import socket
from typing import Union, Optional

import nnunetv2
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from torch.backends import cudnn
from nnunetv2.training.nnUNetTrainer.Joint_nnUNetTrainer import JointTrainer
from nnunetv2.training.nnUNetTrainer.Joint_nnUNetTrainer_1opti import JointTrainer2

def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def get_trainer_from_args(dataset_name_or_id: Union[int, str],
                          configuration: str,
                          fold: int,
                          trainer_name: str = 'nnUNetTrainer',
                          plans_identifier: str = 'nnUNetPlans',
                          device: torch.device = torch.device('cuda'),
                          use_meanloss_for_multitask: bool = False,
                          primary_network_index: Optional[int] = None,
                          use_f1_loss: bool = False):

    trainer_class = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer_name, 'nnunetv2.training.nnUNetTrainer'
    )
    if trainer_class is None:
        raise RuntimeError(f'Could not find requested nnunet trainer {trainer_name}')
    assert issubclass(trainer_class, nnUNetTrainer), 'Trainer must inherit from nnUNetTrainer'

    if not str(dataset_name_or_id).startswith('Dataset'):
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError(f'dataset_name_or_id must be DatasetXXX_YYY or int, got {dataset_name_or_id}')
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

    if trainer_name.startswith('JointTrainer'):
        plans_files = [p for p in os.environ.get('JOINT_PLANS_FILES', '').split(',') if p.strip()]
        dataset_dirs = [d for d in os.environ.get('JOINT_DATASET_DIRS', '').split(',') if d.strip()]
        assert len(plans_files) == len(dataset_dirs), \
            "JOINT_PLANS_FILES and JOINT_DATASET_DIRS must be set and of equal length."

        plans = load_json(plans_files[0])
        dataset_json = load_json(join(dataset_dirs[0], 'dataset.json'))

        nnunet_trainer = trainer_class(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
            plans_files=plans_files,
            dataset_directories=dataset_dirs,
            use_meanloss_for_multitask=use_meanloss_for_multitask,
            use_f1_loss=use_f1_loss
        )
        if primary_network_index is not None:
            nnunet_trainer.primary_network_index = primary_network_index
    else:
        preprocessed_dataset_folder_base = join(nnUNet_preprocessed, dataset_name)
        plans_file = join(preprocessed_dataset_folder_base, plans_identifier + '.json')
        dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))
        plans = load_json(plans_file)

        nnunet_trainer = trainer_class(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device
        )

    return nnunet_trainer



def maybe_load_checkpoint(nnunet_trainer: nnUNetTrainer,
                          continue_training: bool,
                          validation_only: bool,
                          pretrained_weights_file: str = None):
    if continue_training and pretrained_weights_file is not None:
        raise RuntimeError('Cannot both continue a training AND load pretrained weights. '
                           'Pretrained weights can only be used at the beginning of the training.')

    expected_checkpoint_file = None

    if continue_training:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_latest.pth')
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_best.pth')
        if not isfile(expected_checkpoint_file):
            print("WARNING: Cannot continue training because no checkpoint was found. Starting new training...")
            expected_checkpoint_file = None

    elif validation_only:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            raise RuntimeError("Cannot run validation because the training is not finished yet!")

    elif pretrained_weights_file is not None:
        if not nnunet_trainer.was_initialized:
            nnunet_trainer.initialize()

        # ✅ JointTrainer일 경우, networks 리스트 전체에 대해 pretrained weights 적용
        if hasattr(nnunet_trainer, "networks") and isinstance(nnunet_trainer.networks, list):
            for net in nnunet_trainer.networks:
                load_pretrained_weights(net, pretrained_weights_file, verbose=True)
        else:
            # 일반 nnUNetTrainer의 경우
            load_pretrained_weights(nnunet_trainer.network, pretrained_weights_file, verbose=True)

    if expected_checkpoint_file is not None:
        nnunet_trainer.load_checkpoint(expected_checkpoint_file)



def setup_ddp(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def run_ddp(rank, dataset_name_or_id, configuration, fold, tr, p, disable_checkpointing, c, val,
            pretrained_weights, npz, val_with_best, world_size, use_meanloss_for_multitask, primary_network_index, use_f1_loss):
    setup_ddp(rank, world_size)
    
        # ✅ 안전한 디바이스 설정
    n_gpus = torch.cuda.device_count()
    if rank >= n_gpus:
        raise RuntimeError(f"[DDP] Invalid GPU rank {rank}: only {n_gpus} GPUs available")

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, tr, p, 
                                           device=torch.device('cuda', dist.get_rank()), use_meanloss_for_multitask=use_meanloss_for_multitask,primary_network_index=primary_network_index, use_f1_loss=use_f1_loss)


    if disable_checkpointing:
        nnunet_trainer.disable_checkpointing = disable_checkpointing

    assert not (c and val), f'Cannot set --c and --val flag at the same time. Dummy.'

    maybe_load_checkpoint(nnunet_trainer, c, val, pretrained_weights)

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not val:
        nnunet_trainer.run_training()

    if val_with_best:
        nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
    nnunet_trainer.perform_actual_validation(npz)
    cleanup_ddp()


def run_training(dataset_name_or_id: Union[str, int],
                 configuration: str, fold: Union[int, str],
                 trainer_class_name: str = 'nnUNetTrainer',
                 plans_identifier: str = 'nnUNetPlans',
                 pretrained_weights: Optional[str] = None,
                 num_gpus: int = 1,
                 export_validation_probabilities: bool = False,
                 continue_training: bool = False,
                 only_run_validation: bool = False,
                 disable_checkpointing: bool = False,
                 val_with_best: bool = False,
                 use_meanloss_for_multitask: bool = False,
                 primary_network_index: Optional[int] = None,
                 use_f1_loss: bool = False,
                 device: torch.device = torch.device('cuda')):
    if plans_identifier == 'nnUNetPlans':
        print("\n############################\n"
              "INFO: You are using the old nnU-Net default plans. We have updated our recommendations. "
              "Please consider using those instead! "
              "Read more here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md"
              "\n############################\n")
    if isinstance(fold, str):
        if fold != 'all':
            try:
                fold = int(fold)
            except ValueError as e:
                print(f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!')
                raise e

    if val_with_best:
        assert not disable_checkpointing, '--val_best is not compatible with --disable_checkpointing'

    if num_gpus > 1:
        assert device.type == 'cuda', f"DDP training (triggered by num_gpus > 1) is only implemented for cuda devices. Your device: {device}"

        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ.keys():
            port = str(find_free_network_port())
            print(f"using port {port}")
            os.environ['MASTER_PORT'] = port  # str(port)

        mp.spawn(run_ddp,
                 args=(
                     dataset_name_or_id,
                     configuration,
                     fold,
                     trainer_class_name,
                     plans_identifier,
                     disable_checkpointing,
                     continue_training,
                     only_run_validation,
                     pretrained_weights,
                     export_validation_probabilities,
                     val_with_best,
                     num_gpus,
                     use_meanloss_for_multitask,
                     primary_network_index,
                     use_f1_loss),
                 nprocs=num_gpus,
                 join=True)
    else:
        nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, trainer_class_name,
                                               plans_identifier, device=device, use_meanloss_for_multitask=use_meanloss_for_multitask, primary_network_index=primary_network_index, use_f1_loss=use_f1_loss)

        if disable_checkpointing:
            nnunet_trainer.disable_checkpointing = disable_checkpointing

        assert not (continue_training and only_run_validation), f'Cannot set --c and --val flag at the same time. Dummy.'

        maybe_load_checkpoint(nnunet_trainer, continue_training, only_run_validation, pretrained_weights)

        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        if not only_run_validation:
            nnunet_trainer.run_training()

        if val_with_best:
            nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
        nnunet_trainer.perform_actual_validation(export_validation_probabilities)


def run_training_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id', type=str,
                        help="Dataset name or ID to train with")
    parser.add_argument('configuration', type=str,
                        help="Configuration that should be trained")
    parser.add_argument('fold', type=str,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4.')

    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='[OPTIONAL] Specify a custom trainer (e.g., JointTrainer). Default: nnUNetTrainer')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='[OPTIONAL] Specify a custom plans identifier. Default: nnUNetPlans')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='[OPTIONAL] Path to nnU-Net checkpoint to use as pretrained model.')
    parser.add_argument('-num_gpus', type=int, default=1, required=False,
                        help='Number of GPUs to use for training')

    # ✅ JointTrainer 전용 옵션 (환경변수 통해 plans & dataset 전달)
    parser.add_argument('--joint_plans_files', type=str, required=False, default=None,
                        help='[JointTrainer only] Comma-separated list of plans file paths')
    parser.add_argument('--joint_dataset_dirs', type=str, required=False, default=None,
                        help='[JointTrainer only] Comma-separated list of dataset directories')

    parser.add_argument('--npz', action='store_true', required=False,
                        help='[OPTIONAL] Save softmax predictions from validation as npz.')
    parser.add_argument('--c', action='store_true', required=False,
                        help='[OPTIONAL] Continue training from latest checkpoint')
    parser.add_argument('--val', action='store_true', required=False,
                        help='[OPTIONAL] Only run the validation. Training must be finished.')
    parser.add_argument('--val_best', action='store_true', required=False,
                        help='[OPTIONAL] Validate using checkpoint_best.pth. Not compatible with --disable_checkpointing!')
    parser.add_argument('--disable_checkpointing', action='store_true', required=False,
                        help='[OPTIONAL] Disable saving checkpoints (for testing).')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Device to use: 'cuda', 'cpu', or 'mps' (Apple M1/M2)")
    parser.add_argument('--use_meanloss_for_multitask', action='store_true', required=False,
                    help='[JointTrainer only] Whether to average losses from multiple modalities.')
    # run_training.py 상단에 인자 추가
    parser.add_argument("--primary_network_index", type=int, default=None, 
                        help="Which modality (index) to use as primary for validation/inference (e.g., SWI = 1)")
    parser.add_argument('--use_f1_loss', action='store_true', required=False,
                    help='[JointTrainer only] Whether to include lesion-level F1 loss (default: off)')




    args = parser.parse_args()

    assert args.device in ['cpu', 'cuda', 'mps'], f"Invalid device: {args.device}"
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

    # ✅ JointTrainer용 환경변수 설정
    if args.tr.startswith('JointTrainer'):
        if args.joint_plans_files:
            os.environ['JOINT_PLANS_FILES'] = args.joint_plans_files
        if args.joint_dataset_dirs:
            os.environ['JOINT_DATASET_DIRS'] = args.joint_dataset_dirs

    run_training(args.dataset_name_or_id, args.configuration, args.fold,
                 args.tr, args.p, args.pretrained_weights, args.num_gpus,
                 args.npz, args.c, args.val, args.disable_checkpointing,
                 args.val_best, device=device, use_meanloss_for_multitask=args.use_meanloss_for_multitask, primary_network_index=args.primary_network_index, use_f1_loss=args.use_f1_loss)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # reduces the number of threads used for compiling. More threads don't help and can cause problems
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    # multiprocessing.set_start_method("spawn")
    run_training_entry()
    
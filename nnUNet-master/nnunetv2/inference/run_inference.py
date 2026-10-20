import argparse
import os
import torch
from nnunetv2.paths import maybe_mkdir_p
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def run_inference():
    parser = argparse.ArgumentParser(description='Run inference using nnU-Net or JointPredictor (modality-specific)')

    # Required
    parser.add_argument('-i', type=str, required=True, help='Input folder (e.g., imagesTs)')
    parser.add_argument('-o', type=str, required=True, help='Output folder')
    parser.add_argument('-d', type=str, required=True, help='Dataset name or ID (e.g., Dataset123_CMB)')
    parser.add_argument('-c', type=str, required=True, help='Configuration name (e.g., 3d_fullres)')

    # Optional
    parser.add_argument('-p', type=str, default='nnUNetPlans', help='Plans identifier')
    parser.add_argument('-tr', type=str, default='nnUNetTrainer', help='Trainer class name')
    parser.add_argument('-f', nargs='+', type=str, default=("all",), help='Folds to use (e.g., 0 1 2) or "all"')
    parser.add_argument('-chk', type=str, default='checkpoint_final.pth', help='Checkpoint to use')
    parser.add_argument('--modality_idx', type=int, default=0, help='Primary modality index for JointPredictor')
    parser.add_argument('--ensemble', action='store_true', help='Enable ensemble mode (only if supported)')

    parser.add_argument('-step_size', type=float, default=0.5, help='Sliding window step size')
    parser.add_argument('--disable_tta', action='store_true', help='Disable test-time augmentation')
    parser.add_argument('--save_probabilities', action='store_true', help='Save softmax probabilities as .npz')
    parser.add_argument('--continue_prediction', action='store_true', help='Skip already predicted cases')
    parser.add_argument('-npp', type=int, default=4, help='Num threads for preprocessing')
    parser.add_argument('-nps', type=int, default=2, help='Num threads for segmentation export')
    parser.add_argument('-prev_stage_predictions', type=str, default=None, help='Path to previous stage segmentations')
    parser.add_argument('-num_parts', type=int, default=1, help='Total parts in distributed inference')
    parser.add_argument('-part_id', type=int, default=0, help='Index of this part (starts from 0)')
    parser.add_argument('-device', type=str, default='cuda', help='Device to use: cuda | cpu | mps')
    parser.add_argument('--disable_progress_bar', action='store_true', help='Disable tqdm progress bar')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    # Load correct model folder
    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)
    if not os.path.isdir(args.o):
        maybe_mkdir_p(args.o)

    # Device setup
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

    # Predictor selection
    if args.tr == 'JointTrainer':
        from nnunetv2.inference.predictor_classes.joint_predictorV2 import JointPredictor
        predictor = JointPredictor(
            tile_step_size=args.step_size,
            use_gaussian=True,
            use_mirroring=not args.disable_tta,
            perform_everything_on_device=(args.device == 'cuda'),
            device=device,
            verbose=args.verbose,
            verbose_preprocessing=args.verbose,
            allow_tqdm=not args.disable_progress_bar
        )
        predictor.initialize_from_trained_model_folder(
            model_training_output_dir=model_folder,
            use_folds=None if args.ensemble else args.f,
            checkpoint_name=args.chk,
            primary_network_index=args.modality_idx
        )
    else:
        predictor = nnUNetPredictor(
            tile_step_size=args.step_size,
            use_gaussian=True,
            use_mirroring=not args.disable_tta,
            perform_everything_on_device=(args.device == 'cuda'),
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
        list_of_lists_or_source_folder=args.i,
        output_folder_or_list_of_truncated_output_files=args.o,
        save_probabilities=args.save_probabilities,
        overwrite=not args.continue_prediction,
        num_processes_preprocessing=args.npp,
        num_processes_segmentation_export=args.nps,
        folder_with_segs_from_prev_stage=args.prev_stage_predictions,
        num_parts=args.num_parts,
        part_id=args.part_id
    )


if __name__ == '__main__':
    run_inference()

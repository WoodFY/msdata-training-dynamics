import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import yaml
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from utils.file_utils import get_file_paths_grouped_by_class
from utils.split_utils import split_dataset_files_by_class_stratified
from utils.patch_ms import process_patching
from utils.select_patches import process_patch_selection
from datasets.transforms import get_augmentation_pipeline
from datasets.datasets import MSIMGDataset
from models.resnet_2d import build_resnet_2d
from models.densenet_2d import build_densenet_2d
from models.efficientnet_2d import build_efficientnet_2d
from callbacks.early_stopping import EarlyStopping
from utils.data_loader import load_ms_img_dataset
from utils.train_utils import train, test
from utils.metrics import calculate_bootstrap_ci


def set_seeds(seed):
    """
    Set random seeds for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_params_from_yaml(file_path, key=None):
    """
    Load parameters from a YAML file.

    :param file_path: Path to the YAML file.
    :return: Dictionary containing the parameters.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file {file_path} does not exist.")

    with open(file_path, 'r') as file:
        params = yaml.safe_load(file)

    if not isinstance(params, dict):
        raise ValueError("YAML file must contain a dictionary of parameters.")

    if key:
        if key not in params:
            raise KeyError(f"Key '{key}' not found in the YAML file.")
        return params.get(key)
    else:
        return params


def _update_file_path(file_instance: dict, prefix: str = None, new_dir: str = None, suffix: str = None) -> dict:
    """
    Update the file path by replacing the prefix with the new prefix.
    Example: 'dataset/old_prefix/class/file.npz' to 'dataset/new_prefix/class/file.npz'.

    :param file_instance: Dictionary containing the file information.
    :param prefix: Optional prefix to be added to the file path.
    :param new_dir: Optional new directory to be used instead of the current one.
    :param suffix: Optional suffix to be replaced in the file path.
    :return: Updated file path.
    """
    if 'file_path' not in file_instance:
        raise KeyError("The input dictionary must contain the 'file_path' key.")

    if prefix and new_dir:
        raise KeyError("The 'new_dir' and 'prefix' are mutually exclusive.")

    p = Path(file_instance['file_path'])

    dir_to_modify = p.parent.parent
    target_dir_name = None
    if prefix:
        target_dir_name = f"{prefix}_{dir_to_modify.name}"
    elif new_dir:
        target_dir_name = new_dir

    target_filename = p.name
    if suffix:
        target_filename = p.stem + suffix

    base_dir = dir_to_modify.parent
    class_dir = p.parent.name
    new_path = base_dir / target_dir_name / class_dir / target_filename

    updated_file_instance = file_instance.copy()
    updated_file_instance['file_path'] = str(new_path)
    return updated_file_instance


def _create_dataset(patches_list, positions_list, labels, return_positions, transform=None):
    return MSIMGDataset(
        patches_list=patches_list,
        positions_list=positions_list,
        labels=labels,
        return_positions=return_positions,
        transform=transform
    )


def run_experiment(args):
    set_seeds(args.random_seed)
    print(f"Dataset directory: {args.dataset_dir}")
    binned_dataset_dir = os.path.join(args.dataset_dir, args.bin_prefix)
    binned_file_paths_by_class = get_file_paths_grouped_by_class(base_dir=binned_dataset_dir, suffix='.npz')
    train_set, test_set = split_dataset_files_by_class_stratified(
        file_paths_by_class=binned_file_paths_by_class,
        train_size=0.8,
        test_size=0.2,
        random_seed=args.random_seed
    )

    patched_train_set = [_update_file_path(file_instance, prefix=args.patch_prefix) for file_instance in train_set]
    patched_test_set = [_update_file_path(file_instance, prefix=args.patch_prefix) for file_instance in test_set]
    is_patched_files_exist = all(Path(file_instance['file_path']).exists() for file_instance in (patched_train_set + patched_test_set))

    if is_patched_files_exist:
        print("Skipping patching processing steps.")
    else:
        print('Processing Patching...')
        process_patching(
            args, binned_dataset_dir=binned_dataset_dir, binned_file_paths=[file_instance['file_path'] for file_instance in train_set]
        )
        process_patching(
            args, binned_dataset_dir=binned_dataset_dir, binned_file_paths=[file_instance['file_path'] for file_instance in test_set]
        )

    selected_train_set = [_update_file_path(file_instance, args.select_prefix) for file_instance in patched_train_set]
    selected_test_set = [_update_file_path(file_instance, args.select_prefix) for file_instance in patched_test_set]
    is_selected_files_exist = all(Path(file_instance['file_path']).exists() for file_instance in (selected_train_set + selected_test_set))

    if is_selected_files_exist:
        print("Skipping patch selection processing steps.")
    else:
        print('Processing Patch Selection...')
        dir_name = os.path.dirname(binned_dataset_dir)
        base_name = os.path.basename(binned_dataset_dir)
        patched_dataset_dir = os.path.join(dir_name, f"{args.patch_prefix}_{base_name}")
        process_patch_selection(args, patched_dataset_dir=patched_dataset_dir, patched_file_paths=[file_instance['file_path'] for file_instance in patched_train_set])
        process_patch_selection(args, patched_dataset_dir=patched_dataset_dir, patched_file_paths=[file_instance['file_path'] for file_instance in patched_test_set])

    train_patches_list, train_positions_list, train_labels = load_ms_img_dataset(dataset=selected_train_set, label_mapping=args.label_mapping)
    test_patches_list, test_positions_list, test_labels = load_ms_img_dataset(dataset=selected_test_set, label_mapping=args.label_mapping)

    exp_dir_name = (f"{args.model_name}_{args.dataset_name}_{args.patch_strategy}_"
                    f"PATCH_{args.patch_params.get('patch_height')}x{args.patch_params.get('patch_width')}_"
                    f"IN_CHANNELS_{args.num_patches}_NUM_CLASSES_{args.num_classes}_BATCH_SIZE_{args.batch_size}")
    print(exp_dir_name)
    exp_base_dir = os.path.join(args.save_dir, exp_dir_name)

    if not os.path.exists(exp_base_dir):
        os.makedirs(exp_base_dir)

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_seed)
    fold_test_metrics_list = []

    for fold_idx, (train_fold_indices, valid_fold_indices) in enumerate(skf.split(train_patches_list, train_labels)):
        print(f"{args.model_name} Fold {fold_idx + 1}/{args.k_folds}")
        exp_model_name = f"FOLD_{fold_idx + 1}_{args.model_name}_{args.dataset_name}_IN_CHANNELS_{args.num_patches}_NUM_CLASSES_{args.num_classes}"

        train_fold_patches_list, train_fold_positions_list, train_fold_labels = \
            train_patches_list[train_fold_indices], train_positions_list[train_fold_indices], train_labels[train_fold_indices]
        valid_fold_patches_list, valid_fold_positions_list, valid_fold_labels = \
            train_patches_list[valid_fold_indices], train_positions_list[valid_fold_indices], train_labels[valid_fold_indices]
        print(f'X_train.shape: {train_fold_patches_list.shape}, y_train.shape: {train_fold_labels.shape}')
        print(f'X_valid.shape: {valid_fold_patches_list.shape}, y_valid.shape: {valid_fold_labels.shape}')
        print(f'X_test.shape: {test_patches_list.shape}, y_test.shape: {test_labels.shape}')

        model = None
        return_positions = False
        if 'ResNet' in args.model_name:
            model = build_resnet_2d(args)
            return_positions = False
        elif 'DenseNet' in args.model_name:
            model = build_densenet_2d(args)
            return_positions = False
        elif 'EfficientNet' in args.model_name:
            model = build_efficientnet_2d(args)
            return_positions = False

        if args.multi_gpu and torch.cuda.device_count() > 1:
            print(f'Using {torch.cuda.device_count()} GPUs for training.')
            print("DataParallel typically expects model on primary GPU (cuda:0). Moving model to cuda:0 before DataParallel.")
            model = model.to(args.device)
            model = nn.DataParallel(model)  # Wrap the models with DataParallel for multi-GPU support
        else:
            model = model.to(args.device)

        class_weights = compute_class_weight('balanced', classes=np.array(list(args.label_mapping.values())), y=train_fold_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=args.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        # criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        if args.early_stopping:
            early_stopping = EarlyStopping(patience=args.patience)
        else:
            early_stopping = None

        train_loader = DataLoader(
            _create_dataset(
                patches_list=train_fold_patches_list,
                positions_list=train_fold_positions_list,
                labels=train_fold_labels,
                return_positions=return_positions,
                transform=get_augmentation_pipeline()
            ),
            batch_size=args.batch_size,
            # num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True
        )
        valid_loader = DataLoader(
            _create_dataset(
                patches_list=valid_fold_patches_list,
                positions_list=valid_fold_positions_list,
                labels=valid_fold_labels,
                return_positions=return_positions,
                transform=None
            ),
            batch_size=args.batch_size,
            # num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True
        )
        test_loader = DataLoader(
            _create_dataset(
                patches_list=test_patches_list,
                positions_list=test_positions_list,
                labels=test_labels,
                return_positions=return_positions,
                transform=None
            ),
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True
        )

        train(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            epochs=args.epochs,
            device=args.device,
            exp_base_dir=exp_base_dir,
            exp_model_name=exp_model_name,
            metrics_visualization=True
        )

        accuracy, precision, recall, f1_score = test(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            label_mapping=args.label_mapping,
            device=args.device,
            exp_base_dir=exp_base_dir,
            exp_model_name=exp_model_name,
            metrics_visualization=True
        )

        fold_test_metrics_list.append({
            'Fold': fold_idx + 1,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score
        })

    fold_test_metrics_df = pd.DataFrame(fold_test_metrics_list)
    mean_metrics = fold_test_metrics_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']].mean()
    std_metrics = fold_test_metrics_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']].std()

    summary_stats_list = []
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
        metric_values = fold_test_metrics_df[metric].values
        ci_lower, ci_upper = calculate_bootstrap_ci(metric_values, random_seed=args.random_seed)
        summary_stats_list.append({
            'Metric': metric,
            'Mean_Test_on_Holdout': mean_metrics.get(metric, np.nan),
            'Std_Test_on_Holdout': std_metrics.get(metric, np.nan),
            '95% CI Lower': ci_lower,
            '95% CI Upper': ci_upper
        })

    print(exp_dir_name)
    summary_stats_df = pd.DataFrame(summary_stats_list)
    print("Summary Statistics (Mean, Std, 95% Bootstrap CI from K-Fold models tested on hold-out):")
    print(summary_stats_df)

    time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    fold_test_metrics_csv_path = os.path.join(exp_base_dir, f"{args.model_name}_{args.dataset_name}_NUM_CLASSES_{args.num_classes}_IN_CHANNELS_{args.num_patches}_KFOLD_Tested_on_Holdout_Metrics_{time_stamp}.csv")
    summary_stats_csv_path = os.path.join(exp_base_dir, f"{args.model_name}_{args.dataset_name}_NUM_CLASSES_{args.num_classes}_IN_CHANNELS_{args.num_patches}_KFOLD_Tested_on_Holdout_Summary_Stats_{time_stamp}.csv")
    fold_test_metrics_df.to_csv(fold_test_metrics_csv_path, index=False)
    summary_stats_df.to_csv(summary_stats_csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='Mass Spectra 2D Image')
    parser.add_argument('--root_dir', type=str, default='../', help='Root directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--model_name', type=str, default='ResNet50', help='Model name')
    parser.add_argument('--dataset_name', type=str, help='Dataset to use')

    parser.add_argument('--patch_strategy', type=str, required=True, choices=['GP', 'DAPS'], help='Strategy to generate patches')
    parser.add_argument('--score_strategy', type=str, default='Entropy', choices=['Entropy', 'Mean'], help='Strategy to calculate patch scores (e.g. Entropy: 1D image entropy, Mean: mean intensity)')
    parser.add_argument('--num_patches', type=int, default=64, help='Number of patches to be selected')

    parser.add_argument('--k_folds', type=int, default=5, help='Number of patches to be selected')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=64, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--early_stopping', action='store_true', help='Use early stopping')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--random_seed', type=int, default=3407, help='Random seed for reproducibility')

    args = parser.parse_args()

    if args.multi_gpu:
        args.device = torch.device("cuda:0")

    # Set save directory
    save_dir = os.path.join(args.root_dir, args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir

    args.dataset_params = load_params_from_yaml('../configs/dataset_config.yaml', key=args.dataset_name)
    if args.dataset_params is None:
        raise ValueError(f"Dataset parameters for '{args.dataset_name}' not found in the YAML file.")
    args.bin_prefix = f"MZ_{args.dataset_params.get('mz_min')}-{args.dataset_params.get('mz_max')}_BIN_SIZE_{args.dataset_params.get('bin_size')}"

    patch_params = load_params_from_yaml('../configs/patch_config.yaml', key=args.patch_strategy)
    args.patch_params = patch_params
    if patch_params is None:
        raise ValueError(f"Patch parameters for strategy '{args.patch_strategy}' not found in the YAML file.")
    if args.patch_strategy == 'GP':
        args.patch_prefix = (f"{args.patch_strategy}_PATCH_{patch_params.get('patch_height')}x{patch_params.get('patch_width')}_"
                             f"OVERLAP_{patch_params.get('overlap_row')}x{patch_params.get('overlap_col')}")
    elif args.patch_strategy == 'DAPS':
        args.patch_prefix = (
            f"{args.patch_strategy}_PATCH_{patch_params.get('patch_height')}x{patch_params.get('patch_width')}_"
            f"WINDOW_{args.patch_params.get('window_size')}_INT_THR_{args.patch_params.get('intensity_threshold')}_"
            f"DENS_THR_{args.patch_params.get('density_threshold')}_MIN_PKS_{args.patch_params.get('min_peaks_in_patch')}")
    else:
        raise ValueError(f"Invalid patch strategy: {args.patch_strategy}.")

    args.select_prefix = f'{args.score_strategy}_{args.num_patches}'

    dataset_dict = {
        'SPNS': f"datasets/SPNS/",
        'CD': f"datasets/CD/",
    }
    dataset_dir = os.path.join(args.root_dir, dataset_dict[args.dataset_name])
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")
    args.dataset_dir = dataset_dir

    label_mapping = args.dataset_params.get('label_mapping')
    args.label_mapping = label_mapping
    args.num_classes = len(label_mapping)
    args.in_channels = args.num_patches

    run_experiment(args)


if __name__ == '__main__':
    main()
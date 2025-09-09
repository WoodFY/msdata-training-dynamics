import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import math
import yaml
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight

from utils.file_utils import get_file_paths_grouped_by_class
from utils.split_utils import split_dataset_files_by_class_stratified
from utils.data_loader import load_ms_dataset
from datasets.datasets import MSDataset
from models.resnet_1d import build_resnet_1d
from models.densenet_1d import build_densenet_1d
from models.efficientnet_1d import build_efficientnet_1d
from callbacks.early_stopping import EarlyStopping
from utils.train_utils import train, test
from utils.ml_train_utils import train_test_ml
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


def _create_dataset(X, y, transform=False):
    return MSDataset(X=X, y=y, transform=transform)


def run_experiment(args):
    set_seeds(args.random_seed)
    print(f"Dataset directory: {args.dataset_dir}")
    file_paths_by_class = get_file_paths_grouped_by_class(base_dir=args.dataset_dir, suffix='.csv')
    train_set, test_set = split_dataset_files_by_class_stratified(
        file_paths_by_class=file_paths_by_class,
        train_size=0.8,
        test_size=0.2,
        random_seed=args.random_seed
    )
    X_train, y_train = load_ms_dataset(dataset=train_set, label_mapping=args.label_mapping, mz_min=args.mz_min, mz_max=args.mz_max, bin_size=args.bin_size)
    X_test, y_test = load_ms_dataset(dataset=test_set, label_mapping=args.label_mapping, mz_min=args.mz_min, mz_max=args.mz_max, bin_size=args.bin_size)

    exp_dir_name = (f"{args.model_name}_{args.dataset_name}_NUM_BINS_{args.num_bins}_"
                    f"NUM_CLASSES_{args.num_classes}_BATCH_SIZE_{args.batch_size}")
    print(exp_dir_name)
    exp_base_dir = os.path.join(args.save_dir, exp_dir_name)

    if not os.path.exists(exp_base_dir):
        os.makedirs(exp_base_dir)

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_seed)
    fold_test_metrics_list = []

    for fold_idx, (train_fold_indices, valid_fold_indices) in enumerate(skf.split(X_train, y_train)):
        print(f"{args.model_name} Fold {fold_idx + 1}/{args.k_folds}")
        exp_model_name = f"FOLD_{fold_idx + 1}_{args.model_name}_{args.dataset_name}_NUM_CLASSES_{args.num_classes}"

        X_train_fold, y_train_fold = X_train[train_fold_indices], y_train[train_fold_indices]
        X_valid_fold, y_valid_fold = X_train[valid_fold_indices], y_train[valid_fold_indices]
        print(f'X_train_fold.shape: {X_train_fold.shape}, y_train_fold.shape: {y_train_fold.shape}')
        print(f'X_valid_fold.shape: {X_valid_fold.shape}, y_valid_fold.shape: {y_valid_fold.shape}')
        print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')

        train_loader = DataLoader(
            _create_dataset(X=X_train_fold, y=y_train_fold, transform=True),
            batch_size=args.batch_size,
            shuffle=True,
        )
        valid_loader = DataLoader(
            _create_dataset(X=X_valid_fold, y=y_valid_fold, transform=True),
            batch_size=args.batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            _create_dataset(X=X_test, y=y_test, transform=True),
            batch_size=args.batch_size,
            shuffle=False,
        )

        model = None
        if args.model_name in ['RF', 'SVM', 'LDA', 'XGBoost']:
            if args.model_name == 'RF' or args.model_name == 'RandomForest':
                model = RandomForestClassifier(random_state=args.random_seed)
            elif args.model_name == 'SVM':
                model = SVC(kernel='rbf', probability=True, random_state=args.random_seed)
            elif args.model_name == 'LDA':
                model = LinearDiscriminantAnalysis()
            elif args.model_name == 'XGBoost':
                model = XGBClassifier(random_state=args.random_seed)
            else:
                raise ValueError(f'Unknown model: {args.model_name}')

            accuracy, precision, recall, f1_score = train_test_ml(
                model=model,
                train_set=(X_train_fold, y_train_fold),
                test_set=(X_test, y_test),
                label_mapping=args.label_mapping,
                exp_base_dir=exp_base_dir,
                exp_model_name=exp_model_name,
                metrics_visualization=True
            )
        else:
            if 'ResNet' in args.model_name:
                model = build_resnet_1d(args)
            elif 'DenseNet' in args.model_name:
                model = build_densenet_1d(args)
            elif 'EfficientNet' in args.model_name:
                model = build_efficientnet_1d(args)

            if args.multi_gpu and torch.cuda.device_count() > 1:
                print(f'Using {torch.cuda.device_count()} GPUs for training.')
                print("DataParallel typically expects model on primary GPU (cuda:0). Moving model to cuda:0 before DataParallel.")
                model = model.to(args.device)
                model = nn.DataParallel(model)  # Wrap the models with DataParallel for multi-GPU support
            else:
                model = model.to(args.device)

            class_weights = compute_class_weight('balanced', classes=np.array(list(args.label_mapping.values())), y=y_train_fold)
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=args.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            # criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

            if args.early_stopping:
                early_stopping = EarlyStopping(patience=args.patience)
            else:
                early_stopping = None

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
    fold_test_metrics_csv_path = os.path.join(exp_base_dir, f"{args.model_name}_{args.dataset_name}_num_classes_{args.num_classes}_kfold_tested_on_holdout_metrics_{time_stamp}.csv")
    summary_stats_csv_path = os.path.join(exp_base_dir, f"{args.model_name}_{args.dataset_name}_num_classes_{args.num_classes}_kfold_tested_on_holdout_summary_stats_{time_stamp}.csv")
    fold_test_metrics_df.to_csv(fold_test_metrics_csv_path, index=False)
    summary_stats_df.to_csv(summary_stats_csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='Mass Spectra 1D Peak List Classification Experiment')
    parser.add_argument('--root_dir', type=str, default='../', help='Root directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--model_name', type=str, default='ResNet50', help='Model name')
    parser.add_argument('--dataset_name', type=str, required=True, help='List of datasets to use')

    parser.add_argument('--k_folds', type=int, default=5, help='Number of patches to be selected')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=64, help='Number of epochs')
    parser.add_argument('--device', type=str, default=None, help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--early_stopping', action='store_true', help='Use early stopping')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--random_seed', type=int, default=3407, help='Random seed for reproducibility')

    args = parser.parse_args()

    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.multi_gpu:
        args.device = torch.device("cuda:0")

    # Set save directory
    save_dir = os.path.join(args.root_dir, args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir

    dataset_dict = {
        'SPNS': f"datasets/SPNS/PEAK_LIST",
        'RCC': f"datasets/RCC/Positive/PEAK_LIST",
        'CD': f"datasets/CD/PEAK_LIST"
    }
    dataset_dir = os.path.join(args.root_dir, dataset_dict[args.dataset_name])
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")
    args.dataset_dir = dataset_dir
    args.in_channels = 1
    dataset_params = load_params_from_yaml('../configs/dataset_config.yaml', key=args.dataset_name)
    args.mz_min = dataset_params.get('mz_min')
    args.mz_max = dataset_params.get('mz_max')
    args.bin_size = dataset_params.get('bin_size')
    args.num_bins = math.ceil((args.mz_max - args.mz_min) / args.bin_size)
    label_mapping = dataset_params.get('label_mapping')
    args.label_mapping = label_mapping
    args.num_classes = len(label_mapping)

    run_experiment(args)


if __name__ == '__main__':
    main()
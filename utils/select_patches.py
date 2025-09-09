import os
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from functools import partial
from multiprocessing import Pool

from utils.score_patches import parallel_calculate_patches_scores, calculate_average_scores


def _create_save_path(file_path, prefix):
    """
    Create a save path based on the original file path and a prefix.
    """
    p = Path(file_path)

    file_name = p.name
    class_name = p.parent.name
    class_parent_dir_name = p.parent.parent.name
    dataset_dir = p.parent.parent.parent

    new_dir_name = f"{prefix}_{class_parent_dir_name}"
    save_dir = dataset_dir / new_dir_name / class_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / file_name
    return save_path


def select_top_k_patches(patched_file_path, prefix, sorted_indices, top_k, padding_value=0.0):
    try:
        if not (os.path.exists(patched_file_path) and os.path.getsize(patched_file_path) > 0):
            raise FileNotFoundError(f"File not found or empty: {patched_file_path}")

        save_path = _create_save_path(patched_file_path, prefix)
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            return True

        with np.load(patched_file_path) as patched_file_data:
            if 'patches' not in patched_file_data or 'positions' not in patched_file_data:
                raise ValueError(f"'patches' or 'positions' not found in {patched_file_path}")

            patches = patched_file_data['patches']
            positions = patched_file_data['positions']

        num_patches = len(patches)
        top_k_indices = sorted_indices[:top_k]

        padding_patch = np.full(patches[0].shape, padding_value, dtype=patches[0].dtype)
        padding_position = np.array([-1, -1], dtype=positions[0].dtype)

        selected_patches = []
        selected_positions = []
        padding_mask = []
        for idx in top_k_indices:
            selected_patches.append(patches[idx])
            selected_positions.append(positions[idx])
            padding_mask.append(False)

        num_selected = len(selected_patches)
        if num_selected < top_k:
            num_padding = top_k - num_selected

            selected_patches.extend([padding_patch] * num_padding)
            selected_positions.extend([padding_position] * num_padding)
            padding_mask.extend([True] * num_padding)

        selected_patches = np.array(selected_patches)
        selected_positions = np.array(selected_positions)
        padding_mask = np.array(padding_mask)

        np.savez_compressed(save_path, patches=selected_patches, positions=selected_positions, padding_mask=padding_mask)

        return True
    except Exception as e:
        raise RuntimeError(f"Error processing {patched_file_path}: {e}")


def parallel_select_top_k_patches(patched_file_paths, prefix, sorted_indices, top_k, workers=4):
    """
    Select top K patches based on the provided indices and save them.

    :param patched_file_paths: A list of file paths to the pseudo MS images.
    :param prefix: Prefix for the save path pattern.
    :param selection_strategy: Strategy for selecting patches ('class_average' or 'per_file)
    :param sorted_indices: Sorted indices of patches to select from.
    :param top_k: Number of top patches to select.
    :param workers: Number of worker processes to use.
    """
    with Pool(processes=workers) as pool:
        worker = partial(
            select_top_k_patches,
            prefix=prefix,
            sorted_indices=sorted_indices,
            top_k=top_k
        )

        results = list(
            tqdm(
                pool.imap_unordered(worker, patched_file_paths),
                total=len(patched_file_paths),
                desc=f'Selecting top K patches',
            )
        )

    success_rate = sum(results) / len(results)
    print(f"Patch Selection completed. Success rate: {success_rate:.2%}")


def process_patch_selection(args, patched_dataset_dir, patched_file_paths):
    """
    Process the score calculation and selection of top K patches from the patched MS files.
    """
    # Selecting Top K Patches
    print('Selecting Top K Patches...')
    print(f'Generating global sorted indices using {args.score_strategy} scores...')
    if args.patch_strategy == 'GP':
        global_sorted_indices_file_path = os.path.join(
            patched_dataset_dir,
            f"{args.patch_strategy}_PATCH_{args.patch_params.get('patch_height')}x{args.patch_params.get('patch_width')}_"
            f"OVERLAP_{args.patch_params.get('overlap_row')}x{args.patch_params.get('overlap_col')}_"
            f"GLOBAL_{args.score_strategy}_INDICES.pkl"
        )
    elif args.patch_strategy == 'DAPS':
        global_sorted_indices_file_path = os.path.join(
            patched_dataset_dir,
            f"{args.patch_strategy}_PATCH_{args.patch_params.get('patch_height')}x{args.patch_params.get('patch_width')}_"
            f"MIN_PKS_{args.patch_params.get('min_peaks_in_patch')}_BIN_SIZE_{args.dataset_params.get('bin_size')}_"
            f"Global_{args.score_strategy}_Indices.pkl"
        )
    else:
        raise ValueError(f"Unknown patch strategy '{args.patch_strategy}'")

    if os.path.exists(global_sorted_indices_file_path):
        print(f"Loading existing global sorted indices from {global_sorted_indices_file_path}")
        with open(global_sorted_indices_file_path, 'rb') as f:
            global_sorted_indices = pickle.load(f)
    else:
        print(f"Generate global average indices from {len(patched_file_paths)} files...")
        print('Calculating Scores for patches...')
        scores_list = parallel_calculate_patches_scores(
            patched_file_paths=patched_file_paths,
            score_strategy=args.score_strategy,
            workers=args.num_workers
        )

        print(f'Calculating global average {args.score_strategy} scores for {len(patched_file_paths)} files...')
        avg_scores = calculate_average_scores(scores_list=scores_list)
        global_sorted_indices = np.argsort(avg_scores)[::-1]

        print(f"Saving global average {args.score_strategy} sorted indices to {global_sorted_indices_file_path}")
        with open(global_sorted_indices_file_path, 'wb') as f:
            pickle.dump(global_sorted_indices, f)

    print(f"Selecting Top K Patches for {len(patched_file_paths)} files...")
    parallel_select_top_k_patches(
        patched_file_paths=patched_file_paths,
        prefix=args.select_prefix,
        sorted_indices=global_sorted_indices,
        top_k=args.num_patches,
        workers=args.num_workers,
    )

    print('Top K Patches Selection Process Completed.')

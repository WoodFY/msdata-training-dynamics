import os
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime

from utils.metrics import calculate_confusion_matrix, calculate_roc_auc


def plot_metrics(save_dir, plot_title, metrics, titles):
    n = len(metrics)
    plt.figure(figsize=(15, 5 * (n // 2 + n % 2)))
    for i, (train_metrics, valid_metrics) in enumerate(metrics):
        plt.subplot(n // 2 + n % 2, 2, i + 1)
        plt.plot(train_metrics, label='Train')
        plt.plot(valid_metrics, label='Valid')
        plt.title(titles[i])
        plt.xlabel('Epoch')
        plt.ylabel(titles[i])
        plt.legend()
    plt.suptitle(plot_title.split('.')[0])
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    metrics_file_path = os.path.join(save_dir, f"{plot_title}_train_valid_metrics_plot_{time_stamp}.png")
    plt.savefig(metrics_file_path)
    # plt.show()
    plt.close()


def plot_confusion_matrix(targets, predicts, label_mapping, plot_title, save_dir, cm=None):
    if cm is None:
        cm, class_labels = calculate_confusion_matrix(targets, predicts, label_mapping)
    else:
        class_labels = [label for label, _ in sorted(label_mapping.items(), key=lambda item: item[1])]

    # visualize confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Target')
    plt.title(f"{plot_title.split('.')[0]} Confusion Matrix")

    time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    cm_file_path = os.path.join(save_dir, f"{plot_title.split('.')[0]}_confusion_matrix_plot_{time_stamp}.png")
    plt.savefig(cm_file_path)
    # plt.show()
    plt.close()


def format_plot_title(filename_part: str) -> str:
    model_map = {
        'RF': 'Random Forest',
        'SVM': 'Support Vector Machine',
        'LDA': 'Linear Discriminant Analysis'
    }

    parts = filename_part.split('_')

    fold_info = ""
    try:
        fold_index = parts.index('FOLD')
        fold_number = parts[fold_index + 1]
        fold_info = f"FOLD {fold_number}"
    except ValueError:
        fold_info = "N/A"

    model_name = "Unknown Model"
    for part in parts:
        if part in model_map:
            model_name = model_map[part]
            break
        elif part in ['ResNet50', 'DenseNet121', 'EfficientNetB0']:
            model_name = part
            break

    dataset_name = parts[3]

    formatted_title = f"ROC Curve for {model_name}\non the {dataset_name} Dataset"
    return formatted_title

def plot_roc_auc_curve(targets, predicts, label_mapping, plot_title, save_dir):
    """
    Plot ROC curve

    Args:
        fpr: false positive rate
        tpr: true positive rate
        auc_scores: area under the curve
        num_classes: number of classes

    Returns:
        None
    """
    num_classes = len(label_mapping)
    fpr, tpr, auc_scores = calculate_roc_auc(targets, predicts, num_classes)
    class_labels = [label for label, _ in sorted(label_mapping.items(), key=lambda item: item[1])]

    plt.figure(figsize=(8, 6))
    for i, class_label in enumerate(class_labels):
        plt.plot(fpr[i], tpr[i], label=f'{class_label} (AUC = {auc_scores[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    formatted_title = format_plot_title(plot_title.split('.')[0])
    plt.title(formatted_title, fontsize=20, pad=10)
    plt.legend(loc='lower right', fontsize=20)

    time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    roc_auc_file_path = os.path.join(save_dir, f"{plot_title}_roc_auc_plot_{time_stamp}")
    plt.savefig(f'{roc_auc_file_path}.svg', dpi=300, bbox_inches='tight')
    plt.close()
"""
Training and evaluation script for DINOv3-based fish segmentation.

This script:
- Builds DataLoaders using the glob-based fish dataset.
- Instantiates a DINOv3SegmentationModel with a frozen backbone.
- Trains only the decoder using BCEWithLogitsLoss.
- Tracks per-iteration and per-epoch loss and IoU.
- Saves advanced training plots and the model checkpoint.
- Visualizes a set of random, best, and worst IoU examples.
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import set_seed, create_dataloaders
from model import DINOv3SegmentationModel


def calculate_iou_batch(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute IoU (Intersection over Union) for a batch of predictions and labels.

    Parameters
    ----------
    preds : torch.Tensor
        Logits predicted by the model, shape (B, 1, H, W).
    labels : torch.Tensor
        Ground-truth masks in {0, 1}, shape (B, 1, H, W).

    Returns
    -------
    float
        IoU score for the whole batch.
    """
    preds_bin = (torch.sigmoid(preds) > 0.5).float()
    intersection = (preds_bin * labels).sum()
    union = preds_bin.sum() + labels.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return float(iou.item())


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_train: bool = True,
) -> (List[float], List[float]):
    """
    Run a single epoch over the given DataLoader.

    Parameters
    ----------
    model : nn.Module
        Segmentation model.
    loader : DataLoader
        DataLoader for train or validation set.
    optimizer : Optimizer
        Optimizer used during training.
    criterion : nn.Module
        Loss function (e.g., BCEWithLogitsLoss).
    device : torch.device
        Target device to compute on.
    is_train : bool
        If True, gradients are computed and weights are updated.

    Returns
    -------
    losses : List[float]
        List of loss values for each batch in the epoch.
    ious : List[float]
        List of IoU values for each batch in the epoch.
    """
    if is_train:
        model.decoder.train()
    else:
        model.decoder.eval()

    losses: List[float] = []
    ious: List[float] = []

    with torch.set_grad_enabled(is_train):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)

            if is_train:
                loss.backward()
                optimizer.step()

            losses.append(float(loss.item()))
            ious.append(calculate_iou_batch(outputs, masks))

    return losses, ious


def save_advanced_plots(
    train_data: Dict[str, List[List[float]]],
    val_data: Dict[str, List[List[float]]],
    save_dir: str,
) -> None:
    """
    Save a collection of detailed training curves to disk.

    The function produces:
    1. Train iteration-wise loss.
    2. Train iteration-wise IoU.
    3. Train vs validation loss (mean ± std per epoch).
    4. Validation iteration-wise loss.
    5. Train vs validation IoU (mean ± std per epoch).
    6. Validation iteration-wise IoU.

    Parameters
    ----------
    train_data : dict
        Dictionary containing 'losses' and 'ious' for each epoch (list of lists).
    val_data : dict
        Same structure as train_data but for the validation set.
    save_dir : str
        Output directory where the PNG plots will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Flatten per-epoch lists into a single per-iteration list
    train_iter_loss = [item for sub in train_data["losses"] for item in sub]
    train_iter_iou = [item for sub in train_data["ious"] for item in sub]
    val_iter_loss = [item for sub in val_data["losses"] for item in sub]
    val_iter_iou = [item for sub in val_data["ious"] for item in sub]

    epochs = range(1, len(train_data["losses"]) + 1)

    # Per-epoch statistics
    t_loss_mean = [np.mean(x) for x in train_data["losses"]]
    t_loss_std = [np.std(x) for x in train_data["losses"]]
    v_loss_mean = [np.mean(x) for x in val_data["losses"]]
    v_loss_std = [np.std(x) for x in val_data["losses"]]

    t_iou_mean = [np.mean(x) for x in train_data["ious"]]
    t_iou_std = [np.std(x) for x in train_data["ious"]]
    v_iou_mean = [np.mean(x) for x in val_data["ious"]]
    v_iou_std = [np.std(x) for x in val_data["ious"]]

    # 1. Train iteration loss
    plt.figure(figsize=(12, 6))
    plt.plot(train_iter_loss, label="Train Iteration Loss", linewidth=1)
    plt.title("Train Iteration Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "1_Train_Iteration_Loss.png"))
    plt.close()

    # 2. Train iteration IoU
    plt.figure(figsize=(12, 6))
    plt.plot(train_iter_iou, label="Train Iteration IoU", linewidth=1)
    plt.title("Train Iteration IoU")
    plt.xlabel("Iteration")
    plt.ylabel("IoU")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "2_Train_Iteration_IoU.png"))
    plt.close()

    # 3. Train vs validation loss (mean ± std per epoch)
    plt.figure(figsize=(12, 6))
    t_loss_mean_arr = np.array(t_loss_mean)
    t_loss_std_arr = np.array(t_loss_std)
    v_loss_mean_arr = np.array(v_loss_mean)
    v_loss_std_arr = np.array(v_loss_std)

    plt.plot(epochs, t_loss_mean_arr, "o-", label="Train Loss")
    plt.fill_between(
        epochs,
        t_loss_mean_arr - t_loss_std_arr,
        t_loss_mean_arr + t_loss_std_arr,
        alpha=0.2,
        label="Train Loss ± Std",
    )
    plt.plot(epochs, v_loss_mean_arr, "o-", label="Val Loss")
    plt.fill_between(
        epochs,
        v_loss_mean_arr - v_loss_std_arr,
        v_loss_mean_arr + v_loss_std_arr,
        alpha=0.2,
        label="Val Loss ± Std",
    )
    plt.title("Train vs Validation Loss (Mean ± Std)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "3_Train_Val_Loss_MeanStd.png"))
    plt.close()

    # 4. Validation iteration loss
    plt.figure(figsize=(12, 6))
    plt.plot(val_iter_loss, label="Val Iteration Loss", linewidth=1)
    plt.title("Validation Iteration Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "4_Val_Iteration_Loss.png"))
    plt.close()

    # 5. Train vs validation IoU (mean ± std per epoch)
    plt.figure(figsize=(12, 6))
    t_iou_mean_arr = np.array(t_iou_mean)
    t_iou_std_arr = np.array(t_iou_std)
    v_iou_mean_arr = np.array(v_iou_mean)
    v_iou_std_arr = np.array(v_iou_std)

    plt.plot(epochs, t_iou_mean_arr, "o-", label="Train IoU")
    plt.fill_between(
        epochs,
        t_iou_mean_arr - t_iou_std_arr,
        t_iou_mean_arr + t_iou_std_arr,
        alpha=0.2,
        label="Train IoU ± Std",
    )
    plt.plot(epochs, v_iou_mean_arr, "o-", label="Val IoU")
    plt.fill_between(
        epochs,
        v_iou_mean_arr - v_iou_std_arr,
        v_iou_mean_arr + v_iou_std_arr,
        alpha=0.2,
        label="Val IoU ± Std",
    )
    plt.title("Train vs Validation IoU (Mean ± Std)")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "5_Train_Val_IoU_MeanStd.png"))
    plt.close()

    # 6. Validation iteration IoU
    plt.figure(figsize=(12, 6))
    plt.plot(val_iter_iou, label="Val Iteration IoU", linewidth=1)
    plt.title("Validation Iteration IoU")
    plt.xlabel("Iteration")
    plt.ylabel("IoU")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "6_Val_Iteration_IoU.png"))
    plt.close()

    print(f"[INFO] Training plots saved under: {save_dir}")


def show_results(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 300,
) -> None:
    """
    Visualize segmentation performance on the test set.

    The function:
    - Collects (image, ground truth, prediction, IoU) tuples for up to `max_samples`.
    - Randomly shuffles the list (with a fixed seed if you want reproducibility).
    - Displays:
        * 20 random examples,
        * top-5 best IoU examples,
        * bottom-5 worst IoU examples.

    Parameters
    ----------
    model : nn.Module
        Trained segmentation model.
    loader : DataLoader
        DataLoader over the test set.
    device : torch.device
        Device on which the model is running.
    max_samples : int
        Maximum number of samples to collect for visualization.
    """
    model.eval()
    all_res = []

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            for k in range(images.size(0)):
                inter = (preds[k] * masks[k]).sum().item()
                union = preds[k].sum().item() + masks[k].sum().item() - inter
                iou = (inter + 1e-6) / (union + 1e-6)

                all_res.append(
                    {
                        "img": images[k].cpu(),
                        "gt": masks[k].cpu(),
                        "pred": preds[k].cpu(),
                        "iou": iou,
                    }
                )

            if len(all_res) >= max_samples:
                break

    # Shuffle results to get random examples
    import random as pyrandom

    pyrandom.shuffle(all_res)

    # Sort by IoU to pick best and worst
    sorted_res = sorted(all_res, key=lambda x: x["iou"])
    worst_5 = sorted_res[:5]
    best_5 = list(reversed(sorted_res[-5:]))
    random_20 = all_res[:20]

    def plot_list(lst, title_prefix: str) -> None:
        plt.figure(figsize=(15, 3 * len(lst)))
        for idx, item in enumerate(lst):
            # Unnormalize image for visualization
            img = item["img"].permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array(
                [0.485, 0.456, 0.406]
            )
            img = np.clip(img, 0.0, 1.0)

            plt.subplot(len(lst), 3, idx * 3 + 1)
            plt.imshow(img)
            plt.title(f"{title_prefix} #{idx + 1}\nIoU: {item['iou'] * 100:.1f}%")
            plt.axis("off")

            plt.subplot(len(lst), 3, idx * 3 + 2)
            plt.imshow(item["pred"][0], cmap="gray")
            plt.title("Prediction")
            plt.axis("off")

            plt.subplot(len(lst), 3, idx * 3 + 3)
            plt.imshow(item["gt"][0], cmap="gray")
            plt.title("Ground Truth")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    print("\n[VIS] 20 Random Predictions")
    plot_list(random_20, "Random")

    print("\n[VIS] Top 5 Best Predictions")
    plot_list(best_5, "Best")

    print("\n[VIS] Bottom 5 Worst Predictions")
    plot_list(worst_5, "Worst")


def main(args: argparse.Namespace) -> None:
    # 1. Reproducibility and device selection
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 2. Data
    train_loader, test_loader = create_dataloaders(
        root_dir=args.data_root,
        img_size=args.image_size,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    # 3. Model
    model = DINOv3SegmentationModel(
        dino_model_id=args.model_id,
        hf_token=args.hf_token,
        freeze_backbone=True,
    ).to(device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)

    if args.eval_only and os.path.exists(checkpoint_path):
        print(f"[INFO] Loading model from {checkpoint_path} (evaluation only).")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        show_results(model, test_loader, device, max_samples=args.max_vis_samples)
        return

    # 4. Training setup
    optimizer = optim.AdamW(model.decoder.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    train_history = {"losses": [], "ious": []}
    val_history = {"losses": [], "ious": []}

    # 5. Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\n[INFO] Epoch {epoch}/{args.epochs}")

        train_losses, train_ious = run_epoch(
            model, train_loader, optimizer, criterion, device, is_train=True
        )
        val_losses, val_ious = run_epoch(
            model, test_loader, optimizer, criterion, device, is_train=False
        )

        train_history["losses"].append(train_losses)
        train_history["ious"].append(train_ious)
        val_history["losses"].append(val_losses)
        val_history["ious"].append(val_ious)

        print(
            f"       Train Mean IoU: {np.mean(train_ious) * 100:.2f}% | "
            f"Val Mean IoU: {np.mean(val_ious) * 100:.2f}%"
        )

    # 6. Save checkpoint
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[INFO] Model checkpoint saved at: {checkpoint_path}")

    # 7. Save plots
    save_advanced_plots(
        train_history,
        val_history,
        save_dir=args.plots_dir,
    )

    # 8. Show qualitative results
    show_results(model, test_loader, device, max_samples=args.max_vis_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DINOv3-based segmentation model on fish dataset."
    )

    # Data and paths
    parser.add_argument(
        "--data-root",
        type=str,
        default="fish_dataset/Fish_Dataset/Fish_Dataset",
        help="Root directory of the fish dataset.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs/checkpoints",
        help="Directory where the model checkpoint will be stored.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="dinov3_fish_segmentation_final.pth",
        help="Filename for the model checkpoint.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="outputs/plots",
        help="Directory where training plots will be saved.",
    )

    # Model / HF
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/dinov3-vits16-pretrain-lvd1689m",
        help="Hugging Face model id for the DINOv3 backbone.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face access token (Individuals access token shall be typed).",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument(
        "--image-size", type=int, default=448, help="Target image size (square)."
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Fraction of samples used for training."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Evaluation / visualization
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="If set, load the checkpoint and only run qualitative evaluation.",
    )
    parser.add_argument(
        "--max-vis-samples",
        type=int,
        default=300,
        help="Maximum number of samples to collect for qualitative visualization.",
    )

    args = parser.parse_args()
    main(args)

#!/usr/bin/env python
"""
Training script for ASGML multimodal learning.

Usage:
    python scripts/train.py --config configs/cremad.yaml
    python scripts/train.py --config configs/cremad.yaml --no-asgml  # Baseline
"""

import argparse
import os
import sys
import yaml
import random
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MultimodalModel
from src.datasets import CREMADDataset, AVEDataset, KineticsSoundsDataset
from src.losses import ASGMLLoss
from src.utils import setup_logger, AverageMeter, MetricTracker


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset(config: dict, split: str):
    """Create dataset based on config."""
    name = config["dataset"]["name"]
    root = config["dataset"]["root"]

    if name == "cremad":
        return CREMADDataset(root=root, split=split)
    elif name == "ave":
        return AVEDataset(root=root, split=split)
    elif name == "kinetics_sounds":
        return KineticsSoundsDataset(root=root, split=split)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_optimizer(model: nn.Module, config: dict):
    """Create optimizer based on config."""
    opt_name = config["training"]["optimizer"]
    lr = config["training"]["lr"]
    momentum = config["training"].get("momentum", 0.9)
    weight_decay = config["training"].get("weight_decay", 1e-4)

    if opt_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif opt_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif opt_name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def get_scheduler(optimizer, config: dict):
    """Create LR scheduler based on config."""
    sched_name = config["training"].get("scheduler", "step")

    if sched_name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["training"].get("step_size", 40),
            gamma=config["training"].get("gamma", 0.1),
        )
    elif sched_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["epochs"],
        )
    elif sched_name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


def compute_gradient_norms(model: nn.Module, modalities: list) -> dict:
    """Compute gradient norms for each modality encoder."""
    grad_norms = {}
    for modality in modalities:
        total_norm = 0.0
        for p in model.encoders[modality].parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norms[modality] = total_norm ** 0.5
    return grad_norms


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    asgml_loss: ASGMLLoss,
    device: torch.device,
    epoch: int,
    config: dict,
    logger,
    writer: SummaryWriter = None,
):
    """Train for one epoch."""
    model.train()

    loss_meter = AverageMeter("Loss")
    metric_tracker = MetricTracker(config["evaluation"]["metrics"])
    modalities = config["dataset"]["modalities"]

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        inputs = {m: batch[m].to(device) for m in modalities}
        targets = batch["label"].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits, unimodal_logits, features = model(inputs, return_features=True)

        # Compute gradient norms from previous step (for ASGML tracking)
        # Note: On first batch, this will be zero
        grad_norms = compute_gradient_norms(model, modalities)

        # Compute loss
        if config["asgml"]["enabled"]:
            loss, loss_info = asgml_loss(
                logits, unimodal_logits, targets, grad_norms
            )
        else:
            loss = nn.functional.cross_entropy(logits, targets)
            loss_info = {"fusion_loss": loss.item()}

        # Backward pass
        loss.backward()

        # Apply gradient scaling for ASGML
        if config["asgml"]["enabled"]:
            update_mask = asgml_loss.get_update_mask()
            grad_scales = asgml_loss.get_gradient_scales()

            for modality in modalities:
                if update_mask[modality]:
                    scale = grad_scales[modality]
                    for p in model.encoders[modality].parameters():
                        if p.grad is not None:
                            p.grad.data *= scale
                else:
                    # Zero out gradients for non-updating modalities
                    for p in model.encoders[modality].parameters():
                        if p.grad is not None:
                            p.grad.data.zero_()

        optimizer.step()

        # Update metrics
        loss_meter.update(loss.item(), targets.size(0))
        metric_tracker.update(logits, targets)

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

        # Log to tensorboard
        if writer is not None and batch_idx % config["logging"]["log_interval"] == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("train/loss", loss.item(), global_step)

            if config["asgml"]["enabled"]:
                for m in modalities:
                    writer.add_scalar(
                        f"train/staleness_{m}",
                        loss_info["staleness"][m],
                        global_step,
                    )
                    writer.add_scalar(
                        f"train/learning_speed_{m}",
                        loss_info["learning_speeds"][m],
                        global_step,
                    )

    # Compute epoch metrics
    metrics = metric_tracker.compute()
    metrics["loss"] = loss_meter.avg

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: dict,
):
    """Evaluate model on dataset."""
    model.eval()

    loss_meter = AverageMeter("Loss")
    metric_tracker = MetricTracker(config["evaluation"]["metrics"])
    modalities = config["dataset"]["modalities"]

    for batch in tqdm(dataloader, desc="Evaluating"):
        inputs = {m: batch[m].to(device) for m in modalities}
        targets = batch["label"].to(device)

        logits, _, _ = model(inputs, return_features=False)
        loss = nn.functional.cross_entropy(logits, targets)

        loss_meter.update(loss.item(), targets.size(0))
        metric_tracker.update(logits, targets)

    metrics = metric_tracker.compute()
    metrics["loss"] = loss_meter.avg

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train ASGML model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--no-asgml", action="store_true", help="Disable ASGML (baseline)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override ASGML setting if specified
    if args.no_asgml:
        config["asgml"]["enabled"] = False

    # Set seed
    set_seed(args.seed)

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config['dataset']['name']}_{'asgml' if config['asgml']['enabled'] else 'baseline'}_{timestamp}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Setup logger
    logger = setup_logger("asgml", log_file=str(output_dir / "train.log"))
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Config: {config}")

    # Setup tensorboard
    writer = None
    if config["logging"]["tensorboard"]:
        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create datasets
    train_dataset = get_dataset(config, "train")
    test_dataset = get_dataset(config, "test")

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )

    # Create model
    model = MultimodalModel(
        modalities=config["dataset"]["modalities"],
        num_classes=config["dataset"]["num_classes"],
        encoder_config={
            m: {"backbone": config["model"]["backbone"], "pretrained": config["model"]["pretrained"]}
            for m in config["dataset"]["modalities"]
        },
        fusion_type=config["model"]["fusion_type"],
        feature_dim=config["model"]["feature_dim"],
        fusion_dim=config["model"]["fusion_dim"],
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # Create ASGML loss
    asgml_loss = ASGMLLoss(
        modalities=config["dataset"]["modalities"],
        tau_base=config["asgml"]["tau_base"],
        tau_min=config["asgml"]["tau_min"],
        tau_max=config["asgml"]["tau_max"],
        beta=config["asgml"]["beta"],
        lambda_comp=config["asgml"]["lambda_comp"],
        gamma=config["asgml"]["gamma"],
        window_size=config["asgml"]["window_size"],
    )

    # Training loop
    best_acc = 0.0
    for epoch in range(1, config["training"]["epochs"] + 1):
        # Reset ASGML state at start of epoch
        asgml_loss.reset()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, asgml_loss,
            device, epoch, config, logger, writer
        )

        # Evaluate
        test_metrics = evaluate(model, test_loader, device, config)

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Log
        logger.info(
            f"Epoch {epoch}: "
            f"Train Loss={train_metrics['loss']:.4f}, "
            f"Train Acc={train_metrics['accuracy']:.4f}, "
            f"Test Acc={test_metrics['accuracy']:.4f}, "
            f"Test F1={test_metrics['f1_macro']:.4f}"
        )

        if writer is not None:
            writer.add_scalar("test/accuracy", test_metrics["accuracy"], epoch)
            writer.add_scalar("test/f1_macro", test_metrics["f1_macro"], epoch)
            writer.add_scalar("test/loss", test_metrics["loss"], epoch)

        # Save checkpoint
        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "config": config,
            }, output_dir / "best_model.pt")
            logger.info(f"New best model saved with accuracy: {best_acc:.4f}")

        # Periodic checkpoint
        if epoch % config["logging"]["save_interval"] == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, output_dir / f"checkpoint_epoch{epoch}.pt")

    # Final summary
    logger.info(f"Training complete. Best accuracy: {best_acc:.4f}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()

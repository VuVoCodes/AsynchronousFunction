#!/usr/bin/env python
"""
Training script for ASGML multimodal learning.

Supports multiple experimental conditions:
- Baseline: Standard joint training (all modalities update every step)
- Fixed frequency: Dominant modality updates every k steps
- Fixed staleness: Dominant modality uses τ-step-old gradients
- Adaptive ASGML: Probe-driven staleness/frequency adaptation

Usage:
    # Baseline (no ASGML)
    python scripts/train.py --config configs/cremad.yaml --mode baseline

    # Fixed frequency (dominant updates every 2 steps)
    python scripts/train.py --config configs/cremad.yaml --mode frequency --fixed-ratio 2

    # Fixed staleness (dominant uses 2-step-old gradients)
    python scripts/train.py --config configs/cremad.yaml --mode staleness --fixed-staleness 2

    # Adaptive ASGML (full method)
    python scripts/train.py --config configs/cremad.yaml --mode adaptive
"""

import argparse
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
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MultimodalModel, ProbeManager
from src.datasets import CREMADDataset, AVEDataset, KineticsSoundsDataset
from src.losses import (
    ASGMLLoss,
    ASGMLScheduler,
    compute_gradient_norms,
    apply_staleness_gradients,
)
from src.utils import setup_logger, AverageMeter
from src.utils.metrics import MetricTracker


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
        fps = config["dataset"].get("fps", 1)
        num_frames = config["dataset"].get("num_frames", 1)
        return CREMADDataset(root=root, split=split, fps=fps, num_frames=num_frames)
    elif name == "ave":
        return AVEDataset(root=root, split=split)
    elif name == "kinetics_sounds":
        return KineticsSoundsDataset(root=root, split=split)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_optimizer(
    model: nn.Module,
    config: dict,
    modality_lr_scales: dict = None,
):
    """
    Create optimizer based on config with optional per-modality learning rates.

    Args:
        model: The multimodal model
        config: Training configuration
        modality_lr_scales: Dict mapping modality names to LR scale factors.
            E.g., {"audio": 0.5, "visual": 1.0} means audio gets half the base LR.
            If None, all modalities use the same LR.
    """
    opt_name = config["training"]["optimizer"]
    base_lr = config["training"]["lr"]
    momentum = config["training"].get("momentum", 0.9)
    weight_decay = config["training"].get("weight_decay", 1e-4)

    # Build parameter groups for per-modality LRs
    if modality_lr_scales is not None:
        param_groups = []

        # Encoder parameters (per-modality LR)
        for modality in model.modalities:
            scale = modality_lr_scales.get(modality, 1.0)
            param_groups.append({
                'params': list(model.encoders[modality].parameters()),
                'lr': base_lr * scale,
                'name': f'encoder_{modality}',
            })

        # Fusion + classifier parameters (base LR)
        fusion_params = (
            list(model.fusion.parameters()) +
            list(model.classifier.parameters()) +
            list(model.unimodal_classifiers.parameters())
        )
        param_groups.append({
            'params': fusion_params,
            'lr': base_lr,
            'name': 'fusion',
        })

        params = param_groups
    else:
        params = model.parameters()

    if opt_name == "sgd":
        return optim.SGD(
            params,
            lr=base_lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif opt_name == "adam":
        return optim.Adam(
            params,
            lr=base_lr,
            weight_decay=weight_decay,
        )
    elif opt_name == "adamw":
        return optim.AdamW(
            params,
            lr=base_lr,
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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: ASGMLLoss,
    scheduler: ASGMLScheduler,
    probe_manager: ProbeManager,
    device: torch.device,
    epoch: int,
    config: dict,
    logger,
    writer: SummaryWriter = None,
    scaler: GradScaler = None,
    use_amp: bool = False,
):
    """
    Train for one epoch with ASGML.

    This implements the full ASGML training loop:
    1. Forward pass through model
    2. Compute loss
    3. Backward pass (compute gradients)
    4. Store gradients in staleness buffer (if staleness mode)
    5. Apply staleness/frequency mask to gradients
    6. Optimizer step (fusion head always updates)
    7. Periodically train and evaluate probes
    """
    model.train()

    loss_meter = AverageMeter("Loss")
    metric_tracker = MetricTracker(config["evaluation"]["metrics"])
    modalities = config["dataset"]["modalities"]

    eval_freq = config["asgml"].get("eval_freq", 100)
    probe_train_steps = config["asgml"].get("probe_train_steps", 50)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        global_step = (epoch - 1) * len(dataloader) + batch_idx

        # Move data to device (non_blocking for async transfer)
        inputs = {m: batch[m].to(device, non_blocking=True) for m in modalities}
        targets = batch["label"].to(device, non_blocking=True)

        # ========== Forward Pass ==========
        optimizer.zero_grad()

        with autocast(device_type='cuda', enabled=use_amp):
            logits, unimodal_logits, features = model(inputs, return_features=True)

            # Get update mask and staleness values from scheduler
            update_mask = scheduler.get_update_mask()

            # ========== Compute Loss ==========
            loss, loss_dict = loss_fn(
                logits, unimodal_logits, targets, update_mask
            )

        # ========== Backward Pass ==========
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            # CRITICAL FIX: Unscale gradients BEFORE storing/modifying
            # This prevents AMP scale factor mismatch when applying stale gradients
            scaler.unscale_(optimizer)

            # Check for inf/nan gradients after unscaling
            # scaler.unscale_() can produce inf/nan on overflow
            grads_valid = True
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        grads_valid = False
                        break
        else:
            loss.backward()
            grads_valid = True

        # ========== Compute Gradient Norms (for tracking) ==========
        grad_norms = compute_gradient_norms(model, modalities) if grads_valid else {m: 0.0 for m in modalities}

        # ========== Store Gradients (for staleness mode) ==========
        # Skip storing if AMP overflow detected (grads_valid=False)
        # fp32 mode: grads_valid is always True (no overflow issues)
        if scheduler.mode == "staleness" and grads_valid:
            for m in modalities:
                encoder_params = {
                    name: param
                    for name, param in model.named_parameters()
                    if f"encoders.{m}" in name
                }
                scheduler.staleness_buffer.store_gradients(
                    m, encoder_params, global_step
                )

        # ========== Apply ASGML Gradient Modifications ==========
        # Only modify gradients if they are valid (no inf/nan from AMP overflow)
        if config["asgml"]["enabled"] and grads_valid:
            if scheduler.mode == "staleness":
                # Apply stale gradients for dominant modality
                staleness_vals = scheduler.get_staleness_values()
                grad_scales = scheduler.get_gradient_scales()

                for m in modalities:
                    if staleness_vals[m] > 0:
                        apply_staleness_gradients(
                            model,
                            scheduler.staleness_buffer,
                            m,
                            staleness_vals[m],
                            grad_scales[m],
                        )
            else:
                # Frequency mode: reduce encoder gradients for non-updating modalities
                # Options:
                #   - hard_mask=True (default): Zero all encoder gradients (original behavior)
                #   - hard_mask=False: Scale gradients by small factor to preserve some signal
                hard_mask = config["asgml"].get("hard_frequency_mask", False)
                soft_scale = config["asgml"].get("soft_mask_scale", 0.1)  # 10% of gradient

                for m in modalities:
                    if not update_mask[m]:
                        for name, param in model.named_parameters():
                            if f"encoders.{m}" in name and param.grad is not None:
                                if hard_mask:
                                    param.grad.data.zero_()
                                else:
                                    # Soft masking: reduce but don't eliminate gradient
                                    # This preserves fusion-path learning signal
                                    param.grad.data.mul_(soft_scale)

        # ========== Optimizer Step ==========
        # NOTE: Fusion head always updates regardless of modality schedules
        if use_amp and scaler is not None:
            # Gradients already unscaled above, so use step() directly
            # Check for inf/nan before stepping (scaler.step handles this)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # ========== Update Scheduler ==========
        scheduler.dynamics.update(
            {m: loss_dict[f'unimodal_{m}'].item() for m in modalities},
            grad_norms,
        )
        scheduler.step()

        # ========== Update Metrics ==========
        loss_meter.update(loss.item(), targets.size(0))
        metric_tracker.update(logits, targets)

        # ========== Probe Training & Evaluation (periodic) ==========
        if (batch_idx + 1) % eval_freq == 0:
            # Train probes on current features (DETACHED - safe)
            probe_manager.train_probes(features, targets, num_steps=probe_train_steps)

            # Evaluate probes
            probe_results = probe_manager.evaluate_probes(features, targets)
            utilization_gap = probe_manager.compute_utilization_gap()
            dominant = probe_manager.get_dominant_modality()

            # Update scheduler with probe signals (for adaptive mode)
            if scheduler.adaptation == "adaptive" and utilization_gap is not None:
                scheduler.set_dominant_modality(dominant)
                scheduler.update_from_utilization(
                    probe_manager.get_utilization_scores(),
                    utilization_gap,
                )

            # Log probe metrics
            if writer is not None:
                for m in modalities:
                    writer.add_scalar(
                        f"probe/accuracy_{m}",
                        probe_results[m]['accuracy'],
                        global_step,
                    )
                if utilization_gap is not None:
                    writer.add_scalar("probe/utilization_gap", utilization_gap, global_step)

        # ========== Logging ==========
        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "mask": str({m: int(v) for m, v in update_mask.items()}),
        })

        if writer is not None and batch_idx % config["logging"]["log_interval"] == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/fusion_loss", loss_dict['fusion'].item(), global_step)

            for m in modalities:
                writer.add_scalar(f"train/grad_norm_{m}", grad_norms[m], global_step)
                writer.add_scalar(f"train/unimodal_loss_{m}", loss_dict[f'unimodal_{m}'].item(), global_step)
                writer.add_scalar(f"train/staleness_{m}", scheduler.current_tau[m], global_step)
                writer.add_scalar(f"train/update_{m}", int(update_mask[m]), global_step)

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
    probe_manager: ProbeManager = None,
):
    """Evaluate model on dataset."""
    model.eval()

    loss_meter = AverageMeter("Loss")
    metric_tracker = MetricTracker(config["evaluation"]["metrics"])
    modalities = config["dataset"]["modalities"]

    all_features = {m: [] for m in modalities}
    all_targets = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        inputs = {m: batch[m].to(device) for m in modalities}
        targets = batch["label"].to(device)

        logits, unimodal_logits, features = model(inputs, return_features=True)
        loss = nn.functional.cross_entropy(logits, targets)

        loss_meter.update(loss.item(), targets.size(0))
        metric_tracker.update(logits, targets)

        # Collect features for probe evaluation
        if probe_manager is not None:
            for m in modalities:
                all_features[m].append(features[m].cpu())
            all_targets.append(targets.cpu())

    metrics = metric_tracker.compute()
    metrics["loss"] = loss_meter.avg

    # Evaluate probes on full eval set
    if probe_manager is not None and all_targets:
        cat_features = {m: torch.cat(all_features[m], dim=0).to(device) for m in modalities}
        cat_targets = torch.cat(all_targets, dim=0).to(device)

        probe_results = probe_manager.evaluate_probes(cat_features, cat_targets)
        for m in modalities:
            metrics[f"probe_acc_{m}"] = probe_results[m]['accuracy']

        utilization_gap = probe_manager.compute_utilization_gap()
        if utilization_gap is not None:
            metrics["utilization_gap"] = utilization_gap

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train ASGML model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "frequency", "staleness", "adaptive"],
        help="Training mode",
    )
    parser.add_argument("--fixed-ratio", type=int, default=2, help="Fixed frequency ratio (for frequency mode)")
    parser.add_argument("--fixed-staleness", type=int, default=2, help="Fixed staleness τ (for staleness mode)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision training (AMP)")
    parser.add_argument(
        "--modality-lr",
        type=str,
        default=None,
        help="Per-modality LR scales as 'modality:scale,...' (e.g., 'audio:0.5,visual:1.0')"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Configure ASGML based on mode
    if args.mode == "baseline":
        config["asgml"]["enabled"] = False
    else:
        config["asgml"]["enabled"] = True

    # Override epochs if specified
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    # Parse per-modality LR scales
    modality_lr_scales = None
    if args.modality_lr:
        modality_lr_scales = {}
        for item in args.modality_lr.split(","):
            modality, scale = item.split(":")
            modality_lr_scales[modality.strip()] = float(scale)

    # Set seed
    set_seed(args.seed)

    # Setup output directory
    if args.resume:
        # When resuming, use the same directory as the checkpoint
        output_dir = Path(args.resume).parent
        exp_name = output_dir.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{config['dataset']['name']}_{args.mode}_{timestamp}"
        output_dir = Path(args.output_dir) / exp_name
        output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_copy = config.copy()
    config_copy["experiment"] = {
        "mode": args.mode,
        "fixed_ratio": args.fixed_ratio,
        "fixed_staleness": args.fixed_staleness,
        "seed": args.seed,
        "amp": args.amp,
        "modality_lr_scales": modality_lr_scales,
    }
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config_copy, f)

    # Setup logger
    logger = setup_logger("asgml", log_file=str(output_dir / "train.log"))
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {config}")

    # Setup tensorboard
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

    modalities = config["dataset"]["modalities"]

    # Create model
    model = MultimodalModel(
        modalities=modalities,
        num_classes=config["dataset"]["num_classes"],
        encoder_config={
            m: {"backbone": config["model"]["backbone"], "pretrained": config["model"]["pretrained"]}
            for m in modalities
        },
        fusion_type=config["model"]["fusion_type"],
        feature_dim=config["model"]["feature_dim"],
        fusion_dim=config["model"]["fusion_dim"],
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config, modality_lr_scales)
    lr_scheduler = get_scheduler(optimizer, config)

    # Create AMP scaler if enabled
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler('cuda') if use_amp else None
    if use_amp:
        logger.info("Mixed precision training (AMP) enabled")
    if modality_lr_scales:
        logger.info(f"Per-modality LR scales: {modality_lr_scales}")

    # Create ASGML components
    loss_fn = ASGMLLoss(
        modalities=modalities,
        gamma=config["asgml"]["gamma"],
    )

    # Configure ASGML scheduler based on mode
    if args.mode == "baseline":
        asgml_scheduler = ASGMLScheduler(
            modalities=modalities,
            mode="frequency",
            adaptation="fixed",
            fixed_ratio=1,  # Update every step = baseline
        )
    elif args.mode == "frequency":
        asgml_scheduler = ASGMLScheduler(
            modalities=modalities,
            mode="frequency",
            adaptation="fixed",
            fixed_ratio=args.fixed_ratio,
        )
    elif args.mode == "staleness":
        asgml_scheduler = ASGMLScheduler(
            modalities=modalities,
            mode="staleness",
            adaptation="fixed",
            fixed_staleness=args.fixed_staleness,
        )
    elif args.mode == "adaptive":
        asgml_scheduler = ASGMLScheduler(
            modalities=modalities,
            mode=config["asgml"].get("asgml_mode", "frequency"),
            adaptation="adaptive",
            tau_base=config["asgml"]["tau_base"],
            tau_min=config["asgml"]["tau_min"],
            tau_max=config["asgml"]["tau_max"],
            threshold_delta=config["asgml"].get("threshold_delta", 0.1),
            beta=config["asgml"]["beta"],
            lambda_comp=config["asgml"]["lambda_comp"],
            max_staleness_ratio=config["asgml"].get("max_staleness_ratio", 3.0),
        )

    # Create probe manager
    probe_manager = ProbeManager(
        modalities=modalities,
        feature_dim=config["model"]["feature_dim"],
        num_classes=config["dataset"]["num_classes"],
        probe_type=config["asgml"].get("probe_type", "linear"),
        probe_lr=config["asgml"].get("probe_lr", 1e-3),
        device=device,
        ema_alpha=config["asgml"].get("probe_ema_alpha", 0.1),  # EMA smoothing for stable adaptive control
    )

    # Set initial dominant modality (will be updated by probes)
    # For CREMA-D, audio is typically dominant
    if "audio" in modalities:
        asgml_scheduler.set_dominant_modality("audio")

    # Resume from checkpoint if specified
    start_epoch = 1
    best_acc = 0.0
    if args.resume:
        if Path(args.resume).exists():
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_acc = checkpoint.get("best_acc", 0.0)
            if "probe_manager_state" in checkpoint:
                probe_manager.load_state_dict(checkpoint["probe_manager_state"])
            if scaler is not None and "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            # Update LR scheduler to correct epoch
            if lr_scheduler is not None:
                for _ in range(checkpoint["epoch"]):
                    lr_scheduler.step()
            logger.info(f"Resumed from epoch {checkpoint['epoch']}, best_acc={best_acc:.4f}")
        else:
            logger.warning(f"Checkpoint not found: {args.resume}, starting from scratch")

    # Training loop
    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        # Reset ASGML state at start of epoch (optional - can also persist)
        # asgml_scheduler.reset()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, asgml_scheduler,
            probe_manager, device, epoch, config, logger, writer,
            scaler=scaler, use_amp=use_amp
        )

        # Evaluate
        test_metrics = evaluate(model, test_loader, device, config, probe_manager)

        # Update LR scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Log
        log_str = (
            f"Epoch {epoch}: "
            f"Train Loss={train_metrics['loss']:.4f}, "
            f"Train Acc={train_metrics['accuracy']:.4f}, "
            f"Test Acc={test_metrics['accuracy']:.4f}, "
            f"Test F1={test_metrics['f1_macro']:.4f}"
        )
        if "utilization_gap" in test_metrics:
            log_str += f", Util Gap={test_metrics['utilization_gap']:.4f}"
        logger.info(log_str)

        # Tensorboard logging
        writer.add_scalar("test/accuracy", test_metrics["accuracy"], epoch)
        writer.add_scalar("test/f1_macro", test_metrics["f1_macro"], epoch)
        writer.add_scalar("test/loss", test_metrics["loss"], epoch)
        if "utilization_gap" in test_metrics:
            writer.add_scalar("test/utilization_gap", test_metrics["utilization_gap"], epoch)
        for m in modalities:
            if f"probe_acc_{m}" in test_metrics:
                writer.add_scalar(f"test/probe_acc_{m}", test_metrics[f"probe_acc_{m}"], epoch)

        # Save checkpoint
        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
            best_checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "probe_manager_state": probe_manager.state_dict(),
                "best_acc": best_acc,
                "config": config_copy,
            }
            if scaler is not None:
                best_checkpoint["scaler_state_dict"] = scaler.state_dict()
            torch.save(best_checkpoint, output_dir / "best_model.pt")
            logger.info(f"New best model saved with accuracy: {best_acc:.4f}")

        # Periodic checkpoint
        if epoch % config["logging"]["save_interval"] == 0:
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "probe_manager_state": probe_manager.state_dict(),
                "best_acc": best_acc,
                "config": config_copy,
            }
            if scaler is not None:
                checkpoint_data["scaler_state_dict"] = scaler.state_dict()
            torch.save(checkpoint_data, output_dir / f"checkpoint_epoch{epoch}.pt")

    # Final summary
    logger.info(f"Training complete. Best accuracy: {best_acc:.4f}")
    writer.close()


if __name__ == "__main__":
    main()

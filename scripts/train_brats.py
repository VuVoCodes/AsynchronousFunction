#!/usr/bin/env python3
"""
BraTS 2021 training script with ASGML probe-guided boosting.

Adapts CGGM's DeepLab multi-input architecture for fair comparison.
Modes: baseline, ogm_ge, asgml_boost, cggm

Usage:
    python scripts/train_brats.py --mode baseline --seed 42
    python scripts/train_brats.py --mode asgml_boost --seed 42
    python scripts/train_brats.py --mode cggm --seed 42
"""

import os
import sys
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# Add project root and CGGM to path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'Papers', 'CGGM'))

from Papers.CGGM.datasets.BratsDataset import BraTSData
from Papers.CGGM.models.segmodel import DeepLabMultiInput, SegClassifier
from Papers.CGGM.src.eval_metrics import cosine_scheduler, cal_cos


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class DiceLoss(nn.Module):
    """Soft Dice Loss for segmentation."""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target.long(), self.n_classes).permute(0, 3, 1, 2).float()
        intersection = (pred * target_onehot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice = (2 * intersection + 1e-5) / (union + 1e-5)
        return 1 - dice.mean()


class SegLoss(nn.Module):
    """Combined Dice + CE loss matching CGGM."""
    def __init__(self, n_classes=4, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss(n_classes)

    def forward(self, pred, target):
        return self.ce(pred, target.long()) + self.dice(pred, target)


def cal_dice(pred, target):
    """Calculate Dice scores for WT, TC, ET."""
    pred = torch.argmax(pred, dim=1).cpu().numpy()
    target = target.cpu().numpy()

    # WT: labels 1,2,3 (whole tumor)
    wt_pred = (pred > 0).astype(np.float32)
    wt_true = (target > 0).astype(np.float32)
    wt_dice = (2 * (wt_pred * wt_true).sum() + 1e-5) / (wt_pred.sum() + wt_true.sum() + 1e-5)

    # TC: labels 1,3 (tumor core, note label 4->3 in dataset)
    tc_pred = ((pred == 1) | (pred == 3)).astype(np.float32)
    tc_true = ((target == 1) | (target == 3)).astype(np.float32)
    tc_dice = (2 * (tc_pred * tc_true).sum() + 1e-5) / (tc_pred.sum() + tc_true.sum() + 1e-5)

    # ET: label 3 (enhancing tumor, originally 4)
    et_pred = (pred == 3).astype(np.float32)
    et_true = (target == 3).astype(np.float32)
    et_dice = (2 * (et_pred * et_true).sum() + 1e-5) / (et_pred.sum() + et_true.sum() + 1e-5)

    return wt_dice, tc_dice, et_dice


class ProbeManager:
    """Lightweight probes for monitoring per-modality representation quality."""

    def __init__(self, feature_dim, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.modalities = ['flair', 't1ce', 't1', 't2']

        # Simple probes: global avg pool + linear classifier
        self.probes = {}
        self.probe_optims = {}
        for m in self.modalities:
            probe = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, num_classes),  # ASPP output is 256-dim
            ).to(device)
            self.probes[m] = probe
            self.probe_optims[m] = optim.Adam(probe.parameters(), lr=1e-3)

        self.prev_acc = {m: 0.0 for m in self.modalities}

    def train_and_eval(self, features, labels):
        """Train probes on detached features, return per-modality accuracy."""
        accs = {}
        # Downsample labels to match feature spatial size
        labels_down = F.interpolate(
            labels.float().unsqueeze(1), size=features[0].shape[2:], mode='nearest'
        ).squeeze(1).long()

        for i, m in enumerate(self.modalities):
            feat = features[i].detach()
            self.probe_optims[m].zero_grad()
            pred = self.probes[m](feat)

            # Use global label (majority class in the patch)
            global_label = (labels_down > 0).float().mean(dim=(1, 2))
            binary_label = (global_label > 0.1).long()

            loss = F.cross_entropy(pred, binary_label)
            loss.backward()
            self.probe_optims[m].step()

            with torch.no_grad():
                pred_cls = pred.argmax(dim=1)
                acc = (pred_cls == binary_label).float().mean().item()
                accs[m] = acc

        return accs

    def get_boost_scales(self, accs, alpha=0.5):
        """Compute boost scales: weaker modalities get higher scaling."""
        min_acc = min(accs.values())
        max_acc = max(accs.values())
        gap = max_acc - min_acc + 1e-8

        scales = {}
        for m in self.modalities:
            rel_weakness = 1.0 - (accs[m] - min_acc) / gap
            scales[m] = min(1.0 + alpha * rel_weakness, 2.0)
        return scales


def train_epoch(model, loader, optimizer, criterion, scheduler, epoch, device, args,
                classifier=None, cls_optimizer=None, probe_mgr=None):
    """Train one epoch."""
    model.train()
    total_loss = 0
    acc1 = [0] * 4
    l_gm = None

    for i_batch, batch in enumerate(loader):
        it = len(loader) * (epoch - 1) + i_batch
        optimizer.param_groups[0]['lr'] = scheduler[it]

        flair, t1ce, t1, t2, labels = [x.cuda() for x in batch]

        model.zero_grad()
        preds, hf, lf = model(flair, t1ce, t1, t2)

        # Main loss
        raw_loss = criterion(preds, labels)

        # CGGM: add L_gm from previous iteration
        if args.mode == 'cggm' and l_gm is not None:
            raw_loss = raw_loss + args.cggm_lamda * l_gm

        raw_loss.backward()

        # ========== Mode-specific gradient modulation ==========

        if args.mode == 'asgml_boost' and probe_mgr is not None:
            # Train probes and get boost scales
            accs = probe_mgr.train_and_eval(hf, labels)
            scales = probe_mgr.get_boost_scales(accs, alpha=args.boost_alpha)

            # Scale encoder gradients
            backbone_names = ['backbone1', 'backbone2', 'backbone3', 'backbone4']
            modality_names = ['flair', 't1ce', 't1', 't2']
            for bname, mname in zip(backbone_names, modality_names):
                for name, param in model.named_parameters():
                    if bname in name and param.grad is not None:
                        param.grad *= scales[mname]

        elif args.mode == 'cggm' and classifier is not None:
            cls_optimizer.zero_grad()
            cls_res = classifier(hf, lf)

            # Get fusion gradient
            fusion_grad = None
            for name, para in model.named_parameters():
                if 'decoder.last_conv.7.weight' in name:
                    fusion_grad = para
                    break

            cls_loss = sum(criterion(cls_res[i], labels) for i in range(4))
            cls_loss.backward()

            # Get classifier gradients
            cls_grad = []
            for name, para in classifier.named_parameters():
                if 'last_conv.7.weight' in name:
                    cls_grad.append(para)

            # Cosine similarity
            if fusion_grad is not None and len(cls_grad) == 4:
                llist = cal_cos(cls_grad, fusion_grad)

                # Accuracy-based coefficients
                acc2 = []
                for r in cls_res:
                    pred_cls = torch.argmax(r, dim=1)
                    acc = (pred_cls == labels).float().mean().item()
                    acc2.append(acc)

                diff = [acc2[i] - acc1[i] for i in range(4)]
                diff_sum = sum(diff) + 1e-8
                coeff = [(diff_sum - d) / diff_sum for d in diff]
                acc1 = acc2

                l_gm_val = sum(abs(coeff[i]) - coeff[i] * llist[i] for i in range(4)) / 4
                l_gm = l_gm_val

                # Scale encoder gradients
                backbone_names = ['backbone1', 'backbone2', 'backbone3', 'backbone4']
                for i, bname in enumerate(backbone_names):
                    for name, params in model.named_parameters():
                        if bname in name and params.grad is not None:
                            params.grad *= (coeff[i] * args.cggm_rou)

            cls_optimizer.step()

        optimizer.step()
        total_loss += raw_loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """Evaluate model, return loss and dice scores."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            flair, t1ce, t1, t2, labels = [x.cuda() for x in batch]
            preds, _, _ = model(flair, t1ce, t1, t2)
            total_loss += criterion(preds, labels).item()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    wt, tc, et = cal_dice(all_preds, all_labels)
    avg_dice = (wt + tc + et) / 3

    return total_loss / len(loader), wt, tc, et, avg_dice


def main():
    parser = argparse.ArgumentParser(description='BraTS training with ASGML')
    parser.add_argument('--mode', type=str, default='baseline',
                        choices=['baseline', 'asgml_boost', 'cggm'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--base-lr', type=float, default=0.01)
    parser.add_argument('--data-dir', type=str, default='data/BraTS/h5_data')
    parser.add_argument('--output-dir', type=str, default='outputs/sweep_brats')
    parser.add_argument('--exp-name', type=str, default=None)
    # ASGML
    parser.add_argument('--boost-alpha', type=float, default=0.5)
    # CGGM
    parser.add_argument('--cggm-rou', type=float, default=1.3)
    parser.add_argument('--cggm-lamda', type=float, default=0.2)
    parser.add_argument('--cggm-cls-lr', type=float, default=0.001)
    args = parser.parse_args()

    set_seed(args.seed)

    exp_name = args.exp_name or f'brats_{args.mode}_seed{args.seed}'
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / 'train.log'
    log_fh = open(log_file, 'w')

    def log(msg):
        print(msg, flush=True)
        log_fh.write(msg + '\n')
        log_fh.flush()

    log(f'Mode: {args.mode}, Seed: {args.seed}')
    log(f'Data: {args.data_dir}')

    # Data
    train_data = BraTSData(args.data_dir, 'train')
    valid_data = BraTSData(args.data_dir, 'valid')
    test_data = BraTSData(args.data_dir, 'test')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=4)

    log(f'Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}')

    # Model
    device = torch.device('cuda')
    model = DeepLabMultiInput(output_stride=16, num_classes=4).to(device)
    log(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    # Loss
    criterion = SegLoss(n_classes=4, weight=torch.tensor([0.2, 0.3, 0.25, 0.25]).cuda())

    # Optimizer + scheduler
    optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=1e-4, momentum=0.9)
    warmup_ep = min(5, args.epochs // 2)
    scheduler = cosine_scheduler(
        base_value=args.base_lr, final_value=1e-6,
        epochs=args.epochs, niter_per_ep=len(train_loader),
        warmup_epochs=warmup_ep, start_warmup_value=1e-4
    )

    # Mode-specific setup
    classifier, cls_optimizer, probe_mgr = None, None, None

    if args.mode == 'cggm':
        classifier = SegClassifier(num_classes=4).to(device)
        cls_optimizer = optim.SGD(classifier.parameters(), lr=args.cggm_cls_lr,
                                  weight_decay=1e-4, momentum=0.9)
        log(f'CGGM: rou={args.cggm_rou}, lamda={args.cggm_lamda}')

    elif args.mode == 'asgml_boost':
        probe_mgr = ProbeManager(256, 4, device)
        log(f'ASGML boost: alpha={args.boost_alpha}')

    # Training
    best_dice = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, epoch, device, args,
            classifier=classifier, cls_optimizer=cls_optimizer, probe_mgr=probe_mgr,
        )
        val_loss, wt, tc, et, avg_dice = evaluate(model, valid_loader, criterion, device)
        elapsed = time.time() - t0

        log(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Val Dice={avg_dice:.4f} '
            f'(WT={wt:.4f}, TC={tc:.4f}, ET={et:.4f}) [{elapsed:.1f}s]')

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            log(f'New best model saved with dice: {best_dice:.4f}')

    # Final test
    model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
    test_loss, wt, tc, et, avg_dice = evaluate(model, test_loader, criterion, device)
    log(f'Test: Dice={avg_dice:.4f} (WT={wt:.4f}, TC={tc:.4f}, ET={et:.4f})')
    log(f'Training complete. Best val dice: {best_dice:.4f}')

    log_fh.close()


if __name__ == '__main__':
    main()

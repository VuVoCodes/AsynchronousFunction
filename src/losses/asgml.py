"""
Asynchronous Staleness Guided Multimodal Learning (ASGML) Loss Function.

This module implements the core ASGML mechanism:
1. Monitor learning speed via gradient magnitude and loss descent rate
2. Compute adaptive staleness thresholds per modality
3. Determine which modalities update at each step
4. Apply gradient compensation for stale updates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import deque


class LearningSpeedTracker:
    """
    Tracks learning speed signals for each modality.

    Monitors:
    - Gradient magnitude ratios (instantaneous)
    - Loss descent rates (temporal, over window of k steps)
    """

    def __init__(
        self,
        modalities: List[str],
        window_size: int = 10,
        beta: float = 0.5,
    ):
        """
        Args:
            modalities: List of modality names
            window_size: Number of steps for loss descent calculation
            beta: Weight for combining gradient vs loss signals
        """
        self.modalities = modalities
        self.window_size = window_size
        self.beta = beta

        # Loss history for each modality
        self.loss_history: Dict[str, deque] = {
            m: deque(maxlen=window_size) for m in modalities
        }

        # Gradient norm history (for smoothing)
        self.grad_norm_history: Dict[str, deque] = {
            m: deque(maxlen=5) for m in modalities
        }

    def update(
        self,
        losses: Dict[str, float],
        grad_norms: Dict[str, float],
    ) -> None:
        """Update history with current step's losses and gradient norms."""
        for m in self.modalities:
            self.loss_history[m].append(losses.get(m, 0.0))
            self.grad_norm_history[m].append(grad_norms.get(m, 0.0))

    def compute_learning_speed(self) -> Dict[str, float]:
        """
        Compute learning speed score for each modality.

        Returns:
            Dictionary mapping modality names to learning speed scores.
            Score > 1 means faster than average, < 1 means slower.
        """
        # Compute gradient magnitude ratios
        grad_ratios = self._compute_gradient_ratios()

        # Compute loss descent ratios
        loss_ratios = self._compute_loss_descent_ratios()

        # Combine signals
        learning_speeds = {}
        for m in self.modalities:
            g_ratio = grad_ratios.get(m, 1.0)
            l_ratio = loss_ratios.get(m, 1.0)
            learning_speeds[m] = self.beta * g_ratio + (1 - self.beta) * l_ratio

        return learning_speeds

    def _compute_gradient_ratios(self) -> Dict[str, float]:
        """Compute gradient magnitude relative to mean."""
        # Get smoothed gradient norms
        grad_norms = {
            m: sum(self.grad_norm_history[m]) / max(len(self.grad_norm_history[m]), 1)
            for m in self.modalities
        }

        mean_norm = sum(grad_norms.values()) / max(len(grad_norms), 1)
        if mean_norm < 1e-8:
            return {m: 1.0 for m in self.modalities}

        return {m: norm / mean_norm for m, norm in grad_norms.items()}

    def _compute_loss_descent_ratios(self) -> Dict[str, float]:
        """Compute loss descent rate relative to mean."""
        descents = {}
        for m in self.modalities:
            history = list(self.loss_history[m])
            if len(history) < 2:
                descents[m] = 0.0
            else:
                # Loss descent = first - last (positive means decreasing loss)
                descents[m] = max(history[0] - history[-1], 0.0)

        mean_descent = sum(descents.values()) / max(len(descents), 1)
        if mean_descent < 1e-8:
            return {m: 1.0 for m in self.modalities}

        return {m: d / mean_descent for m, d in descents.items()}

    def reset(self) -> None:
        """Reset all history."""
        for m in self.modalities:
            self.loss_history[m].clear()
            self.grad_norm_history[m].clear()


class ASGMLLoss(nn.Module):
    """
    ASGML Loss Function with adaptive staleness control.

    Combines:
    - Multimodal fusion loss
    - Unimodal regularization losses (applied conditionally)
    - Adaptive staleness-based update decisions
    """

    def __init__(
        self,
        modalities: List[str],
        tau_base: float = 2.0,
        tau_min: float = 1.0,
        tau_max: float = 5.0,
        beta: float = 0.5,
        lambda_comp: float = 0.1,
        gamma: float = 1.0,
        window_size: int = 10,
    ):
        """
        Args:
            modalities: List of modality names
            tau_base: Baseline staleness threshold
            tau_min: Minimum staleness (1 = update every step)
            tau_max: Maximum staleness bound
            beta: Weight for gradient vs loss descent signals
            lambda_comp: Gradient compensation factor
            gamma: Unimodal regularization weight
            window_size: Window size for loss descent computation
        """
        super().__init__()
        self.modalities = modalities
        self.tau_base = tau_base
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.lambda_comp = lambda_comp
        self.gamma = gamma

        # Learning speed tracker
        self.tracker = LearningSpeedTracker(
            modalities=modalities,
            window_size=window_size,
            beta=beta,
        )

        # Current staleness values
        self.staleness: Dict[str, float] = {m: 1.0 for m in modalities}

        # Step counter
        self.current_step = 0

    def compute_staleness(self) -> Dict[str, float]:
        """Compute adaptive staleness thresholds based on learning speeds."""
        learning_speeds = self.tracker.compute_learning_speed()

        staleness = {}
        for m in self.modalities:
            # Higher learning speed -> higher staleness (fewer updates)
            raw_tau = self.tau_base * learning_speeds[m]
            staleness[m] = max(self.tau_min, min(self.tau_max, raw_tau))

        self.staleness = staleness
        return staleness

    def get_update_mask(self) -> Dict[str, bool]:
        """
        Determine which modalities should update at current step.

        Returns:
            Dictionary mapping modality names to update decisions.
        """
        update_mask = {}
        for m in self.modalities:
            tau = int(round(self.staleness[m]))
            update_mask[m] = (self.current_step % tau) == 0
        return update_mask

    def get_gradient_scales(self) -> Dict[str, float]:
        """
        Compute gradient scaling factors for staleness compensation.

        Returns:
            Dictionary mapping modality names to gradient scale factors.
        """
        scales = {}
        for m in self.modalities:
            scales[m] = 1.0 + self.lambda_comp * self.staleness[m]
        return scales

    def forward(
        self,
        fusion_logits: torch.Tensor,
        unimodal_logits: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        grad_norms: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, any]]:
        """
        Compute ASGML loss.

        Args:
            fusion_logits: Fused prediction logits (B, num_classes)
            unimodal_logits: Dict of unimodal logits per modality
            targets: Ground truth labels (B,)
            grad_norms: Optional gradient norms from previous step

        Returns:
            total_loss: Combined loss value
            info: Dictionary with debugging/logging information
        """
        # Compute fusion loss (always applied)
        fusion_loss = F.cross_entropy(fusion_logits, targets)

        # Compute unimodal losses
        unimodal_losses = {}
        for m in self.modalities:
            unimodal_losses[m] = F.cross_entropy(unimodal_logits[m], targets)

        # Update tracker with current losses
        loss_dict = {m: unimodal_losses[m].item() for m in self.modalities}
        if grad_norms is not None:
            self.tracker.update(loss_dict, grad_norms)

        # Compute adaptive staleness
        staleness = self.compute_staleness()

        # Get update mask
        update_mask = self.get_update_mask()

        # Compute total unimodal regularization (only for updating modalities)
        unimodal_reg = torch.tensor(0.0, device=fusion_logits.device)
        for m in self.modalities:
            if update_mask[m]:
                unimodal_reg = unimodal_reg + unimodal_losses[m]

        # Total loss
        total_loss = fusion_loss + self.gamma * unimodal_reg

        # Increment step
        self.current_step += 1

        # Gather info for logging
        info = {
            "fusion_loss": fusion_loss.item(),
            "unimodal_losses": {m: unimodal_losses[m].item() for m in self.modalities},
            "staleness": staleness,
            "update_mask": update_mask,
            "gradient_scales": self.get_gradient_scales(),
            "learning_speeds": self.tracker.compute_learning_speed(),
        }

        return total_loss, info

    def reset(self) -> None:
        """Reset loss function state (call at start of each epoch)."""
        self.current_step = 0
        self.staleness = {m: 1.0 for m in self.modalities}
        self.tracker.reset()

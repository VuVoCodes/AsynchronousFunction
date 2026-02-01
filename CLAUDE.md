# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeurIPS 2026 submission: **Asynchronous Staleness Guided Multimodal Learning (ASGML)**

A novel approach to balanced multimodal learning through adaptive asynchronous updates, where each modality has an independent update frequency determined by its learning dynamics. Unlike existing gradient modulation methods (OGM-GE, MMPareto, CGGM) that scale gradient magnitudes synchronously, ASGML introduces **temporal separation** — dominant modalities receive stale (delayed) gradient updates while weaker modalities update with fresh gradients every step.

## Repository Structure

```
├── paper/                  # NeurIPS LaTeX paper
│   ├── main.tex
│   └── sections/           # Introduction, method, experiments, etc.
├── src/
│   ├── models/             # Encoders (audio, visual, text) and fusion modules
│   ├── datasets/           # CREMA-D, AVE, Kinetics-Sounds loaders
│   ├── losses/             # ASGML loss function
│   └── utils/              # Metrics, logging
├── configs/                # YAML configs per dataset
├── scripts/                # Training scripts
├── docs/plans/             # Design documents
└── Papers/                 # Reference papers
```

## Build Commands

```bash
# Install dependencies
pip install -e .

# Train with ASGML on CREMA-D
python scripts/train.py --config configs/cremad.yaml

# Train baseline (no ASGML)
python scripts/train.py --config configs/cremad.yaml --no-asgml
```

## Key Files

- `src/losses/asgml.py` - Core ASGML loss with adaptive staleness
- `src/models/multimodal.py` - N-modality multimodal model
- `docs/plans/2026-02-01-ASGML-design.md` - Full method design document

---

## Research Context

### The Problem: Prime Learning Window Suppression

In late-fusion multimodal networks, the dominant modality converges faster during early training, pulling the shared fusion head toward a **unimodal saddle manifold**. This suppresses the weaker modality's gradient signal before it develops discriminative features. This critical early phase is the **Prime Learning Window** — once it closes, the weaker modality is permanently underutilized.

**Theoretical basis:**
- Huang et al. (ICML 2022): exact calculations for unimodal phase duration
- Zhang et al. (ICML 2024): unimodal phase as function of architecture + initialization
- Huang et al. (NeurIPS 2021): provable multimodal superiority under latent space assumptions

### The Solution: ASGML

True asynchronous updates (not just gradient reweighting) where:
- Fast-learning modalities get **higher staleness** (fewer updates, gradients computed at older parameters)
- Slow-learning modalities get **lower staleness** (more updates, fresh gradients)
- Staleness adapts based on **gradient magnitude ratio + loss descent rate**
- Independent **probe networks** monitor each modality's learning state without coupling to the training objective

### Critical Distinction: Staleness vs. Frequency

```
Reduced frequency (simpler, our starting point):
  θ_{t+1} = θ_t - η ∇_{θ_t} L    (correct gradients, applied less often)

True staleness (our full contribution):
  θ_{t+τ+1} = θ_{t+τ} - η ∇_{θ_t} L  (gradients from old params applied to new params)
```

Both are implemented. Frequency-based is the minimal intervention; true staleness adds implicit regularization that may further prevent dominant modality takeover.

---

## Experimental Protocol

### Phase 1: Baseline Reproduction + Probe Diagnostics (Week 1–2)

**Goal:** Reproduce OGM-GE numbers on CREMA-D. Validate that probe monitoring provides useful signal.

**Dataset:** CREMA-D (audio-visual emotion recognition)
- Audio is typically dominant, video is weaker
- Standard benchmark in modality imbalance literature
- Direct published baselines: OGM-GE (~72–75%), MMPareto (~75%), baseline (~66%)

**Architecture:** Standard late-fusion
- Visual encoder: ResNet18 (pretrained ImageNet)
- Audio encoder: ResNet18 or Transformer variant
- Fusion: concatenation → MLP → classification head

**Probe setup (CRITICAL — probes must NEVER backprop into encoders):**
```python
# Probes are completely decoupled from main training
with torch.no_grad():
    features_A = encoder_A(x_A).detach()
    features_B = encoder_B(x_B).detach()
probe_loss_A = probe_criterion(probe_A(features_A), y)  # only updates probe_A weights
probe_loss_B = probe_criterion(probe_B(features_B), y)  # only updates probe_B weights
```

**Diagnostic metrics to log every eval_freq steps:**
- Probe accuracy per modality (utilization signal)
- Gradient magnitude per encoder: ‖∇_{θ_A} L‖, ‖∇_{θ_B} L‖
- Gradient ratio ρ (as defined in OGM-GE)
- Per-modality loss contribution
- Main task accuracy

### Phase 2: Fixed Async / Staleness Experiments (Week 3)

```python
# Condition 1: Baseline — both modalities update every step
# Condition 2: Fixed async 2:1 — dominant updates every 2nd step
# Condition 3: Fixed async 1:2 — control (opposite direction)
# Condition 4: True staleness — dominant uses τ-step-old gradients
```

### Phase 3: Adaptive ASGML (Week 4–5)

Full method: probe-detected utilization gap drives staleness/frequency adaptation.

### Phase 4: Ablations + Additional Benchmarks (Week 6–8)

**Required ablation table:**

| Experiment                     | What It Tests                          |
|--------------------------------|----------------------------------------|
| Baseline (joint training)      | Lower bound                            |
| OGM-GE only                   | SOTA synchronous gradient modulation   |
| CGGM only                     | SOTA magnitude + direction modulation  |
| ASGML fixed frequency          | Simplest async intervention            |
| ASGML fixed staleness          | True staleness intervention            |
| ASGML adaptive (full method)   | Full contribution                      |
| OGM-GE + ASGML (combination)  | Complementarity of sync + async        |

**Benchmarks (in priority order):**
1. CREMA-D (audio-visual, 2 modalities) — primary
2. Kinetics-Sounds or AVE (audio-visual) — generalization
3. CMU-MOSEI (language-audio-visual, 3 modalities) — scaling to 3 modalities
4. Medical dataset (GI or cardiovascular) — domain application

**Metrics for every experiment:**
- Final accuracy (mean ± std over 5+ seeds)
- Utilization gap at convergence
- Convergence speed (epochs to 90% of final accuracy)
- Wall-clock time per epoch (overhead measurement)
- Probe accuracy trajectories (visualization)

### Phase 5: Paper Writing (Week 9–12)

---

## Must-Compare Baselines

| Method   | Citation              | Venue          | Mechanism                                       |
|----------|-----------------------|----------------|--------------------------------------------------|
| OGM-GE   | Peng et al.          | NeurIPS 2022   | Gradient magnitude modulation + Gaussian noise   |
| MMPareto | Wei & Hu             | ICML 2024      | Pareto optimization preserving SGD noise         |
| CGGM     | Guo et al.           | NeurIPS 2024   | Classifier-guided magnitude + direction control  |
| GradNorm | Chen et al.          | ICML 2018      | Gradient magnitude normalization across tasks    |
| PCGrad   | Yu et al.            | NeurIPS 2020   | Project conflicting gradients                    |
| DWA      | Liu et al.           | CVPR 2019      | Dynamic weight averaging via loss decrease rate  |
| G-Blend  | Wang et al.          | CVPR 2020      | Overfitting-to-generalization ratio weighting    |

**All existing methods are synchronous** — they modify gradients within each iteration but update all modalities at the same time. ASGML is the first to introduce temporal separation.

---

## Theoretical Framing

### Convergence Foundation

Build on Koloskova et al. (NeurIPS 2022 Oral):
- Async SGD converges at O(σ²ε⁻² + √(τ_max · τ_avg) ε⁻¹)
- Convergence depends on **average delay**, not maximum
- Our adaptation: "workers" = modality encoders, "delay" = inverse update frequency or staleness τ

### What We Need to Prove

1. **Convergence:** Frequency-differentiated multimodal training converges under standard smoothness + bounded variance assumptions
2. **Utilization improvement:** Matching update frequency to learning speed provably improves modality utilization vs. uniform updates
3. **Detection:** Minimum probe complexity for reliable suppression detection (likely: linear probes suffice)

### Where ASGML Sits in the Taxonomy

| Aspect              | Gradient Modulation (OGM-GE etc.) | ASGML                              |
|---------------------|------------------------------------|-------------------------------------|
| What changes        | Gradient magnitude/direction       | Update timing and gradient freshness|
| Temporal dynamics   | None (synchronous)                 | Modalities update at different times|
| Granularity         | Continuous scaling                 | Discrete frequency + staleness      |
| Monitoring          | Coupled to training objective      | Independent probes (decoupled)      |
| Composable with     | ASGML (complementary)              | OGM-GE, CGGM (complementary)       |

---

## Key Hyperparameters

| Parameter           | Description                                          | Search Range               |
|---------------------|------------------------------------------------------|----------------------------|
| `eval_freq`         | Probe evaluation interval (iterations)               | {50, 100, 200, 500}        |
| `probe_train_steps` | Steps to train probe per evaluation                  | {50, 100, 200}             |
| `threshold_delta`   | Utilization gap threshold triggering adaptation       | {0.05, 0.1, 0.15, 0.2}    |
| `staleness_tau`     | Fixed staleness delay (Phase 2)                      | {1, 2, 4, 8}               |
| `update_ratio_min`  | Minimum update probability for dominant modality      | {0.25, 0.5, 0.75}          |
| `probe_type`        | Probe architecture                                   | {linear, mlp_1layer}       |
| `k_ratio`           | Fixed async frequency ratio (Phase 2)                | {2:1, 3:1, 4:1}            |

---

## Coding Constraints for Claude Code

### Safety-Critical Rules
1. **Probes must NEVER backpropagate into encoders.** Always use `.detach()` on encoder features before passing to probes. Always use separate optimizers for probes.
2. **Staleness buffer management:** When implementing true staleness, store gradient snapshots explicitly. Do not rely on optimizer state — maintain a separate `staleness_buffer` dict keyed by modality.
3. **Fusion head always updates every step** regardless of modality-specific schedules.
4. **Random seed control:** All experiments must accept a seed parameter. Use `torch.manual_seed()`, `np.random.seed()`, `random.seed()`, and `torch.backends.cudnn.deterministic = True`.

### Design Principles
- Config-driven experiments: no hardcoded hyperparameters in training loops
- All training state (probe accuracies, gradient norms, update ratios, staleness values) logged to W&B every step
- Modular architecture: encoders, fusion heads, probes, and staleness schedulers are interchangeable
- Type hints on all function signatures
- NumPy-style docstrings on all public methods
- Research code — prioritize readability and correctness over performance optimization

### Testing Expectations
- Unit tests for staleness buffer correctness (gradients stored/retrieved at right steps)
- Unit tests confirming probes do not affect encoder gradients (check grad computation graph)
- Integration test: one full training epoch on small data subset completes without error

---

## Paper Outline

### 1. Introduction
- Multimodal data integration mirrors clinical practice (Krones et al., Information Fusion 2025)
- The modality imbalance problem: dominant modalities suppress weaker ones
- Gap: all existing methods operate synchronously — none explore temporal separation
- Contribution: ASGML introduces asynchronous staleness-guided updates with independent monitoring

### 2. Related Work
- 2.1 Multimodal fusion architectures (early, intermediate, late, mixed)
- 2.2 Modality imbalance & gradient dynamics (Prime Learning Window, saddle manifolds)
- 2.3 Gradient modulation methods (OGM-GE, MMPareto, CGGM, GradNorm, PCGrad)
- 2.4 Asynchronous optimization in distributed systems (Koloskova et al.)
- 2.5 Representation monitoring via probing (SimCLR, linear probing as evaluation)

### 3. Method
- 3.1 Problem formulation
- 3.2 Independent probe monitoring (architecture, protocol, what they measure)
- 3.3 Staleness-guided asynchronous training (update mechanism)
- 3.4 Adaptive staleness scheduling (probe signals → staleness adjustment)
- 3.5 Theoretical convergence analysis

### 4. Experiments
- 4.1 Datasets and setup
- 4.2 Baselines
- 4.3 Main results
- 4.4 Ablation studies
- 4.5 Analysis (probe trajectories, convergence, overhead, loss landscapes)

### 5. Discussion
- When ASGML helps most (larger modality gaps, more modalities)
- Complementarity with gradient modulation
- Limitations (probe overhead, threshold sensitivity)

### 6. Conclusion

---

## NeurIPS 2025 Positioning Context

Recent NeurIPS 2025 papers show the field shifting toward:
- Game-theoretic regularization (away from direct gradient modulation)
- Causal-aware modality valuation
- Sample-level dynamic balancing (per-sample, not per-epoch)
- Process supervision framing (relevant for reframing probe monitoring)

**All solutions remain synchronous.** The asynchronous direction is distinctly unoccupied, making ASGML's positioning strong.

---

## Context and Memory

**Researcher:** John, PhD student in computer science specializing in multimodal ML methodology.

**Core discipline:** CS methodology and algorithmic innovation (medical AI is testbed, not primary contribution).

**Hardware:** RTX 4090 (24GB VRAM), standard workstation.

**Target venues:** NeurIPS, ICML, ICLR (methods tracks).

**Timeline:** 10–12 weeks to submission-ready paper.

**Key insight from literature review:** Existing gradient modulation methods (OGM-GE achieving 20+ pp improvement on CREMA-D) all operate at the level of gradient magnitude/direction adjustment. ASGML operates at the level of update scheduling — a fundamentally different mechanism that is complementary, not competing.

Purpose & context
John is a PhD student in computer science specializing in multimodal machine learning methodologies. His research focuses on adaptive fusion mechanisms that can dynamically combine different data modalities (vision, text, audio, clinical data) rather than using fixed fusion strategies. While he uses medical AI applications—particularly gastrointestinal and cardiovascular domains—as testbeds for validation, his core contribution targets computer science methodology and algorithmic innovation rather than medical applications themselves.
John's research has evolved toward addressing the "Prime Learning Window" problem, where multimodal networks suppress weaker modalities during early training phases due to gradient dynamics in late-fusion architectures. This represents a shift from general adaptive fusion exploration to targeting a specific theoretical challenge with practical implications for multimodal system performance.
His work operates within the broader context of addressing critical gaps in multimodal AI, including architectural complexity issues, modality information gaps, and efficiency challenges. Success is measured by algorithmic contributions that advance the theoretical understanding of multimodal learning while demonstrating practical improvements on standard benchmarks.
Current state
John is actively developing research on asynchronous loss functions for multimodal learning, specifically exploring how different modalities can be updated at different frequencies while maintaining independent monitoring mechanisms. This work builds on gradient staleness concepts from distributed systems literature and aims to address scenarios where modalities have different computational requirements or update schedules.
His current technical focus involves understanding gradient dynamics in multimodal networks, including how gradient modulation fits into the data flow between gradient computation and parameter updates. He's working on theoretical convergence properties adapted from asynchronous SGD theory and designing probe networks for monitoring individual modality learning states.
The research has progressed from literature review and gap identification to concrete experimental design, with plans for implementation on datasets like CREMA-D using standard hardware (RTX 3090 level).
On the horizon
John has a concrete research roadmap targeting a submission-ready paper within 10-12 weeks. The immediate next steps involve implementing a minimal viable experiment combining asynchronous optimization with independent monitoring mechanisms, starting with frequency-based asynchronous training and expanding to more sophisticated approaches.
He's considering hybrid approaches that combine gradient modulation techniques (like OGM-GE) with asynchronous updates, potentially creating novel integration strategies for multimodal learning scenarios. The research pipeline includes both theoretical analysis of convergence properties and empirical validation across multiple benchmarks.
Future exploration includes extending beyond the current focus to broader adaptive fusion mechanisms and potentially developing unified frameworks for multimodal system optimization.
Key learnings & principles
John has identified that truly asynchronous multimodal optimization with independent update schedules remains significantly underexplored in the literature, despite mature work in gradient modulation and loss reweighting. This represents a genuine research gap with potential for novel contributions.
He's learned that the Prime Learning Window problem stems from gradient dynamics visiting saddle manifolds corresponding to unimodal solutions, with theoretical work providing exact calculations for unimodal phase duration. Solutions like OGM-GE, MMPareto, and CGGM can achieve performance improvements of 20+ percentage points on benchmarks.
A key insight is the distinction between methods that provide reduced update frequency versus true gradient staleness, which have different mathematical implications for multimodal learning systems. John recognizes the importance of combining theoretical understanding with practical implementation considerations.
Approach & patterns
John follows a systematic research methodology starting with comprehensive literature reviews to identify gaps, followed by theoretical analysis and practical implementation. He prefers well-cited technical analysis with clear algorithmic properties and theoretical foundations over application-specific results.
His approach emphasizes positioning new methods within broader taxonomies and understanding comparative advantages across different techniques. He values detailed comparative analysis including performance metrics, complexity trade-offs, and theoretical characterizations.
John's research pattern involves starting with minimal viable experiments and progressively expanding to more sophisticated approaches, maintaining focus on both theoretical contributions and empirical validation across standard benchmarks.
Tools & resources
John regularly references top-tier conferences including NeurIPS, ICML, ICLR, CVPR, and MICCAI for staying current with multimodal learning advances. He uses specialized datasets like CREMA-D, Kvasir/HyperKvasir, and REAL-Colon for experimental validation.
His technical toolkit includes gradient-based optimization methods, transformer architectures, and fusion mechanisms ranging from early/late fusion to more sophisticated adaptive approaches. He works with standard deep learning frameworks capable of implementing asynchronous training and gradient modulation techniques.
For literature research, he relies on comprehensive database searches across multiple venues simultaneously, combining venue-specific queries with dataset-specific searches to identify both domain-specific work and transferable methodological contributions.
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeurIPS 2026 submission: **Asynchronous Staleness Guided Multimodal Learning (ASGML)**

A novel approach to balanced multimodal learning through adaptive asynchronous updates, where each modality has an independent update frequency determined by its learning dynamics. Unlike existing gradient modulation methods (OGM-GE, MMPareto, CGGM) that scale gradient magnitudes synchronously, ASGML introduces **temporal separation** — dominant modalities receive stale (delayed) gradient updates while weaker modalities update with fresh gradients every step.

## Repository Structure

```
├── paper/                  # NeurIPS LaTeX paper
│   ├── main.tex
│   └── sections/           # Introduction, method, experiments, etc.
├── src/
│   ├── models/             # Encoders (audio, visual, text) and fusion modules
│   ├── datasets/           # CREMA-D, AVE, Kinetics-Sounds loaders
│   ├── losses/             # ASGML loss function
│   └── utils/              # Metrics, logging
├── configs/                # YAML configs per dataset
├── scripts/                # Training scripts
├── docs/plans/             # Design documents
└── Papers/                 # Reference papers
```

## Build Commands

```bash
# Install dependencies
pip install -e .

# Train with ASGML on CREMA-D
python scripts/train.py --config configs/cremad.yaml

# Train baseline (no ASGML)
python scripts/train.py --config configs/cremad.yaml --no-asgml
```

## Key Files

- `src/losses/asgml.py` - Core ASGML loss with adaptive staleness
- `src/models/multimodal.py` - N-modality multimodal model
- `docs/plans/2026-02-01-ASGML-design.md` - Full method design document

---

## Research Context

### The Problem: Prime Learning Window Suppression

In late-fusion multimodal networks, the dominant modality converges faster during early training, pulling the shared fusion head toward a **unimodal saddle manifold**. This suppresses the weaker modality's gradient signal before it develops discriminative features. This critical early phase is the **Prime Learning Window** — once it closes, the weaker modality is permanently underutilized.

**Theoretical basis:**
- Huang et al. (ICML 2022): exact calculations for unimodal phase duration
- Zhang et al. (ICML 2024): unimodal phase as function of architecture + initialization
- Huang et al. (NeurIPS 2021): provable multimodal superiority under latent space assumptions

### The Solution: ASGML

True asynchronous updates (not just gradient reweighting) where:
- Fast-learning modalities get **higher staleness** (fewer updates, gradients computed at older parameters)
- Slow-learning modalities get **lower staleness** (more updates, fresh gradients)
- Staleness adapts based on **gradient magnitude ratio + loss descent rate**
- Independent **probe networks** monitor each modality's learning state without coupling to the training objective

### Critical Distinction: Staleness vs. Frequency

```
Reduced frequency (simpler, our starting point):
  θ_{t+1} = θ_t - η ∇_{θ_t} L    (correct gradients, applied less often)

True staleness (our full contribution):
  θ_{t+τ+1} = θ_{t+τ} - η ∇_{θ_t} L  (gradients from old params applied to new params)
```

Both are implemented. Frequency-based is the minimal intervention; true staleness adds implicit regularization that may further prevent dominant modality takeover.

---

## Experimental Protocol

### Phase 1: Baseline Reproduction + Probe Diagnostics (Week 1–2)

**Goal:** Reproduce OGM-GE numbers on CREMA-D. Validate that probe monitoring provides useful signal.

**Dataset:** CREMA-D (audio-visual emotion recognition)
- Audio is typically dominant, video is weaker
- Standard benchmark in modality imbalance literature
- Direct published baselines: OGM-GE (~72–75%), MMPareto (~75%), baseline (~66%)

**Architecture:** Standard late-fusion
- Visual encoder: ResNet18 (pretrained ImageNet)
- Audio encoder: ResNet18 or Transformer variant
- Fusion: concatenation → MLP → classification head

**Probe setup (CRITICAL — probes must NEVER backprop into encoders):**
```python
# Probes are completely decoupled from main training
with torch.no_grad():
    features_A = encoder_A(x_A).detach()
    features_B = encoder_B(x_B).detach()
probe_loss_A = probe_criterion(probe_A(features_A), y)  # only updates probe_A weights
probe_loss_B = probe_criterion(probe_B(features_B), y)  # only updates probe_B weights
```

**Diagnostic metrics to log every eval_freq steps:**
- Probe accuracy per modality (utilization signal)
- Gradient magnitude per encoder: ‖∇_{θ_A} L‖, ‖∇_{θ_B} L‖
- Gradient ratio ρ (as defined in OGM-GE)
- Per-modality loss contribution
- Main task accuracy

### Phase 2: Fixed Async / Staleness Experiments (Week 3)

```python
# Condition 1: Baseline — both modalities update every step
# Condition 2: Fixed async 2:1 — dominant updates every 2nd step
# Condition 3: Fixed async 1:2 — control (opposite direction)
# Condition 4: True staleness — dominant uses τ-step-old gradients
```

### Phase 3: Adaptive ASGML (Week 4–5)

Full method: probe-detected utilization gap drives staleness/frequency adaptation.

### Phase 4: Ablations + Additional Benchmarks (Week 6–8)

**Required ablation table:**

| Experiment                     | What It Tests                          |
|--------------------------------|----------------------------------------|
| Baseline (joint training)      | Lower bound                            |
| OGM-GE only                   | SOTA synchronous gradient modulation   |
| CGGM only                     | SOTA magnitude + direction modulation  |
| ASGML fixed frequency          | Simplest async intervention            |
| ASGML fixed staleness          | True staleness intervention            |
| ASGML adaptive (full method)   | Full contribution                      |
| OGM-GE + ASGML (combination)  | Complementarity of sync + async        |

**Benchmarks (in priority order):**
1. CREMA-D (audio-visual, 2 modalities) — primary
2. Kinetics-Sounds or AVE (audio-visual) — generalization
3. CMU-MOSEI (language-audio-visual, 3 modalities) — scaling to 3 modalities
4. Medical dataset (GI or cardiovascular) — domain application

**Metrics for every experiment:**
- Final accuracy (mean ± std over 5+ seeds)
- Utilization gap at convergence
- Convergence speed (epochs to 90% of final accuracy)
- Wall-clock time per epoch (overhead measurement)
- Probe accuracy trajectories (visualization)

### Phase 5: Paper Writing (Week 9–12)

---

## Must-Compare Baselines

| Method   | Citation              | Venue          | Mechanism                                       |
|----------|-----------------------|----------------|--------------------------------------------------|
| OGM-GE   | Peng et al.          | NeurIPS 2022   | Gradient magnitude modulation + Gaussian noise   |
| MMPareto | Wei & Hu             | ICML 2024      | Pareto optimization preserving SGD noise         |
| CGGM     | Guo et al.           | NeurIPS 2024   | Classifier-guided magnitude + direction control  |
| GradNorm | Chen et al.          | ICML 2018      | Gradient magnitude normalization across tasks    |
| PCGrad   | Yu et al.            | NeurIPS 2020   | Project conflicting gradients                    |
| DWA      | Liu et al.           | CVPR 2019      | Dynamic weight averaging via loss decrease rate  |
| G-Blend  | Wang et al.          | CVPR 2020      | Overfitting-to-generalization ratio weighting    |

**All existing methods are synchronous** — they modify gradients within each iteration but update all modalities at the same time. ASGML is the first to introduce temporal separation.

---

## Theoretical Framing

### Convergence Foundation

Build on Koloskova et al. (NeurIPS 2022 Oral):
- Async SGD converges at O(σ²ε⁻² + √(τ_max · τ_avg) ε⁻¹)
- Convergence depends on **average delay**, not maximum
- Our adaptation: "workers" = modality encoders, "delay" = inverse update frequency or staleness τ

### What We Need to Prove

1. **Convergence:** Frequency-differentiated multimodal training converges under standard smoothness + bounded variance assumptions
2. **Utilization improvement:** Matching update frequency to learning speed provably improves modality utilization vs. uniform updates
3. **Detection:** Minimum probe complexity for reliable suppression detection (likely: linear probes suffice)

### Where ASGML Sits in the Taxonomy

| Aspect              | Gradient Modulation (OGM-GE etc.) | ASGML                              |
|---------------------|------------------------------------|-------------------------------------|
| What changes        | Gradient magnitude/direction       | Update timing and gradient freshness|
| Temporal dynamics   | None (synchronous)                 | Modalities update at different times|
| Granularity         | Continuous scaling                 | Discrete frequency + staleness      |
| Monitoring          | Coupled to training objective      | Independent probes (decoupled)      |
| Composable with     | ASGML (complementary)              | OGM-GE, CGGM (complementary)       |

---

## Key Hyperparameters

| Parameter           | Description                                          | Search Range               |
|---------------------|------------------------------------------------------|----------------------------|
| `eval_freq`         | Probe evaluation interval (iterations)               | {50, 100, 200, 500}        |
| `probe_train_steps` | Steps to train probe per evaluation                  | {50, 100, 200}             |
| `threshold_delta`   | Utilization gap threshold triggering adaptation       | {0.05, 0.1, 0.15, 0.2}    |
| `staleness_tau`     | Fixed staleness delay (Phase 2)                      | {1, 2, 4, 8}               |
| `update_ratio_min`  | Minimum update probability for dominant modality      | {0.25, 0.5, 0.75}          |
| `probe_type`        | Probe architecture                                   | {linear, mlp_1layer}       |
| `k_ratio`           | Fixed async frequency ratio (Phase 2)                | {2:1, 3:1, 4:1}            |

---

## Coding Constraints for Claude Code

### Safety-Critical Rules
1. **Probes must NEVER backpropagate into encoders.** Always use `.detach()` on encoder features before passing to probes. Always use separate optimizers for probes.
2. **Staleness buffer management:** When implementing true staleness, store gradient snapshots explicitly. Do not rely on optimizer state — maintain a separate `staleness_buffer` dict keyed by modality.
3. **Fusion head always updates every step** regardless of modality-specific schedules.
4. **Random seed control:** All experiments must accept a seed parameter. Use `torch.manual_seed()`, `np.random.seed()`, `random.seed()`, and `torch.backends.cudnn.deterministic = True`.

### Design Principles
- Config-driven experiments: no hardcoded hyperparameters in training loops
- All training state (probe accuracies, gradient norms, update ratios, staleness values) logged to W&B every step
- Modular architecture: encoders, fusion heads, probes, and staleness schedulers are interchangeable
- Type hints on all function signatures
- NumPy-style docstrings on all public methods
- Research code — prioritize readability and correctness over performance optimization

### Testing Expectations
- Unit tests for staleness buffer correctness (gradients stored/retrieved at right steps)
- Unit tests confirming probes do not affect encoder gradients (check grad computation graph)
- Integration test: one full training epoch on small data subset completes without error

---

## Paper Outline

### 1. Introduction
- Multimodal data integration mirrors clinical practice (Krones et al., Information Fusion 2025)
- The modality imbalance problem: dominant modalities suppress weaker ones
- Gap: all existing methods operate synchronously — none explore temporal separation
- Contribution: ASGML introduces asynchronous staleness-guided updates with independent monitoring

### 2. Related Work
- 2.1 Multimodal fusion architectures (early, intermediate, late, mixed)
- 2.2 Modality imbalance & gradient dynamics (Prime Learning Window, saddle manifolds)
- 2.3 Gradient modulation methods (OGM-GE, MMPareto, CGGM, GradNorm, PCGrad)
- 2.4 Asynchronous optimization in distributed systems (Koloskova et al.)
- 2.5 Representation monitoring via probing (SimCLR, linear probing as evaluation)

### 3. Method
- 3.1 Problem formulation
- 3.2 Independent probe monitoring (architecture, protocol, what they measure)
- 3.3 Staleness-guided asynchronous training (update mechanism)
- 3.4 Adaptive staleness scheduling (probe signals → staleness adjustment)
- 3.5 Theoretical convergence analysis

### 4. Experiments
- 4.1 Datasets and setup
- 4.2 Baselines
- 4.3 Main results
- 4.4 Ablation studies
- 4.5 Analysis (probe trajectories, convergence, overhead, loss landscapes)

### 5. Discussion
- When ASGML helps most (larger modality gaps, more modalities)
- Complementarity with gradient modulation
- Limitations (probe overhead, threshold sensitivity)

### 6. Conclusion

---

## NeurIPS 2025 Positioning Context

Recent NeurIPS 2025 papers show the field shifting toward:
- Game-theoretic regularization (away from direct gradient modulation)
- Causal-aware modality valuation
- Sample-level dynamic balancing (per-sample, not per-epoch)
- Process supervision framing (relevant for reframing probe monitoring)

**All solutions remain synchronous.** The asynchronous direction is distinctly unoccupied, making ASGML's positioning strong.

---

## Context and Memory

**Researcher:** John, PhD student in computer science specializing in multimodal ML methodology.

**Core discipline:** CS methodology and algorithmic innovation (medical AI is testbed, not primary contribution).

**Hardware:** RTX 4090 (24GB VRAM), standard workstation.

**Target venues:** NeurIPS, ICML, ICLR (methods tracks).

**Timeline:** 10–12 weeks to submission-ready paper.

**Key insight from literature review:** Existing gradient modulation methods (OGM-GE achieving 20+ pp improvement on CREMA-D) all operate at the level of gradient magnitude/direction adjustment. ASGML operates at the level of update scheduling — a fundamentally different mechanism that is complementary, not competing.

Purpose & context
John is a PhD student in computer science specializing in multimodal machine learning methodologies. His research focuses on adaptive fusion mechanisms that can dynamically combine different data modalities (vision, text, audio, clinical data) rather than using fixed fusion strategies. While he uses medical AI applications—particularly gastrointestinal and cardiovascular domains—as testbeds for validation, his core contribution targets computer science methodology and algorithmic innovation rather than medical applications themselves.
John's research has evolved toward addressing the "Prime Learning Window" problem, where multimodal networks suppress weaker modalities during early training phases due to gradient dynamics in late-fusion architectures. This represents a shift from general adaptive fusion exploration to targeting a specific theoretical challenge with practical implications for multimodal system performance.
His work operates within the broader context of addressing critical gaps in multimodal AI, including architectural complexity issues, modality information gaps, and efficiency challenges. Success is measured by algorithmic contributions that advance the theoretical understanding of multimodal learning while demonstrating practical improvements on standard benchmarks.
Current state
John is actively developing research on asynchronous loss functions for multimodal learning, specifically exploring how different modalities can be updated at different frequencies while maintaining independent monitoring mechanisms. This work builds on gradient staleness concepts from distributed systems literature and aims to address scenarios where modalities have different computational requirements or update schedules.
His current technical focus involves understanding gradient dynamics in multimodal networks, including how gradient modulation fits into the data flow between gradient computation and parameter updates. He's working on theoretical convergence properties adapted from asynchronous SGD theory and designing probe networks for monitoring individual modality learning states.
The research has progressed from literature review and gap identification to concrete experimental design, with plans for implementation on datasets like CREMA-D using standard hardware (RTX 3090 level).
On the horizon
John has a concrete research roadmap targeting a submission-ready paper within 10-12 weeks. The immediate next steps involve implementing a minimal viable experiment combining asynchronous optimization with independent monitoring mechanisms, starting with frequency-based asynchronous training and expanding to more sophisticated approaches.
He's considering hybrid approaches that combine gradient modulation techniques (like OGM-GE) with asynchronous updates, potentially creating novel integration strategies for multimodal learning scenarios. The research pipeline includes both theoretical analysis of convergence properties and empirical validation across multiple benchmarks.
Future exploration includes extending beyond the current focus to broader adaptive fusion mechanisms and potentially developing unified frameworks for multimodal system optimization.
Key learnings & principles
John has identified that truly asynchronous multimodal optimization with independent update schedules remains significantly underexplored in the literature, despite mature work in gradient modulation and loss reweighting. This represents a genuine research gap with potential for novel contributions.
He's learned that the Prime Learning Window problem stems from gradient dynamics visiting saddle manifolds corresponding to unimodal solutions, with theoretical work providing exact calculations for unimodal phase duration. Solutions like OGM-GE, MMPareto, and CGGM can achieve performance improvements of 20+ percentage points on benchmarks.
A key insight is the distinction between methods that provide reduced update frequency versus true gradient staleness, which have different mathematical implications for multimodal learning systems. John recognizes the importance of combining theoretical understanding with practical implementation considerations.
Approach & patterns
John follows a systematic research methodology starting with comprehensive literature reviews to identify gaps, followed by theoretical analysis and practical implementation. He prefers well-cited technical analysis with clear algorithmic properties and theoretical foundations over application-specific results.
His approach emphasizes positioning new methods within broader taxonomies and understanding comparative advantages across different techniques. He values detailed comparative analysis including performance metrics, complexity trade-offs, and theoretical characterizations.
John's research pattern involves starting with minimal viable experiments and progressively expanding to more sophisticated approaches, maintaining focus on both theoretical contributions and empirical validation across standard benchmarks.
Tools & resources
John regularly references top-tier conferences including NeurIPS, ICML, ICLR, CVPR, and MICCAI for staying current with multimodal learning advances. He uses specialized datasets like CREMA-D, Kvasir/HyperKvasir, and REAL-Colon for experimental validation.
His technical toolkit includes gradient-based optimization methods, transformer architectures, and fusion mechanisms ranging from early/late fusion to more sophisticated adaptive approaches. He works with standard deep learning frameworks capable of implementing asynchronous training and gradient modulation techniques.
For literature research, he relies on comprehensive database searches across multiple venues simultaneously, combining venue-specific queries with dataset-specific searches to identify both domain-specific work and transferable methodological contributions.

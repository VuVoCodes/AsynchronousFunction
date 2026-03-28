# Probe-guided gradient boosting for multimodal learning: related work and NeurIPS 2026 viability

**ASGML v2 occupies a defensible but narrow novelty gap in an increasingly crowded field.** The core contribution — boost-weak-only gradient scaling guided by decoupled linear probes with split-batch evaluation — is genuinely novel as a unified mechanism. However, the field has exploded since 2024: at least **34 closely related papers** now exist, with CGGM (NeurIPS 2024), IPRM (IJCAI 2025), and AUG (NeurIPS 2025) each sharing significant conceptual overlap. The strongest argument for ASGML v2 is the **OGM\* failure narrative** — that boost-weak fails with coupled signals but succeeds with decoupled probes — but this requires deeper theoretical backing than currently provided. NeurIPS 2026 acceptance is achievable but demands stronger theory, more benchmarks, and sharper positioning against CGGM and IPRM.

---

## Complete inventory of related work (34 papers)

The table below covers every closely related paper found, organized by publication year. All URLs have been verified.

### Foundational methods (2020–2022)

| Paper | Venue | Core mechanism | URL |
|-------|-------|---------------|-----|
| **G-Blend** (Wang et al.) | CVPR 2020 | Overfitting-to-generalization ratio for gradient blending | https://arxiv.org/abs/1905.12681 |
| **Huang et al.** | NeurIPS 2021 | Provable multimodal superiority over unimodal | https://arxiv.org/abs/2106.04538 |
| **OGM-GE** (Peng et al.) | CVPR 2022 (Oral) | Throttle dominant modality gradients + Gaussian noise enhancement | https://arxiv.org/abs/2203.15332 |
| **Modality Competition** (Huang et al.) | ICML 2022 (Spotlight) | First theoretical proof of why joint training fails — only a subset of modalities are learned | https://arxiv.org/abs/2203.12221 |
| **MSLR** (Yao & Mihalcea) | ACL Findings 2022 | Modality-specific learning rates for late-fusion models | https://aclanthology.org/2022.findings-acl.143/ |
| **Greedy Modality Selection** (Cheng et al.) | UAI 2022 / JMLR 2024 | Submodular maximization for modality selection | https://arxiv.org/abs/2210.12562 |

### Growth period (2023–2024)

| Paper | Venue | Core mechanism | URL |
|-------|-------|---------------|-----|
| **PMR** (Fan et al.) | CVPR 2023 | Prototype-based modal rebalance with decoupled unimodal evaluation | https://arxiv.org/abs/2211.07089 |
| **AGM** (Li et al.) | ICCV 2023 | Shapley-value gradient modulation — both boost and throttle | https://arxiv.org/abs/2308.07686 |
| **Du et al.** | ICML 2023 | Theoretical analysis of unimodal feature suppression in joint training | Referenced in proceedings; PMLR pp. 8632–8656 |
| **OPM/OGM\*** (Wei et al.) | TPAMI 2024 | Extended OGM-GE with feed-forward modulation (OPM); **OGM\* boost-weak variant shown to fail** | https://arxiv.org/abs/2410.11582 |
| **CGGM** (Guo et al.) | NeurIPS 2024 | Coupled auxiliary classifiers for gradient magnitude + direction modulation | https://arxiv.org/abs/2411.01409 |
| **MMPareto** (Wei & Hu) | ICML 2024 | Pareto optimization resolving gradient conflicts between unimodal/multimodal objectives | https://arxiv.org/abs/2405.17730 |
| **ReconBoost** (Hua et al.) | ICML 2024 | Modality-alternating learning with KL-based reconcilement regularization | https://arxiv.org/abs/2405.09321 |
| **MLA** (Zhang et al.) | CVPR 2024 | Alternating unimodal adaptation with entropy-based dynamic fusion | https://arxiv.org/abs/2311.10707 |
| **D&R** (Wei et al.) | ECCV 2024 | Representation separability diagnosis + soft encoder reinitialization | https://arxiv.org/abs/2407.09705 |
| **Unimodal Bias Theory** (Zhang et al.) | ICML 2024 | Architecture-dependent unimodal phase duration in deep linear networks | https://arxiv.org/abs/2312.00935 |
| **MLGM** (Kontras et al.) | BMVC 2024 | Multi-loss gradient modulation with acceleration + deceleration | https://arxiv.org/abs/2405.07930 |
| **DI-MML** (Fan et al.) | ACM MM 2024 | Detached training with unidirectional contrastive knowledge transfer | https://arxiv.org/abs/2407.19514 |

### Current wave (2025–2026)

| Paper | Venue | Core mechanism | URL |
|-------|-------|---------------|-----|
| **ARM** (Gao et al.) | AAAI 2025 | Conditional mutual information for asymmetric reinforcement | https://arxiv.org/abs/2501.01240 |
| **InfoReg/IAR** (Huang et al.) | CVPR 2025 | Fisher Information regulation during prime learning window | https://arxiv.org/abs/2503.18595 |
| **ARL** (Wei et al.) | ICCV 2025 | Bias-variance framework; inverse-variance weighting for asymmetric regulation | https://arxiv.org/abs/2507.10203 |
| **G2D** (Rakib et al.) | ICCV 2025 | Gradient-guided distillation from unimodal teachers | https://arxiv.org/abs/2506.21514 |
| **IPRM** (Yang et al.) | IJCAI 2025 | **Two-pass probe-and-rebalance: probe imbalance first, then learn under corrected balance** | https://www.ijcai.org/proceedings/2025/0395.pdf |
| **AUG** (Jiang et al.) | NeurIPS 2025 | **Gradient boosting with configurable classifiers + adaptive classifier assignment to weak modality** | https://arxiv.org/abs/2502.20120 |
| **CMoB** | NeurIPS 2025 | Causal modality valuation via Shannon benefit function | https://openreview.net/forum?id=ygHWfrwFmO |
| **Data Remixing** (Ma et al.) | ICML 2025 | Data decoupling and reassembly based on unimodal separability | https://arxiv.org/abs/2506.11550 |
| **MILES** (Guerra-Manzanares & Shamout) | IJCNN 2025 | Learning rate scheduling based on conditional utilization rate differences | https://arxiv.org/abs/2510.17394 |
| **AIM** (Shen et al.) | arXiv 2025 | Auxiliary blocks from under-optimized parameters + depth-adaptive modulation | https://arxiv.org/abs/2508.19769 |
| **GOAL** | Under review (ICLR 2026) | Gradient orthogonalization (PCGrad-style) + entropy-based adaptive learning | https://openreview.net/pdf?id=I3uFqoUZ2Y |
| **M-SAM** (Nowdeh et al.) | arXiv 2025 | SAM-based gradient modulation with Shapley dominance detection | https://arxiv.org/abs/2510.24919 |
| **TCMax** (Wu et al.) | ICLR 2026 | Total correlation maximization between multimodal features and labels | https://arxiv.org/abs/2602.13015 |
| **GMML** | ACM MM 2025 | Imbalance-aware gradient modulation with ℓ₂-norm constraints and convergence guarantees | https://dl.acm.org/doi/10.1145/3746027.3755198 |
| **DynCIM** | arXiv 2025 | Dynamic curriculum with sample-level and modality-level scheduling | https://arxiv.org/abs/2503.06456 |

---

## Three papers that directly threaten novelty

The most important positioning challenge for ASGML v2 comes from three specific papers that share significant conceptual machinery.

**CGGM (NeurIPS 2024) is the closest prior art.** CGGM introduces auxiliary modality-specific classifiers (1–2 MSA layers + FC) that evaluate per-modality utilization and compute unimodal gradient directions. Critically, CGGM's classifiers are **coupled** — they participate in the loss via an alignment term Lₘₘ and receive gradients through the encoder. CGGM modulates both gradient magnitude (via a ratio of inter-modality improvement rates: B_mᵢ = ρ · Σₖ≠ᵢΔεₖ / ΣₖΔεₖ) and gradient direction. It does **both boosting and throttling** through a single scaling factor. CGGM was evaluated on UPMC-Food 101, CMU-MOSI, IEMOCAP, and BraTS 2021 — notably **not** on CREMA-D, AVE, or Kinetics-Sounds, which limits direct comparison with ASGML v2. The paper was accepted as a NeurIPS 2024 poster with improvements ranging from +0.07pp to +4.64pp across datasets.

**IPRM (IJCAI 2025) shares the probing concept.** IPRM performs a **two-pass forward** strategy: the first pass probes modality strength via KL divergence between unimodal and fused predictions (no learning occurs), then recalibrates fusion weights via geodesic mixup on a hypersphere, and only then performs learning under corrected balance in the second pass. While IPRM's "probe then rebalance" philosophy overlaps with ASGML v2's probe-guided approach, the mechanisms differ substantially — IPRM operates at the fusion level (adjusting modality weights in prediction), while ASGML v2 operates at the gradient level (scaling encoder gradients). IPRM also does **not** use detached probes; it probes using the live model's predictions.

**AUG (NeurIPS 2025) uses gradient boosting explicitly.** AUG frames multimodal learning through the lens of Friedman's gradient boosting, adding configurable classifiers iteratively to minimize residual errors. It uses OGM-style monitoring to detect the weak modality and dynamically assigns more classifiers to it via an Adaptive Classifier Assignment strategy. AUG includes convergence analysis of its cross-modal gap function. The key difference from ASGML v2 is that AUG adds entire classifier modules rather than simply scaling gradients, and its "boosting" is literal ensemble boosting rather than gradient magnitude scaling.

---

## The OGM\* failure is ASGML v2's strongest differentiator

The TPAMI 2024 paper explicitly tested **OGM\***, a variant that boosts only the weak modality's gradient rather than throttling the dominant one. OGM\* failed to reduce the discrepancy ratio as effectively as standard OGM. The paper's analysis reveals why: in late-fusion networks, the gradient for all modalities shares the same loss derivative term ∂ℓ/∂f(xᵢ), which is **dominated by the strong modality**. When the dominant modality controls this shared term, boosting the weak modality simply amplifies a corrupted signal.

ASGML v2's argument that **decoupled probes solve this problem** is the paper's most compelling contribution. With `.detach()` preventing gradient flow from probe to encoder, the probe provides a genuinely independent measurement of encoder quality — one that cannot be "fooled" by dominant modality dynamics. The split-batch evaluation (train probes on first half, evaluate on second half) further ensures the probe signal is not overfit. This creates a clean chain of reasoning: accurate measurement → reliable weakness detection → effective boosting.

However, this argument currently lacks formal theoretical backing. A proof or formal analysis showing that decoupled measurement yields unbiased weakness estimates, while coupled measurement yields biased ones, would dramatically strengthen the paper.

---

## NeurIPS 2026 viability: achievable but requires strengthening

**The acceptance bar has risen significantly.** NeurIPS 2024 accepted CGGM as a poster with moderate improvements (+0.07pp to +4.64pp) across 4 diverse benchmarks spanning classification, regression, and segmentation. By NeurIPS 2025, papers like AUG included convergence analysis and CMoB brought causal/information-theoretic frameworks. ICML 2024 papers (MMPareto, ReconBoost) featured **6 benchmarks** and formal theoretical contributions. The trend through 2025–2026 is unmistakable: reviewers increasingly expect mathematical rigor alongside empirical validation.

**Results analysis.** ASGML v2's **+9.86pp on CREMA-D** is genuinely strong — among the largest single-dataset improvements in the field. But **+0.12pp on Kinetics-Sounds** is essentially noise-level. The adaptive behavior finding (boost-only outperforms boost+OGM on low-imbalance datasets) is interesting but needs stronger framing. For context, Data Remixing (ICML 2025) achieves +6.50pp on CREMA-D and +3.41pp on Kinetics-Sounds, and CGGM was accepted with even smaller improvements but across more diverse task types.

**What would push ASGML v2 over the line:**

- **Add theory.** At minimum, formalize why decoupled probes yield unbiased weakness estimates. Ideally, prove convergence properties of the boost-only scheme and characterize when boosting outperforms throttling. The field's theory bar is now set by Zhang et al. (ICML 2024) on unimodal phase duration and AUG's convergence analysis.

- **Expand benchmarks to 6+.** Add at least two more: BraTS 2021 (medical segmentation, 4 modalities) and either CMU-MOSI/IEMOCAP (sentiment/emotion, 3 modalities with text) or VGGSound (large-scale audio-visual). Diverse task types (not just classification) strongly influenced CGGM's acceptance. The current 4-benchmark setup is at the minimum threshold.

- **Sharpen positioning against CGGM and IPRM.** Run CGGM as a baseline on CREMA-D/AVE/KS (CGGM never reported these). Show that ASGML v2's decoupled probes give more accurate weakness measurements than CGGM's coupled classifiers (e.g., measure probe accuracy vs. CGGM classifier accuracy at predicting true unimodal performance). Against IPRM, emphasize the difference between fusion-level rebalancing (IPRM) and gradient-level boosting (ASGML v2).

- **Strengthen the composability story.** The claim that ASGML v2 is explicitly complementary with OGM-GE (throttle + boost = two-sided pressure) is unique in the field. Demonstrate this systematically: show ASGML v2 + OGM-GE > either alone on high-imbalance, and ASGML v2 alone > combined on low-imbalance. Make composability a first-class contribution with a table showing ASGML v2 plugged into multiple existing methods.

- **Frame the KS +0.12pp result positively.** The low-imbalance finding is actually a feature — the method's adaptive behavior means it automatically applies minimal intervention when imbalance is low. Frame this as graceful degradation / automatic calibration rather than a weak result.

**Scoring estimate.** In its current form, ASGML v2 would likely receive NeurIPS reviewer scores of **5.0–5.5** (borderline reject to borderline accept), insufficient for the poster threshold of ~5.8. With the improvements above — particularly theory and expanded benchmarks — scores of **6.0–6.5** (weak accept) are realistic. The paper's narrative strength (OGM\* failure → decoupled probes → boost-weak works) is genuinely compelling and could be a "story" that resonates with reviewers if properly supported.

---

## What a competitive NeurIPS 2026 submission looks like

Based on accepted papers from 2024–2025, a competitive submission in this area needs five elements: **(1)** a clearly articulated insight that prior work missed, **(2)** at least one formal theorem or provable guarantee, **(3)** 5–6 diverse benchmarks spanning multiple task types, **(4)** thorough ablations covering every design choice, and **(5)** practical considerations (compute overhead, scalability, composability).

ASGML v2 has strong versions of (1), (4), and (5). It needs to substantially develop (2) and moderately expand (3). The composability argument — that ASGML v2 can be plugged into any existing throttle-dominant method to create two-sided pressure — is the most distinctive selling point and should be elevated to a primary contribution rather than a secondary finding. No other paper in the 34-paper inventory makes this explicit composability claim, and demonstrating it with 3–4 different base methods (OGM-GE, CGGM, MLA, MLGM) would be genuinely novel as a contribution framework.

The field is moving from "yet another gradient modulation scheme" toward **understanding why these schemes work**. Papers that combine empirical methods with theoretical insight (MMPareto, AUG, Zhang et al. ICML 2024) are consistently placed higher than purely empirical ones. The decoupled-vs-coupled probe analysis is exactly the kind of insight that could elevate ASGML v2 — but it needs to be formalized, not just argued informally.
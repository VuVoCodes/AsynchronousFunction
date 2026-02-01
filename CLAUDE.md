# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeurIPS 2026 submission: **Asynchronous Staleness Guided Multimodal Learning (ASGML)**

A novel approach to balanced multimodal learning through adaptive asynchronous updates, where each modality has an independent update frequency determined by its learning dynamics.

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

## Research Focus

**Problem**: Dominant modalities suppress weaker ones during the Prime Learning Window.

**Solution**: True asynchronous updates (not just gradient reweighting) where:
- Fast-learning modalities get higher staleness (fewer updates)
- Slow-learning modalities get lower staleness (more updates)
- Staleness adapts based on gradient magnitude ratio + loss descent rate


## Context and Memory

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
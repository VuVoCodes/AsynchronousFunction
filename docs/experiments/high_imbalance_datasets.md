# High-Imbalance Dataset Candidates

Last updated: 2026-04-17

## Goal
Find 2-3 datasets with HIGH modality imbalance to strengthen empirical results. Current paper shows clear improvement only on CREMA-D (1/8 datasets). Need more benchmarks where boost+OGM-GE decisively outperforms OGM-GE alone.

## Current Datasets (already in paper)

| Dataset | Modalities | Gap | Imbalance | Our Best Result |
|---------|-----------|-----|-----------|-----------------|
| CREMA-D | audio + visual | ~13pp | HIGH | Boost+OGM 71.45% (+2.31pp over OGM) |
| AVE | audio + visual | ~3-4pp | LOW | Boost-only 87.41% (+0.87pp) |
| KS | audio + visual | ~2-3pp | LOW | Boost-only 79.17% (+0.12pp, noise) |
| MOSEI | text + audio + visual | ~6-8pp | MEDIUM | OGM-GE wins (72.47%) |
| MOSI | text + audio + visual | ~3-4pp | LOW | All tied (~72.5%) |
| Twitter15 | text + image | small | LOW | Boost-only 66.85% (+0.25pp, noise) |
| Sarcasm | text + image | small | LOW | Boost-only 82.44% (+0.04pp, noise) |
| BraTS | 4x MRI | small | MEDIUM | Boost+OGM 86.49% (+0.72pp) |

**Problem:** Only 1 HIGH imbalance dataset. Need 2-3 more.

---

## TIER 1: HIGHEST PRIORITY

### 1. UPMC-Food101 (text + image) — MUST ADD

- **Citation:** Wang et al., ICME 2015
- **Modalities:** Food images + recipe text descriptions
- **Dominant modality:** Text (by ~20pp)
- **Task:** 101-class food classification
- **Size:** ~90K samples (~67K train / ~23K test)
- **Unimodal accuracies:**
  - CGGM paper: Text = 84.77%, Image = 68.24% (gap: ~16.5pp)
  - BalanceBenchmark: Text = 86.19%, Image = 65.67% (gap: ~20.5pp)
- **Multimodal baseline:** ~90.32%
- **Download:** Kaggle (free): https://www.kaggle.com/datasets/gianmarco96/upmcfood101
- **Used by:** CGGM (NeurIPS 2024), BalanceBenchmark (2025), DynCIM, MILES, MLA (CVPR 2024)
- **Why ideal:**
  - ~20pp gap is comparable to or larger than CREMA-D
  - Different modality pair (text-image vs audio-visual) — shows cross-domain generality
  - Used by CGGM (our must-compare baseline) as a primary benchmark
  - Easy to integrate: frozen BERT + frozen ResNet features with MLP encoders (same as Twitter15/Sarcasm pipeline)
- **Implementation effort:** LOW — reuse Twitter15/Sarcasm pipeline with text+image features
- **Expected outcome:** HIGH probability of showing clear boost+OGM-GE improvement given large gap

### 2. VGGSound (audio + visual) — STRONG RECOMMENDATION

- **Citation:** Chen et al., ICASSP 2020
- **Modalities:** Audio + visual (video frames)
- **Dominant modality:** Audio (by ~13-15pp)
- **Task:** 309-class audio-visual event classification
- **Size:** ~200K videos (~168K train / ~14K test)
- **Unimodal accuracies:**
  - OGM-GE paper: Audio = 44.3%, Visual = 31.0% (gap: ~13.3pp)
  - BalanceBenchmark: Audio = 41.27%, Visual = 30.43% (gap: ~10.8pp)
- **Multimodal baseline:** ~49.1%
- **Download:** Free from https://www.robots.ox.ac.uk/~vgg/data/vggsound/ (CSV with YouTube URLs, need to download videos)
- **Used by:** OGM-GE (CVPR 2022), OPM (TPAMI 2024), AUG (NeurIPS 2025), G-Blend, BalanceBenchmark
- **Why ideal:**
  - Large-scale (200K) — statistical power
  - Audio-visual like CREMA-D but different content (general events vs emotion)
  - Used by OGM-GE (our key baseline)
  - Same audio-dominant pattern as CREMA-D
- **Implementation effort:** MEDIUM — reuse CREMA-D/AVE ResNet18 pipeline, but need to download videos from YouTube (some may be unavailable)
- **Caveat:** YouTube videos may be taken down; typical availability ~80-85% of original URLs
- **Expected outcome:** HIGH probability of improvement given large gap and audio-visual domain

### 3. MM-IMDb (text + image) — STRONG RECOMMENDATION

- **Citation:** Arevalo et al., ICLR Workshop 2017
- **Modalities:** Movie posters (image) + plot summaries (text)
- **Dominant modality:** Text (by ~25pp — LARGEST GAP of any standard benchmark)
- **Task:** Multi-label genre classification (23 genres)
- **Size:** 25,959 movies (15,552 train / 2,608 val / 7,799 test)
- **Unimodal performance:**
  - Text F1 = 64.4%, Image F1 = 38.9% (gap: ~25.5pp)
- **Download:** Kaggle (free): https://www.kaggle.com/datasets/javierurea/simplified-mm-imdb
- **Used by:** MultiBench, DynCIM, MILES
- **Why ideal:**
  - Largest text-image modality gap (~25pp) among standard benchmarks
  - Multi-label classification is a different evaluation paradigm (F1 score, not accuracy)
  - Adds evaluation diversity
- **Implementation effort:** MEDIUM — need multi-label classification head (BCE loss instead of CE), F1 metric
- **Expected outcome:** HIGH probability of improvement given very large gap

---

## TIER 2: SECONDARY CANDIDATES

### 4. IEMOCAP (audio + video + text)

- **Citation:** Busso et al., 2008
- **Modalities:** Audio, video, text transcriptions (3-modal)
- **Dominant modality:** Text (gap: ~13pp over audio)
- **Task:** 4-class emotion recognition
- **Size:** 5,531 utterances, 5 sessions (leave-one-session-out CV)
- **Unimodal (from CGGM):** Text = 65.35%, Audio = 52.18%, Video = 54.55%
- **Download:** Restricted — requires USC SAIL registration (https://sail.usc.edu/iemocap/)
- **Used by:** CGGM (NeurIPS 2024)
- **Why:** 3-modal with clear text dominance. Used by CGGM.
- **Caveat:** Access restricted; overlaps with MOSEI/MOSI (text+audio+visual sentiment); small dataset; leave-one-session-out CV is non-standard

### 5. UCF-101 (RGB + Optical Flow)

- **Citation:** Soomro et al., 2012
- **Modalities:** RGB video + optical flow
- **Task:** 101-class action recognition
- **Size:** 13,320 videos
- **Unimodal:** RGB = 78.60%, Optical Flow = 70.55% (gap: ~8pp)
- **Download:** Free (https://www.crcv.ucf.edu/data/UCF101.php)
- **Used by:** OPM (TPAMI 2024), BalanceBenchmark
- **Why:** Action recognition domain. But moderate gap (~8pp) and both modalities are visual.

### 6. Facebook Hateful Memes (text + image)

- **Citation:** Kiela et al., NeurIPS 2020
- **Modalities:** Meme images + overlaid text
- **Task:** Binary hate speech detection
- **Size:** ~12K samples
- **Unimodal:** Text AUROC ~0.64, Image AUROC ~0.53
- **Download:** DrivenData (https://www.drivendata.org/competitions/group/hateful-memes/)
- **Why:** Interesting benchmark but moderate gap and small dataset

### 7. MELD (audio + video + text)

- **Citation:** Poria et al., ACL 2019
- **Modalities:** Audio, video, text from Friends TV series
- **Task:** 7-class emotion recognition in conversation
- **Size:** 13,708 utterances
- **Download:** Free (https://github.com/declare-lab/MELD)
- **Why:** Text-dominant, larger than IEMOCAP, freely available. But overlaps with MOSEI/MOSI.

---

## DECISION MATRIX

| Dataset | Gap | Effort | Used by key papers | Download | Different from existing | Priority |
|---------|-----|--------|-------------------|----------|----------------------|----------|
| **UPMC-Food101** | ~20pp | LOW | CGGM, BalanceBench | Free | Yes (text+image food) | **#1** |
| **VGGSound** | ~13pp | MEDIUM | OGM-GE, OPM, AUG | Free* | Partly (audio-visual) | **#2** |
| **MM-IMDb** | ~25pp | MEDIUM | MultiBench, MILES | Free | Yes (multi-label) | **#3** |
| IEMOCAP | ~13pp | LOW | CGGM | Restricted | Partly (3-modal emotion) | #4 |

*VGGSound requires YouTube video downloading

## RECOMMENDATION

**Add these 2 (minimum):**
1. **UPMC-Food101** — must-have, directly matches CGGM's evaluation, huge gap, easy pipeline
2. **VGGSound** or **MM-IMDb** — pick based on implementation preference

**If adding 3:**
1. UPMC-Food101
2. VGGSound (if comfortable with YouTube downloading)
3. MM-IMDb (if want maximum diversity)

## IMPLEMENTATION NOTES

### UPMC-Food101 Pipeline
- Reuse Twitter15/Sarcasm pipeline: frozen BERT (text) + frozen ResNet18 (image) → pre-extracted features
- MLP encoders (2-layer, 512 hidden, dropout 0.3)
- Concat fusion, Adam lr=1e-3, StepLR step=40, 100 epochs
- Standard 70/30 train/test split
- Metric: top-1 accuracy

### VGGSound Pipeline
- Reuse CREMA-D/AVE pipeline: ResNet18 audio + ResNet18 visual
- Need to download videos and extract audio spectrograms + video frames
- SGD lr=0.001, StepLR, 100 epochs
- 309-class classification
- Metric: top-1 accuracy
- **Main effort:** Video downloading + preprocessing (~2-3 days)

### MM-IMDb Pipeline
- Frozen BERT (text) + frozen ResNet (image) → pre-extracted features
- MLP encoders, concat fusion
- **Key difference:** Multi-label classification → BCE loss, F1-micro/macro metric
- Need sigmoid output instead of softmax
- Metric: F1 score (micro or macro)

## ESTIMATED TIMELINE

| Task | Time |
|------|------|
| Download + preprocess Food101 | 1 day |
| Run all methods on Food101 (10 methods x 5 seeds) | 2-3 days |
| Download + preprocess VGGSound | 2-3 days |
| Run all methods on VGGSound | 3-5 days (larger dataset) |
| Update paper tables + text | 1 day |
| **Total** | **~1-2 weeks** |

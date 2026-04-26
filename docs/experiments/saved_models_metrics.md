# Saved-Model Metrics: Task-Appropriate Evaluation

Metrics extracted from `train.log` files at the best-primary-metric epoch. **Tasks differ in primary metric**:
- **Classification** (CREMA-D, AVE, Kinetics-Sounds, Twitter15, Sarcasm, MOSI, MOSEI): Test Accuracy (primary) + F1-macro.
- **Segmentation** (BraTS 2021): Mean Dice (primary) + WT/TC/ET sub-Dice.

**MOSI / MOSEI** are sentiment regression tasks evaluated as binary positive/negative classification per the paper's protocol (matching MMPareto / OGM-GE / CGGM comparisons). Regression-specific metrics (MAE, Pearson correlation) are not logged in `train.log` and would require re-running inference.

**mAP is not retrievable from logs** — per-class probability archives were not retained across training runs. Computing mAP requires re-running inference on saved checkpoints (GPU-bound, deferred until the AVE+Food101 sweep finishes).

- Total completed runs surveyed: **563** (538 classification, 25 segmentation)
- Distinct (sweep, method) groups: **156** (151 classification, 5 segmentation)

Aggregates report `mean ± std` across seeds (sample std, n−1) when ≥2 seeds; otherwise the single-seed value.

# Classification Tasks (Acc / F1-macro)

## (top)

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| probe_stability_val | 1 | 70.30 | 70.54 | - |
| smoketest_boost_agm | 1 | 36.96 | 32.19 | - |
| smoketest_boost_cggm | 1 | 33.87 | 25.67 | - |
| smoketest_boost_gblend | 1 | 38.44 | 31.67 | - |
| smoketest_boost_gblend_v2 | 1 | 38.44 | 31.67 | - |
| smoketest_boost_gblend_v3 | 1 | 38.44 | 31.67 | - |
| smoketest_boost_mmpareto | 1 | 34.81 | 29.69 | - |
| smoketest_boost_ogm | 1 | 38.71 | 33.42 | - |

## cremad_arl

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| cremad_arl | 1 | 63.31 | 63.66 | 42 |

## cremad_arl_v2

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| cremad_arl_v2 | 1 | 62.90 | 63.23 | 42 |

## smoke_arl

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| arl_smoke | 1 | 41.26 | 35.02 | - |

## sweep

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| asgml_default | 5 | 60.40 ± 1.43 | 60.63 ± 1.43 | 0,1,2,3,42 |
| baseline | 5 | 60.30 ± 0.74 | 60.34 ± 0.77 | 0,1,2,3,42 |
| boost_a025 | 1 | 58.74 | 59.09 | 42 |
| boost_a075 | 1 | 58.20 | 57.93 | 42 |
| boost_a100 | 1 | 58.60 | 58.93 | 42 |
| boost_default | 1 | 60.48 | 60.58 | 42 |
| boost_noise | 1 | 58.87 | 58.93 | 42 |
| boost_ogm | 1 | 61.83 | 61.77 | 42 |
| boost_ogm_a075 | 1 | 62.50 | 62.76 | 42 |
| boost_sm150 | 1 | 60.48 | 60.58 | 42 |
| boost_sm300 | 1 | 60.48 | 60.58 | 42 |
| cont_a025 | 1 | 58.47 | 58.37 | 42 |
| cont_a075 | 1 | 56.45 | 56.39 | 42 |
| cont_a100 | 1 | 53.63 | 53.35 | 42 |
| cont_combo | 1 | 52.96 | 53.13 | 42 |
| cont_default | 1 | 57.66 | 57.79 | 42 |
| cont_noise | 1 | 56.18 | 55.74 | 42 |
| cont_ogm | 1 | 59.41 | 59.55 | 42 |
| cont_sm005 | 1 | 57.66 | 57.79 | 42 |
| cont_sm030 | 1 | 57.66 | 57.79 | 42 |
| ogmge | 5 | 62.47 ± 1.43 | 62.67 ± 1.59 | 0,1,2,3,42 |
| p2_beta100 | 5 | 59.76 ± 0.63 | 59.73 ± 1.00 | 0,1,2,3,42 |
| p2_boost_default | 5 | 60.46 ± 0.88 | 60.68 ± 0.76 | 0,1,2,3,42 |
| p2_boost_ogm | 5 | 62.37 ± 0.58 | 62.50 ± 0.64 | 0,1,2,3,42 |
| p2_boost_ogm_a075 | 5 | 62.69 ± 0.23 | 62.88 ± 0.18 | 0,1,2,3,42 |
| p2_default | 5 | 60.40 ± 1.43 | 60.63 ± 1.43 | 0,1,2,3,42 |
| p2_sms000 | 5 | 60.54 ± 1.19 | 60.83 ± 1.34 | 0,1,2,3,42 |
| p2_stale_lc020 | 5 | 59.25 ± 0.90 | 59.38 ± 0.83 | 0,1,2,3,42 |

## sweep_3f

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| 3f_agm | 5 | 57.42 ± 0.78 | 57.45 ± 0.74 | 42,123,456,789,1024 |
| 3f_baseline | 5 | 61.59 ± 0.83 | 61.93 ± 1.05 | 42,123,456,789,1024 |
| 3f_boost_ogm_a075 | 5 | 71.45 ± 1.70 | 71.85 ± 1.74 | 42,123,456,789,1024 |
| 3f_boost_only | 5 | 62.72 ± 1.83 | 62.95 ± 1.87 | 42,123,456,789,1024 |
| 3f_gblend | 5 | 61.10 ± 2.04 | 61.50 ± 2.12 | 42,123,456,789,1024 |
| 3f_inforeg_100ep | 5 | 67.72 ± 0.90 | 68.22 ± 0.84 | 42,123,456,789,1024 |
| 3f_inforeg_paper | 1 | 66.40 | 66.84 | 42 |
| 3f_miles_t005 | 1 | 58.60 | 57.89 | 42 |
| 3f_miles_t02 | 5 | 61.05 ± 2.55 | 61.34 ± 2.64 | 42,123,456,789,1024 |
| 3f_mmpareto | 5 | 65.51 ± 0.90 | 66.10 ± 0.83 | 42,123,456,789,1024 |
| 3f_ogm_ge | 5 | 69.14 ± 1.18 | 69.52 ± 1.14 | 42,123,456,789,1024 |

## sweep_3way_ablation

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| monitor_ogm_noboost | 5 | 69.14 ± 1.18 | 69.52 ± 1.14 | 42,123,456,789,1024 |

## sweep_ave

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| ave_agm | 5 | 84.42 ± 0.29 | 84.05 ± 0.43 | 42,123,456,789,1024 |
| ave_baseline | 5 | 86.54 ± 0.47 | 86.29 ± 0.33 | 42,123,456,789,1024 |
| ave_boost_ogm_a075 | 5 | 87.23 ± 0.65 | 86.86 ± 0.63 | 42,123,456,789,1024 |
| ave_boost_only | 5 | 87.41 ± 0.29 | 87.15 ± 0.39 | 42,123,456,789,1024 |
| ave_gblend | 5 | 87.09 ± 0.33 | 86.79 ± 0.39 | 42,123,456,789,1024 |
| ave_mmpareto | 5 | 86.37 ± 0.38 | 85.91 ± 0.47 | 42,123,456,789,1024 |
| ave_ogm_ge | 5 | 86.96 ± 0.79 | 86.48 ± 0.95 | 42,123,456,789,1024 |

## sweep_ave_scratch

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| ave_scratch_baseline | 5 | 67.55 ± 0.54 | 65.35 ± 0.64 | 42,123,456,789,1024 |
| ave_scratch_boost_ogm | 1 | 63.46 | 60.40 | 42 |
| ave_scratch_boost_ogm_a075 | 4 | 62.87 ± 0.70 | 60.11 ± 0.99 | 123,456,789,1024 |
| ave_scratch_boost_only | 5 | 67.97 ± 0.60 | 65.79 ± 0.71 | 42,123,456,789,1024 |

## sweep_boost_compose

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| boost_agm | 5 | 58.17 ± 2.00 | 58.19 ± 2.15 | 42,123,456,789,1024 |
| boost_cggm | 5 | 50.32 ± 1.15 | 49.62 ± 1.22 | 42,123,456,789,1024 |
| boost_gblend | 5 | 61.99 ± 1.10 | 62.31 ± 1.25 | 42,123,456,789,1024 |
| boost_mmpareto | 5 | 66.00 ± 1.40 | 66.56 ± 1.46 | 42,123,456,789,1024 |

## sweep_boost_compose_ave_food101

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| ave_boost_agm | 5 | 85.68 ± 0.36 | 85.32 ± 0.39 | 42,123,456,789,1024 |
| ave_boost_cggm | 5 | 76.47 ± 0.20 | 74.38 ± 0.50 | 42,123,456,789,1024 |
| ave_boost_gblend | 5 | 86.84 ± 0.46 | 86.41 ± 0.42 | 42,123,456,789,1024 |
| ave_boost_mmpareto | 5 | 87.38 ± 0.95 | 86.86 ± 0.94 | 42,123,456,789,1024 |
| food101_boost_agm | 5 | 83.91 ± 0.15 | 83.83 ± 0.15 | 42,123,456,789,1024 |
| food101_boost_cggm | 5 | 53.00 ± 0.16 | 52.43 ± 0.19 | 42,123,456,789,1024 |
| food101_boost_gblend | 5 | 85.22 ± 0.13 | 85.12 ± 0.13 | 42,123,456,789,1024 |
| food101_boost_mmpareto | 5 | 85.59 ± 0.13 | 85.50 ± 0.14 | 42,123,456,789,1024 |

## sweep_cggm

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| ave_cggm | 5 | 76.72 ± 0.47 | 74.85 ± 0.50 | 42,123,456,789,1024 |
| cremad_cggm | 5 | 50.22 ± 1.55 | 49.35 ± 1.87 | 42,123,456,789,1024 |
| ks_cggm | 5 | 73.18 ± 0.31 | 72.63 ± 0.46 | 42,123,456,789,1024 |
| mosei_cggm | 5 | 68.05 ± 0.51 | 47.49 ± 0.50 | 42,123,456,789,1024 |
| mosi_cggm | 5 | 59.45 ± 0.40 | 37.73 ± 0.56 | 42,123,456,789,1024 |

## sweep_cggm_hp

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| cremad_cggm_r0.5_l0.0 | 1 | 46.77 | 45.87 | 42 |
| cremad_cggm_r0.5_l0.1 | 1 | 46.77 | 45.87 | 42 |
| cremad_cggm_r0.8_l0.0 | 1 | 49.33 | 48.04 | 42 |
| cremad_cggm_r0.8_l0.1 | 1 | 49.33 | 48.04 | 42 |
| cremad_cggm_r1.0_l0.0 | 1 | 50.54 | 49.97 | 42 |
| cremad_cggm_r1.0_l0.1 | 1 | 50.54 | 49.97 | 42 |

## sweep_cremad_opm

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| cremad_baseline_opmarch | 5 | 59.20 ± 1.35 | 59.18 ± 1.52 | 42,123,456,789,1024 |
| cremad_boost_ogm_opmarch | 5 | 71.77 ± 0.94 | 72.16 ± 0.92 | 42,123,456,789,1024 |
| cremad_ogm_opmarch | 5 | 63.74 ± 0.47 | 64.01 ± 0.53 | 42,123,456,789,1024 |
| cremad_opm | 5 | 65.43 ± 0.24 | 65.96 ± 0.16 | 42,123,456,789,1024 |
| cremad_opm_ogm | 5 | 69.84 ± 1.53 | 70.23 ± 1.44 | 42,123,456,789,1024 |

## sweep_food101

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| food101_agm | 5 | 84.14 ± 0.18 | 84.06 ± 0.18 | 42,123,456,789,1024 |
| food101_baseline | 5 | 85.75 ± 0.18 | 85.67 ± 0.18 | 42,123,456,789,1024 |
| food101_boost_ogm_a075 | 5 | 84.11 ± 0.17 | 84.01 ± 0.17 | 42,123,456,789,1024 |
| food101_boost_only | 5 | 85.60 ± 0.18 | 85.51 ± 0.18 | 42,123,456,789,1024 |
| food101_cggm | 5 | 48.59 ± 0.21 | 47.85 ± 0.23 | 42,123,456,789,1024 |
| food101_gblend | 5 | 85.31 ± 0.14 | 85.22 ± 0.15 | 42,123,456,789,1024 |
| food101_mmpareto | 5 | 85.68 ± 0.26 | 85.60 ± 0.26 | 42,123,456,789,1024 |
| food101_ogm_ge | 5 | 84.22 ± 0.18 | 84.12 ± 0.19 | 42,123,456,789,1024 |

## sweep_hp

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| 3f_hp_alpha0.25 | 1 | 68.55 | 69.09 | - |
| 3f_hp_alpha0.5 | 1 | 69.76 | 70.08 | - |
| 3f_hp_alpha1.0 | 1 | 70.30 | 70.40 | - |
| 3f_hp_alpha1.5 | 1 | 70.30 | 70.40 | - |
| 3f_hp_ema0.1 | 1 | 71.64 | 71.83 | - |
| 3f_hp_ema0.5 | 1 | 71.64 | 71.83 | - |
| 3f_hp_ema0.7 | 1 | 71.64 | 71.83 | - |
| 3f_hp_smax1.5 | 1 | 69.76 | 70.08 | - |
| 3f_hp_smax3.0 | 1 | 71.64 | 71.83 | - |

## sweep_k_ablation

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| 3f_boost_ogm_K1 | 1 | 71.24 | 71.64 | 42 |
| 3f_boost_ogm_K10 | 1 | 70.83 | 71.27 | 42 |
| 3f_boost_ogm_K100 | 1 | 71.24 | 71.35 | 42 |
| 3f_boost_ogm_K5 | 1 | 71.64 | 71.83 | 42 |
| 3f_boost_ogm_K50 | 1 | 71.24 | 71.90 | 42 |

## sweep_ks

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| ks_agm | 5 | 77.85 ± 0.34 | 77.24 ± 0.42 | 42,123,456,789,1024 |
| ks_baseline | 5 | 79.05 ± 0.45 | 78.40 ± 0.56 | 42,123,456,789,1024 |
| ks_boost_ogm | 5 | 77.33 ± 0.71 | 76.45 ± 0.61 | 42,123,456,789,1024 |
| ks_boost_only | 5 | 79.17 ± 1.09 | 78.41 ± 1.08 | 42,123,456,789,1024 |
| ks_gblend | 5 | 77.75 ± 1.23 | 77.15 ± 1.17 | 42,123,456,789,1024 |
| ks_mmpareto | 5 | 78.21 ± 0.42 | 77.06 ± 0.58 | 42,123,456,789,1024 |
| ks_ogmge | 5 | 77.25 ± 0.89 | 76.44 ± 0.87 | 42,123,456,789,1024 |

## sweep_mosei

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| mosei_agm | 5 | 69.28 ± 0.75 | 59.00 ± 2.24 | 42,123,456,789,1024 |
| mosei_baseline | 5 | 70.42 ± 0.33 | 60.05 ± 0.64 | 42,123,456,789,1024 |
| mosei_boost_ogm_a075 | 5 | 72.43 ± 0.73 | 62.36 ± 1.82 | 42,123,456,789,1024 |
| mosei_boost_only | 5 | 69.80 ± 0.89 | 60.05 ± 1.70 | 42,123,456,789,1024 |
| mosei_gblend | 5 | 70.15 ± 0.74 | 58.60 ± 1.95 | 42,123,456,789,1024 |
| mosei_mmpareto | 5 | 70.20 ± 0.79 | 58.79 ± 3.28 | 42,123,456,789,1024 |
| mosei_ogm_ge | 5 | 72.47 ± 0.78 | 61.46 ± 1.80 | 42,123,456,789,1024 |

## sweep_mosi

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| mosi_agm | 5 | 72.50 ± 0.69 | 70.11 ± 1.39 | 42,123,456,789,1024 |
| mosi_baseline | 5 | 72.42 ± 0.54 | 70.16 ± 0.73 | 42,123,456,789,1024 |
| mosi_boost_ogm | 5 | 72.60 ± 1.05 | 69.93 ± 1.37 | 42,123,456,789,1024 |
| mosi_boost_only | 5 | 71.89 ± 0.91 | 69.02 ± 2.81 | 42,123,456,789,1024 |
| mosi_cggm | 1 | 59.77 | 38.40 | 42 |
| mosi_gblend | 5 | 72.36 ± 0.74 | 70.50 ± 0.95 | 42,123,456,789,1024 |
| mosi_mmpareto | 5 | 72.68 ± 0.88 | 70.69 ± 0.89 | 42,123,456,789,1024 |
| mosi_ogmge | 5 | 72.68 ± 0.99 | 70.12 ± 1.62 | 42,123,456,789,1024 |

## sweep_sarcasm

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| sarcasm_agm | 5 | 82.32 ± 0.18 | 81.66 ± 0.26 | 42,123,456,789,1024 |
| sarcasm_baseline | 5 | 82.40 ± 0.19 | 81.81 ± 0.23 | 42,123,456,789,1024 |
| sarcasm_boost_ogm_a075 | 5 | 81.76 ± 0.09 | 81.19 ± 0.11 | 42,123,456,789,1024 |
| sarcasm_boost_only | 5 | 82.44 ± 0.39 | 81.83 ± 0.37 | 42,123,456,789,1024 |
| sarcasm_cggm | 5 | 80.71 ± 0.30 | 80.07 ± 0.28 | 42,123,456,789,1024 |
| sarcasm_gblend | 5 | 82.35 ± 0.29 | 81.64 ± 0.33 | 42,123,456,789,1024 |
| sarcasm_mmpareto | 5 | 82.27 ± 0.14 | 81.61 ± 0.20 | 42,123,456,789,1024 |
| sarcasm_ogm_ge | 5 | 81.81 ± 0.25 | 81.09 ± 0.45 | 42,123,456,789,1024 |

## sweep_trial

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| trial_gradclip | 1 | 71.64 | 71.83 | - |

## sweep_twitter

| Method | N | Acc (%) | F1-macro (%) | Seeds |
|---|---|---|---|---|
| twitter_agm | 5 | 66.79 ± 0.52 | 54.15 ± 2.25 | 42,123,456,789,1024 |
| twitter_baseline | 5 | 66.60 ± 0.64 | 51.33 ± 4.67 | 42,123,456,789,1024 |
| twitter_boost_ogm_a075 | 5 | 66.59 ± 0.57 | 52.72 ± 3.78 | 42,123,456,789,1024 |
| twitter_boost_only | 5 | 66.85 ± 0.56 | 54.20 ± 4.83 | 42,123,456,789,1024 |
| twitter_cggm | 5 | 62.41 ± 0.40 | 39.02 ± 0.32 | 42,123,456,789,1024 |
| twitter_gblend | 5 | 66.66 ± 0.64 | 56.29 ± 1.75 | 42,123,456,789,1024 |
| twitter_mmpareto | 5 | 66.81 ± 0.52 | 55.36 ± 2.96 | 42,123,456,789,1024 |
| twitter_ogm_ge | 5 | 66.33 ± 0.24 | 49.98 ± 5.73 | 42,123,456,789,1024 |

# Segmentation Tasks (Dice + WT / TC / ET)

## sweep_brats

| Method | N | Mean Dice (%) | WT Dice (%) | TC Dice (%) | ET Dice (%) | Seeds |
|---|---|---|---|---|---|---|
| brats_asgml | 5 | 87.06 ± 0.66 | 88.09 ± 1.17 | 89.62 ± 1.26 | 83.47 ± 0.90 | 42,123,456,789,1024 |
| brats_baseline | 5 | 86.23 ± 0.24 | 87.64 ± 0.42 | 88.16 ± 0.88 | 82.90 ± 0.83 | 42,123,456,789,1024 |
| brats_boost_ogmge | 5 | 86.49 ± 0.24 | 87.40 ± 0.64 | 88.91 ± 0.98 | 83.17 ± 0.65 | 42,123,456,789,1024 |
| brats_cggm | 5 | 83.00 ± 0.78 | 85.18 ± 0.83 | 85.28 ± 1.04 | 78.55 ± 1.14 | 42,123,456,789,1024 |
| brats_ogmge | 5 | 86.21 ± 0.18 | 87.41 ± 0.30 | 88.09 ± 0.18 | 83.15 ± 0.34 | 42,123,456,789,1024 |

# Per-run Index — Classification

| Sweep | Experiment | Seed | Best Ep | Acc (%) | F1 (%) | ckpt |
|---|---|---|---|---|---|---|
| (top) | probe_stability_val | - | 56 | 70.30 | 70.54 | ✓ |
| (top) | smoketest_boost_agm | - | 1 | 36.96 | 32.19 | ✓ |
| (top) | smoketest_boost_cggm | - | 2 | 33.87 | 25.67 | ✓ |
| (top) | smoketest_boost_gblend | - | 2 | 38.44 | 31.67 | ✓ |
| (top) | smoketest_boost_gblend_v2 | - | 2 | 38.44 | 31.67 | ✓ |
| (top) | smoketest_boost_gblend_v3 | - | 2 | 38.44 | 31.67 | ✓ |
| (top) | smoketest_boost_mmpareto | - | 2 | 34.81 | 29.69 | ✓ |
| (top) | smoketest_boost_ogm | - | 3 | 38.71 | 33.42 | ✓ |
| cremad_arl | cremad_arl_seed42 | 42 | 63 | 63.31 | 63.66 | ✓ |
| cremad_arl_v2 | cremad_arl_v2_seed42 | 42 | 96 | 62.90 | 63.23 | ✓ |
| smoke_arl | arl_smoke | - | 2 | 41.26 | 35.02 | ✓ |
| sweep | asgml_default_seed0 | - | 59 | 59.68 | 59.87 | ✓ |
| sweep | asgml_default_seed1 | 1 | 78 | 59.41 | 59.40 | ✓ |
| sweep | asgml_default_seed2 | 2 | 85 | 62.77 | 62.91 | ✓ |
| sweep | asgml_default_seed3 | 3 | 58 | 59.41 | 59.81 | ✓ |
| sweep | asgml_default_seed42 | 42 | 57 | 60.75 | 61.15 | ✓ |
| sweep | baseline_seed0 | - | 27 | 59.95 | 60.05 | ✓ |
| sweep | baseline_seed1 | 1 | 27 | 61.56 | 61.63 | ✓ |
| sweep | baseline_seed2 | 2 | 100 | 60.35 | 60.45 | ✓ |
| sweep | baseline_seed3 | 3 | 38 | 59.81 | 59.74 | ✓ |
| sweep | baseline_seed42 | 42 | 82 | 59.81 | 59.85 | ✓ |
| sweep | boost_a025_seed42 | 42 | 37 | 58.74 | 59.09 | ✓ |
| sweep | boost_a075_seed42 | 42 | 30 | 58.20 | 57.93 | ✓ |
| sweep | boost_a100_seed42 | 42 | 58 | 58.60 | 58.93 | ✓ |
| sweep | boost_default_seed42 | 42 | 31 | 60.48 | 60.58 | ✓ |
| sweep | boost_noise_seed42 | 42 | 57 | 58.87 | 58.93 | ✓ |
| sweep | boost_ogm_a075_seed42 | 42 | 96 | 62.50 | 62.76 | ✓ |
| sweep | boost_ogm_seed42 | 42 | 75 | 61.83 | 61.77 | ✓ |
| sweep | boost_sm150_seed42 | 42 | 31 | 60.48 | 60.58 | ✓ |
| sweep | boost_sm300_seed42 | 42 | 31 | 60.48 | 60.58 | ✓ |
| sweep | cont_a025_seed42 | 42 | 39 | 58.47 | 58.37 | ✓ |
| sweep | cont_a075_seed42 | 42 | 91 | 56.45 | 56.39 | ✓ |
| sweep | cont_a100_seed42 | 42 | 29 | 53.63 | 53.35 | ✓ |
| sweep | cont_combo_seed42 | 42 | 79 | 52.96 | 53.13 | ✓ |
| sweep | cont_default_seed42 | 42 | 58 | 57.66 | 57.79 | ✓ |
| sweep | cont_noise_seed42 | 42 | 38 | 56.18 | 55.74 | ✓ |
| sweep | cont_ogm_seed42 | 42 | 98 | 59.41 | 59.55 | ✓ |
| sweep | cont_sm005_seed42 | 42 | 58 | 57.66 | 57.79 | ✓ |
| sweep | cont_sm030_seed42 | 42 | 58 | 57.66 | 57.79 | ✓ |
| sweep | ogmge_seed0 | - | 87 | 60.89 | 60.81 | ✓ |
| sweep | ogmge_seed1 | 1 | 94 | 61.02 | 61.10 | ✓ |
| sweep | ogmge_seed2 | 2 | 72 | 63.04 | 63.52 | ✓ |
| sweep | ogmge_seed3 | 3 | 79 | 63.44 | 63.76 | ✓ |
| sweep | ogmge_seed42 | 42 | 98 | 63.98 | 64.16 | ✓ |
| sweep | p2_beta100_seed0 | - | 59 | 60.22 | 60.40 | ✓ |
| sweep | p2_beta100_seed1 | 1 | 65 | 59.14 | 58.28 | ✓ |
| sweep | p2_beta100_seed2 | 2 | 67 | 60.22 | 60.45 | ✓ |
| sweep | p2_beta100_seed3 | 3 | 72 | 59.01 | 59.07 | ✓ |
| sweep | p2_beta100_seed42 | 42 | 73 | 60.22 | 60.43 | ✓ |
| sweep | p2_boost_default_seed0 | - | 37 | 61.02 | 61.35 | ✓ |
| sweep | p2_boost_default_seed1 | 1 | 38 | 60.48 | 60.76 | ✓ |
| sweep | p2_boost_default_seed2 | 2 | 69 | 61.29 | 61.25 | ✓ |
| sweep | p2_boost_default_seed3 | 3 | 31 | 59.01 | 59.44 | ✓ |
| sweep | p2_boost_default_seed42 | 42 | 31 | 60.48 | 60.58 | ✓ |
| sweep | p2_boost_ogm_a075_seed0 | - | 98 | 63.04 | 63.17 | ✓ |
| sweep | p2_boost_ogm_a075_seed1 | 1 | 94 | 62.50 | 62.76 | ✓ |
| sweep | p2_boost_ogm_a075_seed2 | 2 | 78 | 62.63 | 62.76 | ✓ |
| sweep | p2_boost_ogm_a075_seed3 | 3 | 82 | 62.77 | 62.97 | ✓ |
| sweep | p2_boost_ogm_a075_seed42 | 42 | 96 | 62.50 | 62.76 | ✓ |
| sweep | p2_boost_ogm_seed0 | - | 90 | 62.37 | 62.29 | ✓ |
| sweep | p2_boost_ogm_seed1 | 1 | 87 | 61.96 | 62.22 | ✓ |
| sweep | p2_boost_ogm_seed2 | 2 | 87 | 62.37 | 62.75 | ✓ |
| sweep | p2_boost_ogm_seed3 | 3 | 96 | 63.31 | 63.46 | ✓ |
| sweep | p2_boost_ogm_seed42 | 42 | 75 | 61.83 | 61.77 | ✓ |
| sweep | p2_default_seed0 | - | 59 | 59.68 | 59.87 | ✓ |
| sweep | p2_default_seed1 | 1 | 78 | 59.41 | 59.40 | ✓ |
| sweep | p2_default_seed2 | 2 | 85 | 62.77 | 62.91 | ✓ |
| sweep | p2_default_seed3 | 3 | 58 | 59.41 | 59.81 | ✓ |
| sweep | p2_default_seed42 | 42 | 57 | 60.75 | 61.15 | ✓ |
| sweep | p2_sms000_seed0 | - | 70 | 60.08 | 60.27 | ✓ |
| sweep | p2_sms000_seed1 | 1 | 98 | 60.62 | 60.91 | ✓ |
| sweep | p2_sms000_seed2 | 2 | 94 | 62.10 | 62.44 | ✓ |
| sweep | p2_sms000_seed3 | 3 | 77 | 58.87 | 58.91 | ✓ |
| sweep | p2_sms000_seed42 | 42 | 67 | 61.02 | 61.61 | ✓ |
| sweep | p2_stale_lc020_seed0 | - | 69 | 59.01 | 58.87 | ✓ |
| sweep | p2_stale_lc020_seed1 | 1 | 91 | 58.06 | 58.34 | ✓ |
| sweep | p2_stale_lc020_seed2 | 2 | 61 | 60.08 | 60.30 | ✓ |
| sweep | p2_stale_lc020_seed3 | 3 | 40 | 58.87 | 59.25 | ✓ |
| sweep | p2_stale_lc020_seed42 | 42 | 57 | 60.22 | 60.14 | ✓ |
| sweep_3f | 3f_agm_seed1024 | 1024 | 69 | 57.53 | 57.52 | ✓ |
| sweep_3f | 3f_agm_seed123 | 123 | 62 | 58.60 | 58.53 | ✓ |
| sweep_3f | 3f_agm_seed42 | 42 | 38 | 56.85 | 56.74 | ✓ |
| sweep_3f | 3f_agm_seed456 | 456 | 78 | 57.53 | 57.69 | ✓ |
| sweep_3f | 3f_agm_seed789 | 789 | 72 | 56.59 | 56.77 | ✓ |
| sweep_3f | 3f_baseline_seed1024 | 1024 | 45 | 61.02 | 61.16 | ✓ |
| sweep_3f | 3f_baseline_seed123 | 123 | 93 | 62.37 | 62.82 | ✓ |
| sweep_3f | 3f_baseline_seed42 | 42 | 58 | 60.48 | 60.54 | ✓ |
| sweep_3f | 3f_baseline_seed456 | 456 | 70 | 62.37 | 62.93 | ✓ |
| sweep_3f | 3f_baseline_seed789 | 789 | 73 | 61.69 | 62.22 | ✓ |
| sweep_3f | 3f_boost_ogm_a075_seed1024 | 1024 | 86 | 68.95 | 69.23 | ✓ |
| sweep_3f | 3f_boost_ogm_a075_seed123 | 123 | 88 | 72.04 | 72.43 | ✓ |
| sweep_3f | 3f_boost_ogm_a075_seed42 | 42 | 97 | 71.37 | 71.75 | ✓ |
| sweep_3f | 3f_boost_ogm_a075_seed456 | 456 | 83 | 73.66 | 74.05 | ✓ |
| sweep_3f | 3f_boost_ogm_a075_seed789 | 789 | 53 | 71.24 | 71.79 | ✓ |
| sweep_3f | 3f_boost_only_seed1024 | 1024 | 45 | 63.04 | 63.05 | ✓ |
| sweep_3f | 3f_boost_only_seed123 | 123 | 36 | 65.32 | 65.60 | ✓ |
| sweep_3f | 3f_boost_only_seed42 | 42 | 39 | 60.35 | 60.52 | ✓ |
| sweep_3f | 3f_boost_only_seed456 | 456 | 55 | 63.04 | 63.51 | ✓ |
| sweep_3f | 3f_boost_only_seed789 | 789 | 65 | 61.83 | 62.07 | ✓ |
| sweep_3f | 3f_gblend_seed1024 | 1024 | 69 | 60.48 | 60.97 | ✓ |
| sweep_3f | 3f_gblend_seed123 | 123 | 75 | 64.11 | 64.65 | ✓ |
| sweep_3f | 3f_gblend_seed42 | 42 | 77 | 58.60 | 58.90 | ✓ |
| sweep_3f | 3f_gblend_seed456 | 456 | 58 | 61.83 | 62.20 | ✓ |
| sweep_3f | 3f_gblend_seed789 | 789 | 80 | 60.48 | 60.77 | ✓ |
| sweep_3f | 3f_inforeg_100ep_seed1024 | 1024 | 67 | 68.55 | 68.89 | ✓ |
| sweep_3f | 3f_inforeg_100ep_seed123 | 123 | 99 | 68.55 | 69.14 | ✓ |
| sweep_3f | 3f_inforeg_100ep_seed42 | 42 | 73 | 67.07 | 67.47 | ✓ |
| sweep_3f | 3f_inforeg_100ep_seed456 | 456 | 100 | 67.88 | 68.33 | ✓ |
| sweep_3f | 3f_inforeg_100ep_seed789 | 789 | 75 | 66.53 | 67.25 | ✓ |
| sweep_3f | 3f_inforeg_paper_seed42 | 42 | 34 | 66.40 | 66.84 | ✓ |
| sweep_3f | 3f_miles_t005_seed42 | 42 | 71 | 58.60 | 57.89 | ✓ |
| sweep_3f | 3f_miles_t02_seed1024 | 1024 | 68 | 59.81 | 60.18 | ✓ |
| sweep_3f | 3f_miles_t02_seed123 | 123 | 70 | 62.37 | 62.72 | ✓ |
| sweep_3f | 3f_miles_t02_seed42 | 42 | 97 | 64.52 | 65.11 | ✓ |
| sweep_3f | 3f_miles_t02_seed456 | 456 | 11 | 60.75 | 60.48 | ✓ |
| sweep_3f | 3f_miles_t02_seed789 | 789 | 48 | 57.80 | 58.22 | ✓ |
| sweep_3f | 3f_mmpareto_seed1024 | 1024 | 81 | 65.19 | 65.92 | ✓ |
| sweep_3f | 3f_mmpareto_seed123 | 123 | 84 | 64.92 | 65.45 | ✓ |
| sweep_3f | 3f_mmpareto_seed42 | 42 | 67 | 67.07 | 67.54 | ✓ |
| sweep_3f | 3f_mmpareto_seed456 | 456 | 93 | 65.46 | 65.88 | ✓ |
| sweep_3f | 3f_mmpareto_seed789 | 789 | 45 | 64.92 | 65.69 | ✓ |
| sweep_3f | 3f_ogm_ge_seed1024 | 1024 | 86 | 70.83 | 71.13 | ✓ |
| sweep_3f | 3f_ogm_ge_seed123 | 123 | 92 | 68.15 | 68.51 | ✓ |
| sweep_3f | 3f_ogm_ge_seed42 | 42 | 72 | 67.88 | 68.34 | ✓ |
| sweep_3f | 3f_ogm_ge_seed456 | 456 | 96 | 69.35 | 69.71 | ✓ |
| sweep_3f | 3f_ogm_ge_seed789 | 789 | 88 | 69.49 | 69.92 | ✓ |
| sweep_3way_ablation | monitor_ogm_noboost_seed1024 | 1024 | 86 | 70.83 | 71.13 | ✓ |
| sweep_3way_ablation | monitor_ogm_noboost_seed123 | 123 | 92 | 68.15 | 68.51 | ✓ |
| sweep_3way_ablation | monitor_ogm_noboost_seed42 | 42 | 72 | 67.88 | 68.34 | ✓ |
| sweep_3way_ablation | monitor_ogm_noboost_seed456 | 456 | 96 | 69.35 | 69.71 | ✓ |
| sweep_3way_ablation | monitor_ogm_noboost_seed789 | 789 | 88 | 69.49 | 69.92 | ✓ |
| sweep_ave | ave_agm_seed1024 | 1024 | 41 | 84.20 | 83.55 | ✓ |
| sweep_ave | ave_agm_seed123 | 123 | 81 | 84.44 | 84.29 | ✓ |
| sweep_ave | ave_agm_seed42 | 42 | 64 | 84.81 | 84.61 | ✓ |
| sweep_ave | ave_agm_seed456 | 456 | 69 | 84.57 | 84.10 | ✓ |
| sweep_ave | ave_agm_seed789 | 789 | 73 | 84.07 | 83.71 | ✓ |
| sweep_ave | ave_baseline_seed1024 | 1024 | 85 | 86.05 | 85.86 | ✓ |
| sweep_ave | ave_baseline_seed123 | 123 | 83 | 86.05 | 86.06 | ✓ |
| sweep_ave | ave_baseline_seed42 | 42 | 48 | 86.67 | 86.36 | ✓ |
| sweep_ave | ave_baseline_seed456 | 456 | 44 | 86.91 | 86.50 | ✓ |
| sweep_ave | ave_baseline_seed789 | 789 | 76 | 87.04 | 86.67 | ✓ |
| sweep_ave | ave_boost_ogm_a075_seed1024 | 1024 | 76 | 87.16 | 86.67 | ✓ |
| sweep_ave | ave_boost_ogm_a075_seed123 | 123 | 39 | 87.28 | 87.31 | ✓ |
| sweep_ave | ave_boost_ogm_a075_seed42 | 42 | 67 | 86.54 | 86.40 | ✓ |
| sweep_ave | ave_boost_ogm_a075_seed456 | 456 | 90 | 86.91 | 86.22 | ✓ |
| sweep_ave | ave_boost_ogm_a075_seed789 | 789 | 59 | 88.27 | 87.70 | ✓ |
| sweep_ave | ave_boost_only_seed1024 | 1024 | 99 | 87.41 | 87.07 | ✓ |
| sweep_ave | ave_boost_only_seed123 | 123 | 79 | 87.65 | 87.56 | ✓ |
| sweep_ave | ave_boost_only_seed42 | 42 | 51 | 86.91 | 86.57 | ✓ |
| sweep_ave | ave_boost_only_seed456 | 456 | 44 | 87.53 | 87.09 | ✓ |
| sweep_ave | ave_boost_only_seed789 | 789 | 70 | 87.53 | 87.46 | ✓ |
| sweep_ave | ave_gblend_seed1024 | 1024 | 99 | 86.67 | 86.44 | ✓ |
| sweep_ave | ave_gblend_seed123 | 123 | 46 | 87.28 | 87.21 | ✓ |
| sweep_ave | ave_gblend_seed42 | 42 | 67 | 87.04 | 86.59 | ✓ |
| sweep_ave | ave_gblend_seed456 | 456 | 51 | 86.91 | 86.50 | ✓ |
| sweep_ave | ave_gblend_seed789 | 789 | 71 | 87.53 | 87.22 | ✓ |
| sweep_ave | ave_mmpareto_seed1024 | 1024 | 24 | 85.93 | 85.42 | ✓ |
| sweep_ave | ave_mmpareto_seed123 | 123 | 60 | 86.42 | 86.17 | ✓ |
| sweep_ave | ave_mmpareto_seed42 | 42 | 62 | 86.67 | 86.41 | ✓ |
| sweep_ave | ave_mmpareto_seed456 | 456 | 93 | 86.79 | 86.16 | ✓ |
| sweep_ave | ave_mmpareto_seed789 | 789 | 53 | 86.05 | 85.41 | ✓ |
| sweep_ave | ave_ogm_ge_seed1024 | 1024 | 69 | 86.79 | 86.48 | ✓ |
| sweep_ave | ave_ogm_ge_seed123 | 123 | 89 | 87.78 | 87.50 | ✓ |
| sweep_ave | ave_ogm_ge_seed42 | 42 | 61 | 86.42 | 86.10 | ✓ |
| sweep_ave | ave_ogm_ge_seed456 | 456 | 66 | 86.05 | 85.11 | ✓ |
| sweep_ave | ave_ogm_ge_seed789 | 789 | 81 | 87.78 | 87.20 | ✓ |
| sweep_ave_scratch | ave_scratch_baseline_seed1024 | 1024 | 75 | 67.28 | 64.97 | ✓ |
| sweep_ave_scratch | ave_scratch_baseline_seed123 | 123 | 97 | 67.28 | 65.39 | ✓ |
| sweep_ave_scratch | ave_scratch_baseline_seed42 | 42 | 67 | 68.52 | 66.37 | ✓ |
| sweep_ave_scratch | ave_scratch_baseline_seed456 | 456 | 51 | 67.41 | 65.38 | ✓ |
| sweep_ave_scratch | ave_scratch_baseline_seed789 | 789 | 46 | 67.28 | 64.66 | ✓ |
| sweep_ave_scratch | ave_scratch_boost_ogm_a075_seed1024 | 1024 | 83 | 63.46 | 60.83 | ✓ |
| sweep_ave_scratch | ave_scratch_boost_ogm_a075_seed123 | 123 | 94 | 62.10 | 58.80 | ✓ |
| sweep_ave_scratch | ave_scratch_boost_ogm_a075_seed456 | 456 | 93 | 63.46 | 60.91 | ✓ |
| sweep_ave_scratch | ave_scratch_boost_ogm_a075_seed789 | 789 | 93 | 62.47 | 59.91 | ✓ |
| sweep_ave_scratch | ave_scratch_boost_ogm_seed42 | 42 | 97 | 63.46 | 60.40 | ✓ |
| sweep_ave_scratch | ave_scratch_boost_only_seed1024 | 1024 | 75 | 68.02 | 65.92 | ✓ |
| sweep_ave_scratch | ave_scratch_boost_only_seed123 | 123 | 46 | 68.02 | 66.36 | ✓ |
| sweep_ave_scratch | ave_scratch_boost_only_seed42 | 42 | 44 | 68.89 | 66.51 | ✓ |
| sweep_ave_scratch | ave_scratch_boost_only_seed456 | 456 | 42 | 67.65 | 65.33 | ✓ |
| sweep_ave_scratch | ave_scratch_boost_only_seed789 | 789 | 55 | 67.28 | 64.83 | ✓ |
| sweep_boost_compose | boost_agm_seed1024 | 1024 | 57 | 56.45 | 56.33 | ✓ |
| sweep_boost_compose | boost_agm_seed123 | 123 | 72 | 61.29 | 61.55 | ✓ |
| sweep_boost_compose | boost_agm_seed42 | 42 | 60 | 57.93 | 57.83 | ✓ |
| sweep_boost_compose | boost_agm_seed456 | 456 | 62 | 56.45 | 56.41 | ✓ |
| sweep_boost_compose | boost_agm_seed789 | 789 | 73 | 58.74 | 58.81 | ✓ |
| sweep_boost_compose | boost_cggm_seed1024 | 1024 | 96 | 51.21 | 50.35 | ✓ |
| sweep_boost_compose | boost_cggm_seed123 | 123 | 71 | 50.81 | 50.07 | ✓ |
| sweep_boost_compose | boost_cggm_seed42 | 42 | 86 | 51.21 | 50.67 | ✓ |
| sweep_boost_compose | boost_cggm_seed456 | 456 | 97 | 48.52 | 47.59 | ✓ |
| sweep_boost_compose | boost_cggm_seed789 | 789 | 72 | 49.87 | 49.42 | ✓ |
| sweep_boost_compose | boost_gblend_seed1024 | 1024 | 70 | 61.83 | 62.74 | ✓ |
| sweep_boost_compose | boost_gblend_seed123 | 123 | 32 | 61.96 | 62.07 | ✓ |
| sweep_boost_compose | boost_gblend_seed42 | 42 | 71 | 61.29 | 61.93 | ✓ |
| sweep_boost_compose | boost_gblend_seed456 | 456 | 69 | 63.84 | 64.13 | ✓ |
| sweep_boost_compose | boost_gblend_seed789 | 789 | 69 | 61.02 | 60.70 | ✓ |
| sweep_boost_compose | boost_mmpareto_seed1024 | 1024 | 42 | 67.34 | 68.14 | ✓ |
| sweep_boost_compose | boost_mmpareto_seed123 | 123 | 100 | 67.07 | 67.55 | ✓ |
| sweep_boost_compose | boost_mmpareto_seed42 | 42 | 75 | 63.98 | 64.54 | ✓ |
| sweep_boost_compose | boost_mmpareto_seed456 | 456 | 66 | 65.19 | 65.63 | ✓ |
| sweep_boost_compose | boost_mmpareto_seed789 | 789 | 69 | 66.40 | 66.93 | ✓ |
| sweep_boost_compose_ave_food101 | ave_boost_agm_seed1024 | 1024 | 73 | 85.43 | 85.05 | ✓ |
| sweep_boost_compose_ave_food101 | ave_boost_agm_seed123 | 123 | 59 | 85.93 | 85.52 | ✓ |
| sweep_boost_compose_ave_food101 | ave_boost_agm_seed42 | 42 | 100 | 86.05 | 85.73 | ✓ |
| sweep_boost_compose_ave_food101 | ave_boost_agm_seed456 | 456 | 86 | 85.80 | 85.52 | ✓ |
| sweep_boost_compose_ave_food101 | ave_boost_agm_seed789 | 789 | 57 | 85.19 | 84.78 | ✓ |
| sweep_boost_compose_ave_food101 | ave_boost_cggm_seed1024 | 1024 | 98 | 76.30 | 74.66 | ✓ |
| sweep_boost_compose_ave_food101 | ave_boost_cggm_seed123 | 123 | 99 | 76.79 | 75.03 | ✓ |
| sweep_boost_compose_ave_food101 | ave_boost_cggm_seed42 | 42 | 79 | 76.42 | 74.26 | ✓ |
| sweep_boost_compose_ave_food101 | ave_boost_cggm_seed456 | 456 | 82 | 76.54 | 74.24 | ✓ |
| sweep_boost_compose_ave_food101 | ave_boost_cggm_seed789 | 789 | 76 | 76.30 | 73.70 | ✓ |
| sweep_boost_compose_ave_food101 | ave_boost_gblend_seed1024 | 1024 | 44 | 87.16 | 86.78 | ✓ |
| sweep_boost_compose_ave_food101 | ave_boost_gblend_seed123 | 123 | 76 | 87.16 | 86.86 | ✓ |
| sweep_boost_compose_ave_food101 | ave_boost_gblend_seed42 | 42 | 72 | 86.91 | 86.14 | ✓ |
| sweep_boost_compose_ave_food101 | ave_boost_gblend_seed456 | 456 | 68 | 86.05 | 85.86 | ✓ |
| sweep_boost_compose_ave_food101 | ave_boost_gblend_seed789 | 789 | 84 | 86.91 | 86.43 | ✓ |
| sweep_boost_compose_ave_food101 | ave_boost_mmpareto_seed1024 | 1024 | 74 | 87.53 | 87.21 | ✓ |
| sweep_boost_compose_ave_food101 | ave_boost_mmpareto_seed123 | 123 | 59 | 87.04 | 86.75 | ✓ |
| sweep_boost_compose_ave_food101 | ave_boost_mmpareto_seed42 | 42 | 74 | 86.30 | 85.81 | ✓ |
| sweep_boost_compose_ave_food101 | ave_boost_mmpareto_seed456 | 456 | 44 | 88.89 | 88.26 | ✓ |
| sweep_boost_compose_ave_food101 | ave_boost_mmpareto_seed789 | 789 | 70 | 87.16 | 86.26 | ✓ |
| sweep_boost_compose_ave_food101 | food101_boost_agm_seed1024 | 1024 | 97 | 83.92 | 83.84 | ✓ |
| sweep_boost_compose_ave_food101 | food101_boost_agm_seed123 | 123 | 100 | 83.82 | 83.75 | ✓ |
| sweep_boost_compose_ave_food101 | food101_boost_agm_seed42 | 42 | 85 | 83.69 | 83.61 | ✓ |
| sweep_boost_compose_ave_food101 | food101_boost_agm_seed456 | 456 | 86 | 84.05 | 83.97 | ✓ |
| sweep_boost_compose_ave_food101 | food101_boost_agm_seed789 | 789 | 96 | 84.05 | 83.97 | ✓ |
| sweep_boost_compose_ave_food101 | food101_boost_cggm_seed1024 | 1024 | 96 | 52.95 | 52.33 | ✓ |
| sweep_boost_compose_ave_food101 | food101_boost_cggm_seed123 | 123 | 99 | 53.06 | 52.49 | ✓ |
| sweep_boost_compose_ave_food101 | food101_boost_cggm_seed42 | 42 | 96 | 53.01 | 52.46 | ✓ |
| sweep_boost_compose_ave_food101 | food101_boost_cggm_seed456 | 456 | 97 | 52.76 | 52.17 | ✓ |
| sweep_boost_compose_ave_food101 | food101_boost_cggm_seed789 | 789 | 95 | 53.21 | 52.68 | ✓ |
| sweep_boost_compose_ave_food101 | food101_boost_gblend_seed1024 | 1024 | 92 | 85.13 | 85.00 | ✓ |
| sweep_boost_compose_ave_food101 | food101_boost_gblend_seed123 | 123 | 91 | 85.27 | 85.17 | ✓ |
| sweep_boost_compose_ave_food101 | food101_boost_gblend_seed42 | 42 | 90 | 85.42 | 85.33 | ✓ |
| sweep_boost_compose_ave_food101 | food101_boost_gblend_seed456 | 456 | 90 | 85.15 | 85.05 | ✓ |
| sweep_boost_compose_ave_food101 | food101_boost_gblend_seed789 | 789 | 95 | 85.13 | 85.05 | ✓ |
| sweep_boost_compose_ave_food101 | food101_boost_mmpareto_seed1024 | 1024 | 95 | 85.54 | 85.45 | ✓ |
| sweep_boost_compose_ave_food101 | food101_boost_mmpareto_seed123 | 123 | 91 | 85.63 | 85.54 | ✓ |
| sweep_boost_compose_ave_food101 | food101_boost_mmpareto_seed42 | 42 | 90 | 85.41 | 85.32 | ✓ |
| sweep_boost_compose_ave_food101 | food101_boost_mmpareto_seed456 | 456 | 96 | 85.78 | 85.71 | ✓ |
| sweep_boost_compose_ave_food101 | food101_boost_mmpareto_seed789 | 789 | 99 | 85.58 | 85.50 | ✓ |
| sweep_cggm | ave_cggm_seed1024 | 1024 | 86 | 76.79 | 74.99 | ✓ |
| sweep_cggm | ave_cggm_seed123 | 123 | 84 | 76.17 | 74.34 | ✓ |
| sweep_cggm | ave_cggm_seed42 | 42 | 85 | 76.42 | 74.45 | ✓ |
| sweep_cggm | ave_cggm_seed456 | 456 | 90 | 77.41 | 75.60 | ✓ |
| sweep_cggm | ave_cggm_seed789 | 789 | 98 | 76.79 | 74.86 | ✓ |
| sweep_cggm | cremad_cggm_seed1024 | 1024 | 81 | 48.79 | 47.55 | ✓ |
| sweep_cggm | cremad_cggm_seed123 | 123 | 94 | 50.00 | 49.41 | ✓ |
| sweep_cggm | cremad_cggm_seed42 | 42 | 98 | 48.66 | 47.40 | ✓ |
| sweep_cggm | cremad_cggm_seed456 | 456 | 96 | 51.75 | 51.03 | ✓ |
| sweep_cggm | cremad_cggm_seed789 | 789 | 74 | 51.88 | 51.38 | ✓ |
| sweep_cggm | ks_cggm_seed1024 | 1024 | 75 | 73.53 | 73.16 | ✓ |
| sweep_cggm | ks_cggm_seed123 | 123 | 80 | 73.29 | 72.62 | ✓ |
| sweep_cggm | ks_cggm_seed42 | 42 | 81 | 73.05 | 72.50 | ✓ |
| sweep_cggm | ks_cggm_seed456 | 456 | 66 | 72.72 | 71.96 | ✓ |
| sweep_cggm | ks_cggm_seed789 | 789 | 96 | 73.29 | 72.93 | ✓ |
| sweep_cggm | mosei_cggm_seed1024 | 1024 | 88 | 68.05 | 47.56 | ✓ |
| sweep_cggm | mosei_cggm_seed123 | 123 | 76 | 67.18 | 46.64 | ✓ |
| sweep_cggm | mosei_cggm_seed42 | 42 | 66 | 68.49 | 47.84 | ✓ |
| sweep_cggm | mosei_cggm_seed456 | 456 | 78 | 68.27 | 47.86 | ✓ |
| sweep_cggm | mosei_cggm_seed789 | 789 | 77 | 68.27 | 47.53 | ✓ |
| sweep_cggm | mosi_cggm_seed1024 | 1024 | 1 | 59.48 | 37.29 | ✓ |
| sweep_cggm | mosi_cggm_seed123 | 123 | 3 | 58.75 | 38.28 | ✓ |
| sweep_cggm | mosi_cggm_seed42 | 42 | 5 | 59.77 | 38.40 | ✓ |
| sweep_cggm | mosi_cggm_seed456 | 456 | 4 | 59.62 | 37.35 | ✓ |
| sweep_cggm | mosi_cggm_seed789 | 789 | 4 | 59.62 | 37.35 | ✓ |
| sweep_cggm_hp | cremad_cggm_r0.5_l0.0_seed42 | 42 | 89 | 46.77 | 45.87 | ✓ |
| sweep_cggm_hp | cremad_cggm_r0.5_l0.1_seed42 | 42 | 89 | 46.77 | 45.87 | ✓ |
| sweep_cggm_hp | cremad_cggm_r0.8_l0.0_seed42 | 42 | 86 | 49.33 | 48.04 | ✓ |
| sweep_cggm_hp | cremad_cggm_r0.8_l0.1_seed42 | 42 | 86 | 49.33 | 48.04 | ✓ |
| sweep_cggm_hp | cremad_cggm_r1.0_l0.0_seed42 | 42 | 93 | 50.54 | 49.97 | ✓ |
| sweep_cggm_hp | cremad_cggm_r1.0_l0.1_seed42 | 42 | 93 | 50.54 | 49.97 | ✓ |
| sweep_cremad_opm | cremad_baseline_opmarch_seed1024 | 1024 | 90 | 57.80 | 57.87 | ✓ |
| sweep_cremad_opm | cremad_baseline_opmarch_seed123 | 123 | 93 | 59.41 | 59.19 | ✓ |
| sweep_cremad_opm | cremad_baseline_opmarch_seed42 | 42 | 51 | 60.35 | 60.57 | ✓ |
| sweep_cremad_opm | cremad_baseline_opmarch_seed456 | 456 | 94 | 60.62 | 60.81 | ✓ |
| sweep_cremad_opm | cremad_baseline_opmarch_seed789 | 789 | 40 | 57.80 | 57.45 | ✓ |
| sweep_cremad_opm | cremad_boost_ogm_opmarch_seed1024 | 1024 | 75 | 70.97 | 71.37 | ✓ |
| sweep_cremad_opm | cremad_boost_ogm_opmarch_seed123 | 123 | 100 | 72.58 | 72.92 | ✓ |
| sweep_cremad_opm | cremad_boost_ogm_opmarch_seed42 | 42 | 94 | 72.45 | 72.79 | ✓ |
| sweep_cremad_opm | cremad_boost_ogm_opmarch_seed456 | 456 | 100 | 72.31 | 72.76 | ✓ |
| sweep_cremad_opm | cremad_boost_ogm_opmarch_seed789 | 789 | 77 | 70.56 | 70.98 | ✓ |
| sweep_cremad_opm | cremad_ogm_opmarch_seed1024 | 1024 | 100 | 64.52 | 64.82 | ✓ |
| sweep_cremad_opm | cremad_ogm_opmarch_seed123 | 123 | 49 | 63.71 | 63.53 | ✓ |
| sweep_cremad_opm | cremad_ogm_opmarch_seed42 | 42 | 76 | 63.31 | 63.64 | ✓ |
| sweep_cremad_opm | cremad_ogm_opmarch_seed456 | 456 | 70 | 63.71 | 64.28 | ✓ |
| sweep_cremad_opm | cremad_ogm_opmarch_seed789 | 789 | 86 | 63.44 | 63.80 | ✓ |
| sweep_cremad_opm | cremad_opm_ogm_seed1024 | 1024 | 74 | 70.56 | 70.92 | ✓ |
| sweep_cremad_opm | cremad_opm_ogm_seed123 | 123 | 77 | 71.51 | 71.88 | ✓ |
| sweep_cremad_opm | cremad_opm_ogm_seed42 | 42 | 90 | 70.30 | 70.59 | ✓ |
| sweep_cremad_opm | cremad_opm_ogm_seed456 | 456 | 99 | 69.35 | 69.71 | ✓ |
| sweep_cremad_opm | cremad_opm_ogm_seed789 | 789 | 92 | 67.47 | 68.06 | ✓ |
| sweep_cremad_opm | cremad_opm_seed1024 | 1024 | 96 | 65.46 | 65.90 | ✓ |
| sweep_cremad_opm | cremad_opm_seed123 | 123 | 96 | 65.05 | 65.74 | ✓ |
| sweep_cremad_opm | cremad_opm_seed42 | 42 | 93 | 65.46 | 66.00 | ✓ |
| sweep_cremad_opm | cremad_opm_seed456 | 456 | 98 | 65.73 | 66.17 | ✓ |
| sweep_cremad_opm | cremad_opm_seed789 | 789 | 96 | 65.46 | 66.01 | ✓ |
| sweep_food101 | food101_agm_seed1024 | 1024 | 93 | 84.39 | 84.31 | ✓ |
| sweep_food101 | food101_agm_seed123 | 123 | 86 | 84.22 | 84.13 | ✓ |
| sweep_food101 | food101_agm_seed42 | 42 | 100 | 83.97 | 83.90 | ✓ |
| sweep_food101 | food101_agm_seed456 | 456 | 84 | 83.97 | 83.88 | ✓ |
| sweep_food101 | food101_agm_seed789 | 789 | 93 | 84.14 | 84.06 | ✓ |
| sweep_food101 | food101_baseline_seed1024 | 1024 | 99 | 85.64 | 85.55 | ✓ |
| sweep_food101 | food101_baseline_seed123 | 123 | 100 | 85.84 | 85.75 | ✓ |
| sweep_food101 | food101_baseline_seed42 | 42 | 100 | 85.49 | 85.42 | ✓ |
| sweep_food101 | food101_baseline_seed456 | 456 | 96 | 85.83 | 85.73 | ✓ |
| sweep_food101 | food101_baseline_seed789 | 789 | 98 | 85.95 | 85.88 | ✓ |
| sweep_food101 | food101_boost_ogm_a075_seed1024 | 1024 | 92 | 84.07 | 83.97 | ✓ |
| sweep_food101 | food101_boost_ogm_a075_seed123 | 123 | 94 | 83.95 | 83.86 | ✓ |
| sweep_food101 | food101_boost_ogm_a075_seed42 | 42 | 94 | 84.00 | 83.91 | ✓ |
| sweep_food101 | food101_boost_ogm_a075_seed456 | 456 | 94 | 84.13 | 84.03 | ✓ |
| sweep_food101 | food101_boost_ogm_a075_seed789 | 789 | 100 | 84.39 | 84.30 | ✓ |
| sweep_food101 | food101_boost_only_seed1024 | 1024 | 99 | 85.55 | 85.46 | ✓ |
| sweep_food101 | food101_boost_only_seed123 | 123 | 84 | 85.84 | 85.76 | ✓ |
| sweep_food101 | food101_boost_only_seed42 | 42 | 100 | 85.36 | 85.27 | ✓ |
| sweep_food101 | food101_boost_only_seed456 | 456 | 92 | 85.56 | 85.47 | ✓ |
| sweep_food101 | food101_boost_only_seed789 | 789 | 100 | 85.70 | 85.60 | ✓ |
| sweep_food101 | food101_cggm_seed1024 | 1024 | 100 | 48.62 | 47.87 | ✓ |
| sweep_food101 | food101_cggm_seed123 | 123 | 97 | 48.53 | 47.80 | ✓ |
| sweep_food101 | food101_cggm_seed42 | 42 | 98 | 48.40 | 47.63 | ✓ |
| sweep_food101 | food101_cggm_seed456 | 456 | 97 | 48.46 | 47.71 | ✓ |
| sweep_food101 | food101_cggm_seed789 | 789 | 98 | 48.94 | 48.22 | ✓ |
| sweep_food101 | food101_gblend_seed1024 | 1024 | 92 | 85.47 | 85.38 | ✓ |
| sweep_food101 | food101_gblend_seed123 | 123 | 94 | 85.36 | 85.26 | ✓ |
| sweep_food101 | food101_gblend_seed42 | 42 | 97 | 85.08 | 84.97 | ✓ |
| sweep_food101 | food101_gblend_seed456 | 456 | 95 | 85.30 | 85.20 | ✓ |
| sweep_food101 | food101_gblend_seed789 | 789 | 93 | 85.36 | 85.27 | ✓ |
| sweep_food101 | food101_mmpareto_seed1024 | 1024 | 87 | 85.73 | 85.65 | ✓ |
| sweep_food101 | food101_mmpareto_seed123 | 123 | 98 | 85.23 | 85.15 | ✓ |
| sweep_food101 | food101_mmpareto_seed42 | 42 | 97 | 85.80 | 85.71 | ✓ |
| sweep_food101 | food101_mmpareto_seed456 | 456 | 99 | 85.78 | 85.71 | ✓ |
| sweep_food101 | food101_mmpareto_seed789 | 789 | 93 | 85.87 | 85.79 | ✓ |
| sweep_food101 | food101_ogm_ge_seed1024 | 1024 | 90 | 84.45 | 84.37 | ✓ |
| sweep_food101 | food101_ogm_ge_seed123 | 123 | 87 | 84.12 | 84.02 | ✓ |
| sweep_food101 | food101_ogm_ge_seed42 | 42 | 94 | 84.00 | 83.89 | ✓ |
| sweep_food101 | food101_ogm_ge_seed456 | 456 | 98 | 84.21 | 84.10 | ✓ |
| sweep_food101 | food101_ogm_ge_seed789 | 789 | 100 | 84.33 | 84.24 | ✓ |
| sweep_hp | 3f_hp_alpha0.25 | - | 84 | 68.55 | 69.09 | ✓ |
| sweep_hp | 3f_hp_alpha0.5 | - | 82 | 69.76 | 70.08 | ✓ |
| sweep_hp | 3f_hp_alpha1.0 | - | 54 | 70.30 | 70.40 | ✓ |
| sweep_hp | 3f_hp_alpha1.5 | - | 54 | 70.30 | 70.40 | ✓ |
| sweep_hp | 3f_hp_ema0.1 | - | 78 | 71.64 | 71.83 | ✓ |
| sweep_hp | 3f_hp_ema0.5 | - | 78 | 71.64 | 71.83 | ✓ |
| sweep_hp | 3f_hp_ema0.7 | - | 78 | 71.64 | 71.83 | ✓ |
| sweep_hp | 3f_hp_smax1.5 | - | 82 | 69.76 | 70.08 | ✓ |
| sweep_hp | 3f_hp_smax3.0 | - | 78 | 71.64 | 71.83 | ✓ |
| sweep_k_ablation | 3f_boost_ogm_K100_seed42 | 42 | 57 | 71.24 | 71.35 | ✓ |
| sweep_k_ablation | 3f_boost_ogm_K10_seed42 | 42 | 72 | 70.83 | 71.27 | ✓ |
| sweep_k_ablation | 3f_boost_ogm_K1_seed42 | 42 | 99 | 71.24 | 71.64 | ✓ |
| sweep_k_ablation | 3f_boost_ogm_K50_seed42 | 42 | 62 | 71.24 | 71.90 | ✓ |
| sweep_k_ablation | 3f_boost_ogm_K5_seed42 | 42 | 78 | 71.64 | 71.83 | ✓ |
| sweep_ks | ks_agm_seed1024 | 1024 | 21 | 77.52 | 76.97 | ✓ |
| sweep_ks | ks_agm_seed123 | 123 | 20 | 78.01 | 77.42 | ✓ |
| sweep_ks | ks_agm_seed42 | 42 | 22 | 77.44 | 76.73 | ✓ |
| sweep_ks | ks_agm_seed456 | 456 | 23 | 78.18 | 77.26 | ✓ |
| sweep_ks | ks_agm_seed789 | 789 | 13 | 78.09 | 77.83 | ✓ |
| sweep_ks | ks_baseline_seed1024 | 1024 | 13 | 79.23 | 78.74 | ✓ |
| sweep_ks | ks_baseline_seed123 | 123 | 24 | 78.99 | 77.98 | ✓ |
| sweep_ks | ks_baseline_seed42 | 42 | 15 | 78.58 | 77.85 | ✓ |
| sweep_ks | ks_baseline_seed456 | 456 | 16 | 78.75 | 78.22 | ✓ |
| sweep_ks | ks_baseline_seed789 | 789 | 13 | 79.72 | 79.20 | ✓ |
| sweep_ks | ks_boost_ogm_seed1024 | 1024 | 13 | 76.22 | 75.49 | ✓ |
| sweep_ks | ks_boost_ogm_seed123 | 123 | 17 | 78.18 | 77.15 | ✓ |
| sweep_ks | ks_boost_ogm_seed42 | 42 | 17 | 77.52 | 76.46 | ✓ |
| sweep_ks | ks_boost_ogm_seed456 | 456 | 15 | 77.20 | 76.43 | ✓ |
| sweep_ks | ks_boost_ogm_seed789 | 789 | 27 | 77.52 | 76.74 | ✓ |
| sweep_ks | ks_boost_only_seed1024 | 1024 | 13 | 78.58 | 78.30 | ✓ |
| sweep_ks | ks_boost_only_seed123 | 123 | 12 | 80.13 | 79.23 | ✓ |
| sweep_ks | ks_boost_only_seed42 | 42 | 12 | 80.29 | 79.31 | ✓ |
| sweep_ks | ks_boost_only_seed456 | 456 | 14 | 79.15 | 78.55 | ✓ |
| sweep_ks | ks_boost_only_seed789 | 789 | 19 | 77.69 | 76.64 | ✓ |
| sweep_ks | ks_gblend_seed1024 | 1024 | 10 | 77.36 | 76.67 | ✓ |
| sweep_ks | ks_gblend_seed123 | 123 | 15 | 79.23 | 78.54 | ✓ |
| sweep_ks | ks_gblend_seed42 | 42 | 17 | 76.38 | 75.68 | ✓ |
| sweep_ks | ks_gblend_seed456 | 456 | 12 | 78.83 | 78.13 | ✓ |
| sweep_ks | ks_gblend_seed789 | 789 | 13 | 76.95 | 76.75 | ✓ |
| sweep_ks | ks_mmpareto_seed1024 | 1024 | 91 | 78.09 | 76.93 | ✓ |
| sweep_ks | ks_mmpareto_seed123 | 123 | 96 | 78.01 | 76.46 | ✓ |
| sweep_ks | ks_mmpareto_seed42 | 42 | 10 | 78.75 | 77.97 | ✓ |
| sweep_ks | ks_mmpareto_seed456 | 456 | 44 | 77.69 | 76.73 | ✓ |
| sweep_ks | ks_mmpareto_seed789 | 789 | 23 | 78.50 | 77.23 | ✓ |
| sweep_ks | ks_ogmge_seed1024 | 1024 | 20 | 75.98 | 75.09 | ✓ |
| sweep_ks | ks_ogmge_seed123 | 123 | 18 | 77.44 | 76.77 | ✓ |
| sweep_ks | ks_ogmge_seed42 | 42 | 15 | 76.79 | 76.14 | ✓ |
| sweep_ks | ks_ogmge_seed456 | 456 | 22 | 78.26 | 77.37 | ✓ |
| sweep_ks | ks_ogmge_seed789 | 789 | 24 | 77.77 | 76.83 | ✓ |
| sweep_mosei | mosei_agm_seed1024 | 1024 | 4 | 70.02 | 61.17 | ✓ |
| sweep_mosei | mosei_agm_seed123 | 123 | 72 | 68.93 | 59.85 | ✓ |
| sweep_mosei | mosei_agm_seed42 | 42 | 2 | 68.27 | 56.06 | ✓ |
| sweep_mosei | mosei_agm_seed456 | 456 | 2 | 69.15 | 57.21 | ✓ |
| sweep_mosei | mosei_agm_seed789 | 789 | 2 | 70.02 | 60.69 | ✓ |
| sweep_mosei | mosei_baseline_seed1024 | 1024 | 41 | 70.90 | 59.98 | ✓ |
| sweep_mosei | mosei_baseline_seed123 | 123 | 36 | 70.46 | 59.90 | ✓ |
| sweep_mosei | mosei_baseline_seed42 | 42 | 46 | 70.46 | 59.75 | ✓ |
| sweep_mosei | mosei_baseline_seed456 | 456 | 10 | 70.24 | 61.14 | ✓ |
| sweep_mosei | mosei_baseline_seed789 | 789 | 64 | 70.02 | 59.46 | ✓ |
| sweep_mosei | mosei_boost_ogm_a075_seed1024 | 1024 | 3 | 72.21 | 63.94 | ✓ |
| sweep_mosei | mosei_boost_ogm_a075_seed123 | 123 | 4 | 73.09 | 64.17 | ✓ |
| sweep_mosei | mosei_boost_ogm_a075_seed42 | 42 | 3 | 73.30 | 59.77 | ✓ |
| sweep_mosei | mosei_boost_ogm_a075_seed456 | 456 | 59 | 71.77 | 61.51 | ✓ |
| sweep_mosei | mosei_boost_ogm_a075_seed789 | 789 | 3 | 71.77 | 62.41 | ✓ |
| sweep_mosei | mosei_boost_only_seed1024 | 1024 | 26 | 69.58 | 59.84 | ✓ |
| sweep_mosei | mosei_boost_only_seed123 | 123 | 2 | 69.80 | 61.08 | ✓ |
| sweep_mosei | mosei_boost_only_seed42 | 42 | 2 | 70.24 | 58.76 | ✓ |
| sweep_mosei | mosei_boost_only_seed456 | 456 | 78 | 70.90 | 62.37 | ✓ |
| sweep_mosei | mosei_boost_only_seed789 | 789 | 14 | 68.49 | 58.21 | ✓ |
| sweep_mosei | mosei_gblend_seed1024 | 1024 | 60 | 70.68 | 61.23 | ✓ |
| sweep_mosei | mosei_gblend_seed123 | 123 | 2 | 70.68 | 56.15 | ✓ |
| sweep_mosei | mosei_gblend_seed42 | 42 | 25 | 68.93 | 59.77 | ✓ |
| sweep_mosei | mosei_gblend_seed456 | 456 | 2 | 70.46 | 57.93 | ✓ |
| sweep_mosei | mosei_gblend_seed789 | 789 | 2 | 70.02 | 57.90 | ✓ |
| sweep_mosei | mosei_mmpareto_seed1024 | 1024 | 2 | 71.33 | 60.36 | ✓ |
| sweep_mosei | mosei_mmpareto_seed123 | 123 | 17 | 69.15 | 53.49 | ✓ |
| sweep_mosei | mosei_mmpareto_seed42 | 42 | 36 | 70.02 | 60.83 | ✓ |
| sweep_mosei | mosei_mmpareto_seed456 | 456 | 2 | 70.02 | 57.77 | ✓ |
| sweep_mosei | mosei_mmpareto_seed789 | 789 | 27 | 70.46 | 61.48 | ✓ |
| sweep_mosei | mosei_ogm_ge_seed1024 | 1024 | 2 | 72.21 | 59.75 | ✓ |
| sweep_mosei | mosei_ogm_ge_seed123 | 123 | 4 | 73.09 | 63.83 | ✓ |
| sweep_mosei | mosei_ogm_ge_seed42 | 42 | 3 | 73.30 | 59.63 | ✓ |
| sweep_mosei | mosei_ogm_ge_seed456 | 456 | 29 | 72.43 | 62.47 | ✓ |
| sweep_mosei | mosei_ogm_ge_seed789 | 789 | 3 | 71.33 | 61.60 | ✓ |
| sweep_mosi | mosi_agm_seed1024 | 1024 | 12 | 72.30 | 69.95 | ✓ |
| sweep_mosi | mosi_agm_seed123 | 123 | 11 | 71.57 | 68.90 | ✓ |
| sweep_mosi | mosi_agm_seed42 | 42 | 13 | 73.03 | 71.73 | ✓ |
| sweep_mosi | mosi_agm_seed456 | 456 | 13 | 72.30 | 68.66 | ✓ |
| sweep_mosi | mosi_agm_seed789 | 789 | 19 | 73.32 | 71.33 | ✓ |
| sweep_mosi | mosi_baseline_seed1024 | 1024 | 14 | 72.30 | 70.58 | ✓ |
| sweep_mosi | mosi_baseline_seed123 | 123 | 30 | 71.87 | 70.39 | ✓ |
| sweep_mosi | mosi_baseline_seed42 | 42 | 10 | 73.18 | 70.52 | ✓ |
| sweep_mosi | mosi_baseline_seed456 | 456 | 22 | 72.74 | 70.45 | ✓ |
| sweep_mosi | mosi_baseline_seed789 | 789 | 15 | 72.01 | 68.87 | ✓ |
| sweep_mosi | mosi_boost_ogm_seed1024 | 1024 | 14 | 72.45 | 69.97 | ✓ |
| sweep_mosi | mosi_boost_ogm_seed123 | 123 | 17 | 70.99 | 67.77 | ✓ |
| sweep_mosi | mosi_boost_ogm_seed42 | 42 | 10 | 73.47 | 69.84 | ✓ |
| sweep_mosi | mosi_boost_ogm_seed456 | 456 | 20 | 73.62 | 71.45 | ✓ |
| sweep_mosi | mosi_boost_ogm_seed789 | 789 | 36 | 72.45 | 70.62 | ✓ |
| sweep_mosi | mosi_boost_only_seed1024 | 1024 | 14 | 72.74 | 70.89 | ✓ |
| sweep_mosi | mosi_boost_only_seed123 | 123 | 5 | 70.55 | 64.04 | ✓ |
| sweep_mosi | mosi_boost_only_seed42 | 42 | 10 | 72.74 | 69.95 | ✓ |
| sweep_mosi | mosi_boost_only_seed456 | 456 | 15 | 71.57 | 70.36 | ✓ |
| sweep_mosi | mosi_boost_only_seed789 | 789 | 19 | 71.87 | 69.86 | ✓ |
| sweep_mosi | mosi_cggm_seed42 | 42 | 5 | 59.77 | 38.40 | ✓ |
| sweep_mosi | mosi_gblend_seed1024 | 1024 | 15 | 72.89 | 70.63 | ✓ |
| sweep_mosi | mosi_gblend_seed123 | 123 | 16 | 72.16 | 69.76 | ✓ |
| sweep_mosi | mosi_gblend_seed42 | 42 | 10 | 73.18 | 72.01 | ✓ |
| sweep_mosi | mosi_gblend_seed456 | 456 | 10 | 72.30 | 69.62 | ✓ |
| sweep_mosi | mosi_gblend_seed789 | 789 | 11 | 71.28 | 70.48 | ✓ |
| sweep_mosi | mosi_mmpareto_seed1024 | 1024 | 12 | 73.32 | 72.11 | ✓ |
| sweep_mosi | mosi_mmpareto_seed123 | 123 | 14 | 72.16 | 70.22 | ✓ |
| sweep_mosi | mosi_mmpareto_seed42 | 42 | 13 | 73.91 | 71.01 | ✓ |
| sweep_mosi | mosi_mmpareto_seed456 | 456 | 22 | 72.01 | 70.18 | ✓ |
| sweep_mosi | mosi_mmpareto_seed789 | 789 | 19 | 72.01 | 69.94 | ✓ |
| sweep_mosi | mosi_ogmge_seed1024 | 1024 | 14 | 73.03 | 70.61 | ✓ |
| sweep_mosi | mosi_ogmge_seed123 | 123 | 15 | 71.43 | 68.02 | ✓ |
| sweep_mosi | mosi_ogmge_seed42 | 42 | 10 | 73.32 | 68.89 | ✓ |
| sweep_mosi | mosi_ogmge_seed456 | 456 | 20 | 73.76 | 71.91 | ✓ |
| sweep_mosi | mosi_ogmge_seed789 | 789 | 76 | 71.87 | 71.16 | ✓ |
| sweep_sarcasm | sarcasm_agm_seed1024 | 1024 | 10 | 82.57 | 82.01 | ✓ |
| sweep_sarcasm | sarcasm_agm_seed123 | 123 | 9 | 82.19 | 81.35 | ✓ |
| sweep_sarcasm | sarcasm_agm_seed42 | 42 | 17 | 82.15 | 81.48 | ✓ |
| sweep_sarcasm | sarcasm_agm_seed456 | 456 | 12 | 82.44 | 81.83 | ✓ |
| sweep_sarcasm | sarcasm_agm_seed789 | 789 | 7 | 82.27 | 81.63 | ✓ |
| sweep_sarcasm | sarcasm_baseline_seed1024 | 1024 | 12 | 82.48 | 82.10 | ✓ |
| sweep_sarcasm | sarcasm_baseline_seed123 | 123 | 12 | 82.40 | 81.82 | ✓ |
| sweep_sarcasm | sarcasm_baseline_seed42 | 42 | 16 | 82.65 | 81.95 | ✓ |
| sweep_sarcasm | sarcasm_baseline_seed456 | 456 | 42 | 82.15 | 81.62 | ✓ |
| sweep_sarcasm | sarcasm_baseline_seed789 | 789 | 18 | 82.32 | 81.54 | ✓ |
| sweep_sarcasm | sarcasm_boost_ogm_a075_seed1024 | 1024 | 13 | 81.82 | 81.16 | ✓ |
| sweep_sarcasm | sarcasm_boost_ogm_a075_seed123 | 123 | 20 | 81.78 | 81.22 | ✓ |
| sweep_sarcasm | sarcasm_boost_ogm_a075_seed42 | 42 | 19 | 81.69 | 81.28 | ✓ |
| sweep_sarcasm | sarcasm_boost_ogm_a075_seed456 | 456 | 18 | 81.86 | 81.26 | ✓ |
| sweep_sarcasm | sarcasm_boost_ogm_a075_seed789 | 789 | 25 | 81.65 | 81.01 | ✓ |
| sweep_sarcasm | sarcasm_boost_only_seed1024 | 1024 | 13 | 82.19 | 81.63 | ✓ |
| sweep_sarcasm | sarcasm_boost_only_seed123 | 123 | 13 | 82.94 | 82.28 | ✓ |
| sweep_sarcasm | sarcasm_boost_only_seed42 | 42 | 16 | 82.77 | 82.18 | ✓ |
| sweep_sarcasm | sarcasm_boost_only_seed456 | 456 | 10 | 82.23 | 81.47 | ✓ |
| sweep_sarcasm | sarcasm_boost_only_seed789 | 789 | 38 | 82.07 | 81.60 | ✓ |
| sweep_sarcasm | sarcasm_cggm_seed1024 | 1024 | 75 | 80.32 | 79.73 | ✓ |
| sweep_sarcasm | sarcasm_cggm_seed123 | 123 | 61 | 80.99 | 80.26 | ✓ |
| sweep_sarcasm | sarcasm_cggm_seed42 | 42 | 42 | 80.99 | 80.29 | ✓ |
| sweep_sarcasm | sarcasm_cggm_seed456 | 456 | 36 | 80.49 | 79.81 | ✓ |
| sweep_sarcasm | sarcasm_cggm_seed789 | 789 | 79 | 80.78 | 80.26 | ✓ |
| sweep_sarcasm | sarcasm_gblend_seed1024 | 1024 | 7 | 82.69 | 82.02 | ✓ |
| sweep_sarcasm | sarcasm_gblend_seed123 | 123 | 7 | 82.11 | 81.27 | ✓ |
| sweep_sarcasm | sarcasm_gblend_seed42 | 42 | 23 | 82.32 | 81.69 | ✓ |
| sweep_sarcasm | sarcasm_gblend_seed456 | 456 | 41 | 82.03 | 81.34 | ✓ |
| sweep_sarcasm | sarcasm_gblend_seed789 | 789 | 16 | 82.61 | 81.90 | ✓ |
| sweep_sarcasm | sarcasm_mmpareto_seed1024 | 1024 | 9 | 82.23 | 81.59 | ✓ |
| sweep_sarcasm | sarcasm_mmpareto_seed123 | 123 | 9 | 82.07 | 81.30 | ✓ |
| sweep_sarcasm | sarcasm_mmpareto_seed42 | 42 | 20 | 82.44 | 81.86 | ✓ |
| sweep_sarcasm | sarcasm_mmpareto_seed456 | 456 | 18 | 82.32 | 81.69 | ✓ |
| sweep_sarcasm | sarcasm_mmpareto_seed789 | 789 | 21 | 82.27 | 81.60 | ✓ |
| sweep_sarcasm | sarcasm_ogm_ge_seed1024 | 1024 | 13 | 81.78 | 81.13 | ✓ |
| sweep_sarcasm | sarcasm_ogm_ge_seed123 | 123 | 13 | 82.03 | 81.53 | ✓ |
| sweep_sarcasm | sarcasm_ogm_ge_seed42 | 42 | 12 | 82.07 | 81.54 | ✓ |
| sweep_sarcasm | sarcasm_ogm_ge_seed456 | 456 | 10 | 81.74 | 80.61 | ✓ |
| sweep_sarcasm | sarcasm_ogm_ge_seed789 | 789 | 12 | 81.44 | 80.66 | ✓ |
| sweep_trial | trial_gradclip | - | 78 | 71.64 | 71.83 | ✓ |
| sweep_twitter | twitter_agm_seed1024 | 1024 | 3 | 66.44 | 51.33 | ✓ |
| sweep_twitter | twitter_agm_seed123 | 123 | 7 | 66.35 | 53.82 | ✓ |
| sweep_twitter | twitter_agm_seed42 | 42 | 5 | 66.54 | 56.53 | ✓ |
| sweep_twitter | twitter_agm_seed456 | 456 | 8 | 67.60 | 52.77 | ✓ |
| sweep_twitter | twitter_agm_seed789 | 789 | 5 | 67.02 | 56.31 | ✓ |
| sweep_twitter | twitter_baseline_seed1024 | 1024 | 5 | 66.92 | 54.56 | ✓ |
| sweep_twitter | twitter_baseline_seed123 | 123 | 5 | 66.73 | 51.43 | ✓ |
| sweep_twitter | twitter_baseline_seed42 | 42 | 3 | 65.86 | 52.08 | ✓ |
| sweep_twitter | twitter_baseline_seed456 | 456 | 7 | 66.06 | 43.47 | ✓ |
| sweep_twitter | twitter_baseline_seed789 | 789 | 11 | 67.41 | 55.13 | ✓ |
| sweep_twitter | twitter_boost_ogm_a075_seed1024 | 1024 | 5 | 66.83 | 48.11 | ✓ |
| sweep_twitter | twitter_boost_ogm_a075_seed123 | 123 | 57 | 66.63 | 50.30 | ✓ |
| sweep_twitter | twitter_boost_ogm_a075_seed42 | 42 | 3 | 65.67 | 53.83 | ✓ |
| sweep_twitter | twitter_boost_ogm_a075_seed456 | 456 | 74 | 66.63 | 53.30 | ✓ |
| sweep_twitter | twitter_boost_ogm_a075_seed789 | 789 | 5 | 67.21 | 58.05 | ✓ |
| sweep_twitter | twitter_boost_only_seed1024 | 1024 | 5 | 67.02 | 57.25 | ✓ |
| sweep_twitter | twitter_boost_only_seed123 | 123 | 4 | 67.12 | 58.22 | ✓ |
| sweep_twitter | twitter_boost_only_seed42 | 42 | 90 | 66.54 | 52.45 | ✓ |
| sweep_twitter | twitter_boost_only_seed456 | 456 | 7 | 66.06 | 46.51 | ✓ |
| sweep_twitter | twitter_boost_only_seed789 | 789 | 12 | 67.50 | 56.58 | ✓ |
| sweep_twitter | twitter_cggm_seed1024 | 1024 | 69 | 62.49 | 39.17 | ✓ |
| sweep_twitter | twitter_cggm_seed123 | 123 | 67 | 62.30 | 38.97 | ✓ |
| sweep_twitter | twitter_cggm_seed42 | 42 | 79 | 63.07 | 39.37 | ✓ |
| sweep_twitter | twitter_cggm_seed456 | 456 | 79 | 62.10 | 38.52 | ✓ |
| sweep_twitter | twitter_cggm_seed789 | 789 | 99 | 62.10 | 39.09 | ✓ |
| sweep_twitter | twitter_gblend_seed1024 | 1024 | 5 | 67.12 | 55.36 | ✓ |
| sweep_twitter | twitter_gblend_seed123 | 123 | 11 | 66.25 | 58.81 | ✓ |
| sweep_twitter | twitter_gblend_seed42 | 42 | 5 | 67.31 | 56.94 | ✓ |
| sweep_twitter | twitter_gblend_seed456 | 456 | 3 | 65.77 | 56.18 | ✓ |
| sweep_twitter | twitter_gblend_seed789 | 789 | 6 | 66.83 | 54.15 | ✓ |
| sweep_twitter | twitter_mmpareto_seed1024 | 1024 | 9 | 66.44 | 56.64 | ✓ |
| sweep_twitter | twitter_mmpareto_seed123 | 123 | 5 | 66.63 | 58.13 | ✓ |
| sweep_twitter | twitter_mmpareto_seed42 | 42 | 6 | 66.25 | 54.01 | ✓ |
| sweep_twitter | twitter_mmpareto_seed456 | 456 | 8 | 67.31 | 50.82 | ✓ |
| sweep_twitter | twitter_mmpareto_seed789 | 789 | 6 | 67.41 | 57.18 | ✓ |
| sweep_twitter | twitter_ogm_ge_seed1024 | 1024 | 9 | 66.35 | 42.43 | ✓ |
| sweep_twitter | twitter_ogm_ge_seed123 | 123 | 16 | 66.35 | 52.54 | ✓ |
| sweep_twitter | twitter_ogm_ge_seed42 | 42 | 3 | 66.63 | 57.90 | ✓ |
| sweep_twitter | twitter_ogm_ge_seed456 | 456 | 8 | 66.35 | 47.90 | ✓ |
| sweep_twitter | twitter_ogm_ge_seed789 | 789 | 7 | 65.96 | 49.14 | ✓ |

# Per-run Index — Segmentation

| Sweep | Experiment | Seed | Best Ep | Dice (%) | WT (%) | TC (%) | ET (%) | ckpt |
|---|---|---|---|---|---|---|---|---|
| sweep_brats | brats_asgml_seed1024 | 1024 | 66 | 86.93 | 87.07 | 89.73 | 84.00 | ✓ |
| sweep_brats | brats_asgml_seed123 | 123 | 97 | 86.88 | 87.07 | 89.75 | 83.80 | ✓ |
| sweep_brats | brats_asgml_seed42 | 42 | 62 | 86.21 | 89.12 | 87.58 | 81.92 | ✓ |
| sweep_brats | brats_asgml_seed456 | 456 | 83 | 87.25 | 87.62 | 90.01 | 84.13 | ✓ |
| sweep_brats | brats_asgml_seed789 | 789 | 88 | 88.03 | 89.55 | 91.03 | 83.50 | ✓ |
| sweep_brats | brats_baseline_seed1024 | 1024 | 67 | 85.89 | 87.04 | 89.08 | 81.56 | ✓ |
| sweep_brats | brats_baseline_seed123 | 123 | 96 | 86.24 | 87.85 | 88.18 | 82.70 | ✓ |
| sweep_brats | brats_baseline_seed42 | 42 | 82 | 86.12 | 87.92 | 86.80 | 83.65 | ✓ |
| sweep_brats | brats_baseline_seed456 | 456 | 98 | 86.44 | 87.35 | 88.77 | 83.21 | ✓ |
| sweep_brats | brats_baseline_seed789 | 789 | 84 | 86.46 | 88.02 | 87.97 | 83.38 | ✓ |
| sweep_brats | brats_boost_ogmge_seed1024 | 1024 | 83 | 86.39 | 87.36 | 88.69 | 83.11 | ✓ |
| sweep_brats | brats_boost_ogmge_seed123 | 123 | 68 | 86.67 | 86.72 | 89.01 | 84.29 | ✓ |
| sweep_brats | brats_boost_ogmge_seed42 | 42 | 77 | 86.63 | 87.95 | 88.95 | 82.99 | ✓ |
| sweep_brats | brats_boost_ogmge_seed456 | 456 | 81 | 86.12 | 88.14 | 87.58 | 82.64 | ✓ |
| sweep_brats | brats_boost_ogmge_seed789 | 789 | 84 | 86.66 | 86.84 | 90.32 | 82.82 | ✓ |
| sweep_brats | brats_cggm_seed1024 | 1024 | 42 | 82.37 | 84.48 | 84.08 | 78.54 | ✓ |
| sweep_brats | brats_cggm_seed123 | 123 | 75 | 83.47 | 85.42 | 85.75 | 79.25 | ✓ |
| sweep_brats | brats_cggm_seed42 | 42 | 60 | 84.00 | 86.30 | 86.26 | 79.44 | ✓ |
| sweep_brats | brats_cggm_seed456 | 456 | 62 | 82.11 | 85.47 | 84.24 | 76.61 | ✓ |
| sweep_brats | brats_cggm_seed789 | 789 | 31 | 83.07 | 84.24 | 86.06 | 78.92 | ✓ |
| sweep_brats | brats_ogmge_seed1024 | 1024 | 84 | 86.15 | 87.17 | 88.30 | 82.98 | ✓ |
| sweep_brats | brats_ogmge_seed123 | 123 | 96 | 85.96 | 87.39 | 87.80 | 82.69 | ✓ |
| sweep_brats | brats_ogmge_seed42 | 42 | 93 | 86.36 | 87.38 | 88.10 | 83.59 | ✓ |
| sweep_brats | brats_ogmge_seed456 | 456 | 98 | 86.18 | 87.20 | 88.14 | 83.20 | ✓ |
| sweep_brats | brats_ogmge_seed789 | 789 | 84 | 86.42 | 87.91 | 88.09 | 83.28 | ✓ |

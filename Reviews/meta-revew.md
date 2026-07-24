Meta Review of Submission12365 by Area Chair wmcn
Meta Reviewby Area Chair wmcn19 Jul 2026, 15:20 (modified: 24 Jul 2026, 00:46)Senior Area Chairs, Area Chairs, Authors, Reviewers Submitted, Program Chairs, Area Chair wmcnRevisions
Metareview:
This paper proposes PGGB, a gradient-balancing method for multimodal learning. The method attaches stop-gradient linear probes to each modality, uses the probe-performance gap to identify underused modalities, and increases their encoder gradients through a bounded scaling rule. It can also be combined with methods such as OGM-GE that reduce the influence of the dominant modality.

The core idea is the strongest part of the paper. tQk1 considers the use of online probes as optimization controllers to be a novel and creative contribution. gN93 similarly finds the decoupled probe signal well motivated, lightweight, and easy to combine with existing training methods. The experiments cover several domains and include ablations and representation-level analysis. The clearest positive result is on CREMA-D, where the method improves the weaker modality while largely preserving the stronger one.

The empirical picture is less convincing across the full benchmark suite. miLe notes that the improvement over the strongest baseline is below 0.5 percentage points on seven of eight datasets. The same reviewer observes that OGM-GE accounts for most of the gain in the main ablation, while the additional PGGB improvement is within the reported seed variation. There is also an important unresolved discrepancy between the OGM-GE result reported on CREMA-D and the result in the original OGM-GE paper.

A second concern is the interaction with the optimizer. miLe argues that direct gradient scaling may have little effect under Adam, which is used on half of the datasets, and that these datasets indeed show only small gains. This needs a precise explanation or an analysis of the actual parameter updates.

The theoretical results are useful as basic safety properties, but they do not show that PGGB closes the modality gap or improves convergence speed, as pointed out by tQk1. gN93 also asks whether the stated guarantees meaningfully cover the high-imbalance regime that motivates the method. The paper should therefore state the scope of these results more carefully.

The most important points for the rebuttal are, in my opinion, as follows:

explain the CREMA-D baseline discrepancy raised by miLe;
clarify whether and how PGGB changes effective updates under Adam;
isolate the contribution of PGGB from OGM-GE with uncertainty over their difference;
provide early probe trajectories, including any warm-up behavior, as requested by tQk1;
clarify the dataset-level imbalance score and the choice of the 0.15 threshold, as requested by gN93; and
revise the theoretical claims so they match what the propositions establish.
To conclude, I lean toward acceptance because the probe-guided monitoring idea is original, simple, and suitable for a Concept and Feasibility contribution. However, the empirical attribution and the CREMA-D comparison need to be resolved in the response.
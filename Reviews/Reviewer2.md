Official Review of Submission12365 by Reviewer gN93
Official Reviewby Reviewer gN9326 Jun 2026, 08:33 (modified: 23 Jul 2026, 21:34)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer gN93Revisions
Summary:
This paper proposes Probe-Guided Gradient Boosting (PGGB) for multimodal learning, targeting the problem where one dominant modality can suppress learning in weaker modalities during joint optimization. The key idea is to attach lightweight linear probes to stop-gradient modality features, use probe accuracy gaps as a decoupled signal of modality imbalance, and boost gradients for weaker modalities using a bounded and smoothed scaling factor. The method can also be combined with existing dominant-modality throttling methods such as OGM-GE. The paper evaluates the approach across audio-visual, sentiment, text-image, and medical segmentation benchmarks, showing strong gains on CREMA-D and competitive or modest gains on other datasets.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths

The paper addresses an important and practical problem in multimodal learning: jointly trained models can over-rely on a dominant modality and underuse weaker but potentially useful modalities. The proposed PGGB method is novel because it uses stop-gradient linear probes to estimate modality utilization separately from the coupled joint loss. This gives a cleaner signal for identifying weaker modalities than methods that rely directly on the shared fusion objective.

The method is also lightweight and easy to combine with existing gradient modulation approaches. It does not require changing the fusion architecture or the main training loss, and the additional probe overhead is small. The experimental evaluation is reasonably broad, covering audio-visual, sentiment, text-image, and medical segmentation benchmarks. The results are strong on CREMA-D, where PGGB combined with OGM-GE improves over both standard joint training and OGM-GE alone. The ablation and post-hoc probe analyses also help support the claim that the method improves weak-modality representation quality while mostly preserving the stronger modality.

Weaknesses

The safety discussion in Section 3.5 is somewhat incomplete. The paper states that the safety properties show the method does not diverge or harm low-imbalance training, but it is less clear what happens in high-imbalance settings, which are the main target of the method. Since the strongest claimed benefit is on high-imbalance data, the authors should explain whether the bounded scaling and descent bound provide meaningful stability guarantees there, or whether high-imbalance behavior is supported mainly by empirical evidence.

The imbalance categorization in Section 4.1 needs clearer explanation in the main text. The paper categorizes datasets as high-imbalance when (delta > 0.15), but it is not immediately clear how one final (delta) value is obtained for a whole dataset. Appendix B.9 indicates that this is based on final-epoch EMA-smoothed probe accuracy gaps averaged over five seeds, but this should be stated explicitly in the main text. The authors should also justify the threshold (0.15) and discuss whether the categorization is sensitive to training epoch, probe quality, seed variation, or the chosen baseline.

The high-imbalance text-dominant sentiment datasets, CMU-MOSI and CMU-MOSEI, deserve deeper analysis. The authors state that text alone is near-multimodally sufficient and therefore only modest gains are expected, but this makes these datasets important extreme cases for testing the method. It would be useful to include more unimodal analysis showing whether PGGB actually improves the weaker audio and visual representations, even if final multimodal accuracy changes little.

Relatedly, it is unclear whether the limited gains on CMU-MOSI and CMU-MOSEI come from optimization imbalance or from intrinsic modality characteristics. Text, audio, and visual modalities may differ in task relevance more fundamentally than, for example, audio and visual modalities in CREMA-D. The authors should discuss whether boosting the weaker modalities is expected to help in such text-dominant settings, and whether increasing the boost strength (alpha) could improve learning from audio/visual modalities or instead amplify noisy, less informative signals.

Quality: 3: good
Clarity: 3: good
Significance: 3: good
Originality: 3: good
Questions:
In Section 3.5, the safety properties are described as ensuring that the method does not harm low-imbalance training. What should we expect in high-imbalance settings? Are there any stability guarantees or diagnostic criteria for when boosting may become harmful?

How exactly is the dataset-level (delta) value computed for imbalance categorization? Is it the final-epoch EMA-smoothed gap averaged over seeds, or some statistic over the full training trajectory? Please clarify this in the main text and justify the threshold (delta > 0.15).

For CMU-MOSI and CMU-MOSEI, can the authors provide a deeper unimodal analysis showing whether PGGB improves the audio and visual representations even when final multimodal accuracy changes little?

On text-dominant sentiment tasks, would increasing (alpha) help the model learn more from weaker audio/visual modalities, or does it degrade performance because those modalities are intrinsically less informative? An (alpha)-sensitivity study on CMU MOSI/MOSEI would be useful.

Limitations:
No. The authors discuss some limitations, including per-setting tuning, probe overhead, and possible regressions when composing with throttling. However, the limitations section should more directly address high-imbalance safety, sensitivity of the imbalance categorization, and the possibility that some weak modalities may be intrinsically less task-informative rather than merely undertrained.

Rating: 4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
No major formatting concerns observed.

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
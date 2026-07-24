Official Review of Submission12365 by Reviewer miLe
Official Reviewby Reviewer miLe21 Jun 2026, 06:57 (modified: 23 Jul 2026, 21:34)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer miLeRevisions
Summary:
The learning dynamics of late-fusion multi-modal architectures favor expedited learning of one modality at the expense of another when it provides stronger or easier-to-learn signal. This paper proposes a method, probe-guided gradient boosting (PGGB), to address this problem. PGGB boosts the gradients to the weaker modality's encoder by scaling it by a factor > 1. Prior work (OGM-GE) has proposed throttling the stronger modality's encoder by scaling it with a factor < 1. On the other hand, prior works proposing to boost the gradient have done so in ways that are still biased toward the stronger modality. PGGB uses linear probes on the modality encoders to estimate the "utilization gap" between the prediction accuracies of the strong and the weak modalities and computes the boosting scale based on that. By using a stop-gradient operator before the linear probe, PGGB decouples the monitoring signal from modality-biased learning dynamics. PGGB can also be combined with OGM-GE. The paper shows empirical results on a number of datasets, gives theoretical guarantees for PGGB and performs various ablation studies.

Contribution Type: Concept and Feasibility: The main contribution is a highly novel, high potential reward idea with scope beyond what can be validated in a single paper. (The significance and originality bar for these contributions is high.)
Strengths And Weaknesses:
Strengths
S1. The method PGGB is simple and reasonable (albeit not for scale-invariant optimizers like Adam, see Weaknesses).

S2. The experiments cover many datasets from different domains.

S3. There are many ablations.

S4. The paper provides theoretical guarantees for PGGB.

Weaknesses
W1. Firstly, I believe this paper would be better served by contribution type "General" instead of "Concept and Feasibility" as the scope of the proposed method is small enough to be validated in a single paper.

W2. My biggest concern is that the results are underwhelming. The main results table (Table 1) shows improvements of < 0.5 pp over the best baseline on 7 out of 8 datasets which is extremely minor.

W3. The proposed method of scaling the gradients should not be expected to give significant gains for the Adam optimizer which is scale-invariant / unit-less. 4 out of 8 datasets use Adam and as expected show barely any improvement. OGM-GE has plausible improvement even with Adam (see Table 6 in their paper) because the GE component goes beyond mere gradient scaling.

W4. The ablation table (Table 2) shows that OGM-GE alone recovers most of the gap between PGGB+OGM-GE and the baseline (2.17pp) and PGGB's contribution is minor (0.22pp) well within the per-seed standard deviation.

Quality: 2: not good
Clarity: 2: not good
Significance: 2: not good
Originality: 3: good
Questions:
Q1. On L17 what are the 2-4 modalities?

Q2. L131, g is undefined.

Q3. The last paragraph of section 3.1 is quite unclear about the intuition and formalism of what is being said.

Q4. L147 is unclear.

Q5. On L172, s_m becomes undefined when modalities are balanced, not what is said in the paper.

Q6. For section 3.5, what are the intuitions and takeaways from the propositions?

Q7. MAJOR. The OGM-GE paper reports 61.59 on CREMA-D. You report 69.14. Where is this discrepancy coming from?

Limitations:
Yes.

Rating: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
None.

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
# Related Work Review v2 (Section 2)

**File reviewed:** `/Users/vuvo/Desktop/RMIT-AI/My PhD/Neurips2026-AsyncFunc/Manuscript/main.tex`, lines 105-183
**Date:** 2026-04-05
**Version:** v2 (revision of Section 2 addressing v1 review at score 7.5/10)
**Score: 8.5 / 10**

---

## Summary

Section 2 consists of four `\paragraph` blocks covering: (1) theoretical foundations of modality imbalance, (2) gradient modulation methods and their throttle-dominant bias, (3) alternative approaches beyond gradient scaling, and (4) representation monitoring via probing and the paper's positioning. The revision addresses the six most important issues from the v1 review: Du et al. characterization, AGM/MLGM bidirectional claims, PMR inconsistency, missing Data Remixing and DI-MML papers, split-batch evaluation protocol, and P2.4 sentence clarity. The resulting section is materially stronger, with a more honest and defensible positioning.

---

## Resolution of v1 Issues

### Issue 1 (W1): Du et al. characterization -- RESOLVED
- **v1 concern:** The text overstated Du et al. as analyzing "active suppression" by the dominant modality.
- **v2 text (line 121):** "Du et al. provide further theoretical analysis of the conditions under which individual modality features fail to be learned in joint multimodal training."
- **Verdict:** Fully resolved. The new phrasing is accurate and measured. It supports the paragraph's argument without overinterpretation.

### Issue 2 (W2): PMR inconsistency between P2.3 and P2.4 -- RESOLVED
- **v1 concern:** P2.3 claimed "all" methods use coupled signals but P2.4 acknowledged PMR as partially decoupled, creating internal tension.
- **v2 text (line 163):** P2.3 now includes the qualifier "(with the partial exception of PMR's decoupled unimodal prototypes [Fan et al., 2023])."
- **Verdict:** Fully resolved. The option (a) approach from v1's suggestion was adopted. The qualifier is precise and honest.

### Issue 3 (W3): Missing Data Remixing and DI-MML -- RESOLVED
- **v1 concern:** Two directly relevant papers (Data Remixing, ICML 2025; DI-MML, ACM MM 2024) were absent.
- **v2 text (line 163):** Both are now cited in P2.3. Data Remixing appears as "data decoupling and reassembly [Ma et al., 2025]" and DI-MML appears as "detached unimodal pre-training with contrastive knowledge transfer [Fan et al., 2024]."
- **Verdict:** Resolved for inclusion, but with a minor concern (see New Issue N1 below regarding DI-MML characterization depth).

### Issue 4 (W4): AGM/MLGM bidirectional claims -- RESOLVED
- **v1 concern:** P2.2 characterized the paradigm as throttle-only without acknowledging AGM and MLGM's bidirectional claims.
- **v2 text (line 141):** Now reads: "Although some methods claim bidirectional modulation (e.g., AGM [Li et al., 2023] and MLGM [Kontras et al., 2024] include acceleration terms), the throttling component empirically dominates, and direct amplification of the weaker modality using only boost signals remains ineffective."
- **Verdict:** Fully resolved. This is essentially the exact suggestion from v1. The phrasing is fair: it acknowledges the claims, states the empirical reality, and connects to the OGM* evidence. This is now one of the strongest sentences in the section.

### Issue 5 (W5): Split-batch evaluation protocol missing from text -- RESOLVED
- **v1 concern:** The split-batch protocol was mentioned only in LaTeX comments, not in the body text.
- **v2 text (line 183):** "...trains fully decoupled linear probes on detached encoder features with separate optimizers and evaluates them on held-out within-batch samples to prevent probe overfitting."
- **Verdict:** Fully resolved. The split-batch protocol is now in the positioning paragraph where it has maximum impact.

### Issue 6 (W6): P2.1 no forward reference -- RESOLVED
- **v1 concern:** P2.1's concluding sentence did not connect the theoretical foundations to the method.
- **v2 text (line 121):** Now ends with "...a structural property of joint multimodal optimization, one whose root cause, the shared loss derivative, contaminates conventional monitoring signals."
- **Verdict:** Resolved, though the approach differs from the v1 suggestion. Instead of adding a `(Section~\ref{sec:method})` forward reference, the text integrates the connection directly into the sentence by naming "the shared loss derivative" as the root cause and identifying that it "contaminates conventional monitoring signals." This is arguably better than a naked forward reference because it states the mechanism rather than deferring explanation. However, the absence of an explicit cross-reference to Section 3 means the reader must trust the connection will be formalized later. A minor improvement would be to add "(formalized in Section~\ref{sec:method})" but this is a stylistic preference, not a deficiency.

### Issue 7 (W7): P2.4 final sentence density -- RESOLVED
- **v1 concern:** The final sentence packed too many claims into one semicolon-joined structure.
- **v2 text (lines 183-184):** Now split into two sentences: "...trains fully decoupled linear probes on detached encoder features with separate optimizers and evaluates them on held-out within-batch samples to prevent probe overfitting. The resulting probe-detected utilization gap directly boosts the weaker modality's encoder gradients, ensuring that the imbalance estimate is not contaminated by the very dynamics it aims to correct."
- **Verdict:** Fully resolved. The two-sentence structure is clearer and gives the contamination argument its own sentence, which is appropriate since it is the punchline of the entire Related Work section.

### Issue 8 (W8): ARL characterization inconsistency with Introduction -- NOT ADDRESSED
- **v1 concern:** ARL is treated as important in the Introduction ("challenges the assumption that balanced learning is optimal") but receives minimal characterization in Related Work (just "bias-variance decomposition for inverse-variance weighting" in a list).
- **v2 status:** The characterization in P2.3 remains unchanged. ARL is still a single item in the enumeration with a brief descriptor.
- **Verdict:** Not addressed, but the impact remains Minor. The Introduction (P3, line 77) now cites ARL within a long parenthetical alongside 9 other methods, which is a more proportionate treatment than the v1 Introduction's singleton mention. The inconsistency has been partially resolved from the Introduction side rather than the Related Work side.

---

## Remaining Weaknesses from v1

### R1. P2.3 enumeration readability (Minor Issue 3 in v1) -- PARTIALLY ADDRESSED
- **v1 concern:** The flat comma-separated list of 9 methods was heavy cognitive load.
- **v2 status:** The list has grown to 11 methods (Data Remixing and DI-MML were added) but remains a flat enumeration. It is now even longer than before.
- **Impact:** Minor-to-Moderate. The list is syntactically correct but cognitively taxing. At 11 items connected by commas, this is the densest sentence in the entire section. A reader skimming quickly will extract nothing from this sentence beyond "many methods exist." The v1 suggestion to group by intervention type (semicolons separating categories) would still improve this substantially.
- **Suggestion:** Group into 3-4 categories separated by semicolons. For example: "These include feed-forward and gradient-based interventions [Wei et al., 2024; Wei & Hu, 2024]; modality-alternating and decoupled training strategies [Hua et al., 2024; Zhang et al., 2024; Fan et al., 2024]; diagnostic and regulatory approaches [Wei et al., 2024b; Huang et al., 2025; Gao et al., 2025; Wei et al., 2025]; and data-level interventions [Yao & Mihalcea, 2022; Ma et al., 2025]." This imposes structure that aids comprehension.

### R2. TCMax (ICLR 2026) still absent -- NOT ADDRESSED
- **v1 concern:** TCMax (Wu et al., ICLR 2026) is the most recently published method at a top venue.
- **v2 status:** Still absent.
- **Impact:** Low-to-Medium. An ICLR 2026 paper on multimodal balance would be noticed by an up-to-date reviewer. However, whether this matters depends on TCMax's mechanism: if it uses coupled monitoring (likely), it simply joins the P2.3 list without changing the argument. If it uses a decoupled approach, its omission is more problematic.
- **Suggestion:** Add TCMax as one item in P2.3's list if it uses coupled monitoring. If it introduces a decoupled mechanism, it needs a sentence in P2.4 explaining how it differs from the proposed method. Verify the mechanism before citing.

### R3. MSLR characterization remains minimal (Minor Issue 4 in v1) -- NOT ADDRESSED
- **v1 concern:** MSLR (Yao & Mihalcea, ACL 2022) gets no descriptor beyond "modality-specific learning rates."
- **v2 status:** Unchanged.
- **Impact:** Minor. MSLR is a lower-priority paper. The brief mention is acceptable for NeurIPS space constraints.

### R4. Minor Issue 2 (GradNorm/PCGrad framing) -- NOT ADDRESSED
- **v1 concern:** "Early multi-task methods" could imply they are early within the multimodal imbalance literature.
- **v2 text (line 141):** Still reads "Early multi-task methods such as GradNorm and PCGrad."
- **Impact:** Minor. The subsequent phrase "subsequently adapted to the multimodal setting" clarifies the timeline.

### R5. Minor Issue 6 (Alain & Bengio workshop paper) -- NOT ADDRESSED
- **v1 concern:** Citing a workshop paper (ICLR 2017 Workshop) as the primary reference for linear probes.
- **v2 status:** Unchanged.
- **Impact:** Minor. Alain & Bengio is widely cited despite being a workshop paper. Supplementing with Chen et al. (2020, SimCLR) would strengthen the reference but is not essential.

### R6. Minor Issue 7 (IPRM sentence structure) -- NOT ADDRESSED
- **v1 concern:** The semicolon-however construction after an "in which" clause is slightly awkward.
- **v2 status:** Unchanged. Still reads "...in which the first pass probes modality strength via KL divergence between unimodal and fused predictions; however, these probes operate on live-model outputs..."
- **Impact:** Minor. Readability issue, not a logical problem.

---

## New Weaknesses Identified in v2

### N1. DI-MML characterization may be too brief for its relevance
- **Concern:** DI-MML (Fan et al., ACM MM 2024) is characterized as "detached unimodal pre-training with contrastive knowledge transfer" and placed at the end of P2.3's 11-item list. However, DI-MML's "detached" training paradigm is directly relevant to the paper's central "decoupled vs. coupled" axis. DI-MML arguably deserves a sentence in P2.4 explaining how its notion of detachment differs from the paper's decoupled probes. Specifically: DI-MML detaches the *training stage* (separate unimodal pre-training then multimodal fine-tuning), while the proposed method detaches the *monitoring signal* (probes on detached features during joint training). This distinction matters because a reviewer could ask: "How is your decoupled monitoring different from DI-MML's detached training?"
- **Impact:** Moderate. A reviewer familiar with DI-MML will note the terminological overlap and expect a distinction.
- **Suggestion:** Add one sentence to P2.4 after the PMR discussion: "DI-MML [Fan et al., 2024] achieves decoupling at the training-stage level through separate unimodal pre-training, whereas our method decouples the monitoring signal while maintaining end-to-end joint training."

### N2. The P2.1 concluding sentence may now overreach slightly
- **Concern:** The revised P2.1 concluding sentence reads: "...a structural property of joint multimodal optimization, one whose root cause, the shared loss derivative, contaminates conventional monitoring signals." The claim that the shared loss derivative is "the root cause" of modality imbalance is a specific theoretical claim that goes beyond what the cited papers (Wang et al., Huang et al., Du et al., Zhang et al.) formally establish. Huang et al. (2022) prove that joint training learns only a subset of modalities under a Gaussian mixture model; they do not specifically identify "the shared loss derivative" as the root cause in those terms. The "shared loss derivative contamination" argument is the *paper's own contribution* (developed in the Introduction, P3, line 77 and presumably Section 3). Attributing it to "these results collectively" could be seen as overclaiming support from prior work.
- **Impact:** Minor-to-Moderate. A theory-oriented reviewer would note the gap between what the cited papers prove and what the sentence claims "these results" establish.
- **Suggestion:** Soften the attribution slightly: "...a structural property of joint multimodal optimization. As we argue in Section~\ref{sec:method}, this imbalance contaminates the very monitoring signals used to detect it." This separates the prior work's established findings from the paper's own theoretical contribution.

### N3. P2.3 does not distinguish its 11 methods by intervention type
- **Concern:** This is related to R1 but has a substantive dimension. With 11 methods now in the list, the paragraph makes a strong unifying claim ("they share a common design choice: coupled monitoring") but provides no structure to help the reader verify this claim across such diverse methods. A reviewer who does not know all 11 methods cannot assess whether the "coupled" characterization applies to each one. DI-MML in particular is called "detached" in its own name, making the "coupled" label potentially confusing.
- **Impact:** Moderate. The unifying claim is rhetorically powerful but becomes harder to verify as the list grows.
- **Suggestion:** Either (a) group the methods and briefly note why each group uses coupled monitoring, or (b) explicitly name DI-MML and Data Remixing as exceptions that are "partially decoupled at the data/training-stage level" alongside the existing PMR exception. This would be more honest than lumping them under the "coupled" umbrella without qualification.

### N4. "Held-out within-batch samples" is ambiguous
- **Concern:** P2.4 (line 183) now mentions "evaluates them on held-out within-batch samples to prevent probe overfitting." The phrase "held-out within-batch samples" is technically accurate (the batch is split: first half for probe training, second half for probe evaluation) but may be confusing to a reader who has not yet read Section 3. "Within-batch" could be misinterpreted as meaning the probe is evaluated on the same samples it was trained on (just within the current batch). The intended meaning -- that the batch is split into training and evaluation subsets -- is not self-evident from the phrase alone.
- **Impact:** Minor. This is a clarity issue. The full mechanism will presumably be explained in Section 3, but at this point the reader encounters the phrase without context.
- **Suggestion:** Clarify slightly: "...and evaluates them on the second half of each training batch (a held-out split) to prevent probe overfitting." Or simply: "...with a split-batch evaluation protocol that prevents probe overfitting."

---

## Strengths (Retained and New)

1. **Narrative arc remains excellent.** The four-paragraph structure (theory -> throttle paradigm -> broader landscape + coupled monitoring -> decoupled probes) is unchanged and still reads as a textbook example of positioning through gap analysis.

2. **The AGM/MLGM acknowledgment is now a strength.** The revised P2.2 sentence that acknowledges bidirectional claims before arguing the throttle component dominates is one of the best sentences in the section. It demonstrates fairness and intellectual honesty -- exactly what a reviewer wants to see.

3. **PMR qualifier in P2.3 is honest and well-placed.** The parenthetical exception for PMR transforms what was an overclaim into a nuanced, defensible observation. This is the kind of precision that earns reviewer trust.

4. **Split-batch protocol in P2.4 sharpens the positioning.** The addition of "evaluates them on held-out within-batch samples to prevent probe overfitting" adds a concrete technical differentiator that was previously invisible. This is now a three-part distinction: (i) detached features, (ii) separate optimizers, (iii) held-out evaluation -- which is a stronger positioning than the v1's two-part claim.

5. **The P2.4 two-sentence conclusion is cleaner.** Splitting the v1 semicolon sentence into two gives each claim appropriate emphasis. The final sentence ("The resulting probe-detected utilization gap directly boosts the weaker modality's encoder gradients, ensuring that the imbalance estimate is not contaminated by the very dynamics it aims to correct") is now the clean punchline the section deserves.

6. **Coverage improved.** Adding Data Remixing and DI-MML brings the section closer to comprehensive coverage. The section now cites approximately 24 papers, which is strong for NeurIPS Related Work.

---

## Questions for Authors

1. **On DI-MML's "detached" vs. your "decoupled":** DI-MML's name literally contains "Detached." How do you distinguish your decoupled probes from DI-MML's detached training? The current text places DI-MML in the "coupled monitoring" list, but a reviewer might push back given the naming. Is DI-MML's monitoring of imbalance itself coupled even though its training stages are detached?

2. **On the shared loss derivative claim in P2.1:** The concluding sentence attributes the "shared loss derivative contaminates monitoring signals" insight to prior work ("These results collectively establish..."). Is this claim formally established in any of the four cited papers, or is it your own theoretical contribution? If the latter, the attribution should be adjusted.

3. **On TCMax (ICLR 2026):** What is TCMax's monitoring mechanism? If it uses coupled monitoring, it should join the P2.3 list. If it uses something novel, it needs more than a list mention.

4. **On the P2.3 enumeration:** With 11 methods now in the list, have you considered grouping them by intervention type? This would both improve readability and make the "all coupled" claim easier for reviewers to verify.

5. **On probe overfitting:** You mention "held-out within-batch samples" to prevent probe overfitting. Has probe overfitting actually been observed in practice (e.g., in IPRM or CGGM)? If so, this is worth citing. If not, the phrase "to prevent probe overfitting" may overstate the risk.

---

## Minor Issues

1. **P2.3, line 163:** The enumeration now contains 11 items separated by commas. This is a 6-line sentence that is syntactically correct but very difficult to parse on first reading. Consider semicolons or grouping (see R1 and N3 above).

2. **P2.2, line 141:** The phrase "direct amplification of the weaker modality using only boost signals remains ineffective" is slightly ambiguous. Does "using only boost signals" modify "direct amplification" (i.e., boost-only is what fails) or "remains ineffective" (i.e., it fails specifically when only boost signals are used)? The intended meaning is clear from context but could be sharper: "and direct amplification of the weaker modality through boost-only signals remains ineffective."

3. **P2.1, line 121:** The sentence "These results collectively establish that modality imbalance is not merely an empirical nuisance but a structural property of joint multimodal optimization, one whose root cause, the shared loss derivative, contaminates conventional monitoring signals" is a single 37-word sentence with an embedded appositive clause. The comma before "one whose root cause" makes this parsable but dense. Consider a period after "optimization" and starting a new sentence: "Its root cause -- the shared loss derivative -- contaminates conventional monitoring signals."

4. **P2.3, line 163:** "data decoupling and reassembly [Ma et al., 2025]" -- verify this characterization matches Data Remixing's actual mechanism. The paper's title is "Improving multimodal learning balance and sufficiency through data remixing." The phrase "data decoupling and reassembly" is close but may not be the paper's own terminology. Consider "data remixing [Ma et al., 2025]" for fidelity to the original.

5. **P2.3, line 163:** The PMR exception is correctly parenthesized but the parenthetical is long: "(with the partial exception of PMR's decoupled unimodal prototypes [Fan et al., 2023])." Consider moving it to a clause: "With the partial exception of PMR [Fan et al., 2023], which uses decoupled unimodal prototypes, these methods share..."

6. **P2.4, line 183:** "trains fully decoupled linear probes on detached encoder features with separate optimizers" -- the word "fully" before "decoupled" implies there are degrees of decoupling, which is actually supported by the PMR discussion. But this raises the question: what would "partially decoupled" probes look like? If the only alternative is coupled vs. decoupled, "fully" is redundant. If there is a spectrum (as the PMR discussion implies), "fully" is justified but should be defined.

---

## Coverage Audit Update

Papers cited in v2 Related Work (approximately 24 papers):
- **P2.1 (Theory):** wang2020gblending, huang2022modality, du2023suppression, zhang2024unimodal
- **P2.2 (Gradient modulation):** chen2018gradnorm, yu2020pcgrad, peng2022ogmge, li2023agm, guo2024cggm, kontras2024mlgm, wei2024opm
- **P2.3 (Alternative approaches):** wei2024opm, wei2024mmpareto, hua2024reconboost, zhang2024mla, wei2024dr, yao2022mslr, huang2025inforeg, gao2025arm, wei2025arl, ma2025remixing, fan2024dimml, fan2023pmr
- **P2.4 (Probing):** alain2017understanding, yang2025iprm, jiang2025aug, fan2023pmr, guo2024cggm

Notable papers still NOT cited:
| Paper | Venue | Relevance | Urgency |
|-------|-------|-----------|---------|
| TCMax (Wu et al.) | ICLR 2026 | Most recent top-venue method | Medium |
| GMML | ACM MM 2025 | Convergence guarantees | Low-Medium |
| G2D (Rakib et al.) | ICCV 2025 | Distillation paradigm | Low |
| CMoB | NeurIPS 2025 | Causal valuation, tangential | Low |

**Verdict on coverage:** The two most important gaps (Data Remixing, DI-MML) are now addressed. TCMax remains a moderate concern depending on its mechanism. The coverage is now good-to-strong for NeurIPS.

---

## Positioning Assessment Update

The positioning has improved from v1. The three-part technical differentiation (detached features + separate optimizers + held-out evaluation) is stronger than the v1's two-part claim. The AGM/MLGM acknowledgment and PMR qualifier demonstrate intellectual honesty that builds reviewer trust.

**Remaining positioning risk:** The DI-MML "detached" terminology overlap (N1 above) is the main new risk. A reviewer who knows DI-MML will want a clear distinction. This is addressable with one sentence in P2.4.

**Composability angle:** Still not explicitly mentioned in Related Work. The Introduction (P4, line 84) makes the composability argument clearly, so its absence from Related Work is defensible -- but including it in P2.2 or P2.4 would strengthen the "what we enable" framing (as noted in v1).

---

## Fairness Assessment Update

Improved from v1. Specific evaluations:

- **AGM/MLGM:** Now fairly treated. Bidirectional claims acknowledged before the counter-argument. This is a significant improvement.
- **PMR:** Now fairly treated. The parenthetical exception in P2.3 is precise and honest.
- **DI-MML:** Fairly cited but under-differentiated (see N1). Not unfair, but incomplete.
- **All other methods:** Unchanged from v1 (fair characterizations).

No methods are strawmanned. The section's fairness is now a strength rather than a concern.

---

## Overall Assessment

This is a strong revision that addresses six of eight issues from the v1 review, resolving all five high/medium-priority items and one low-priority item. The most impactful changes are: (1) the AGM/MLGM bidirectional acknowledgment, which transforms a potential reviewer complaint into a demonstration of intellectual honesty; (2) the PMR qualifier, which fixes the internal inconsistency; and (3) the split-batch protocol addition, which concretizes the positioning. The addition of Data Remixing and DI-MML closes the most conspicuous coverage gaps.

The section now reads as a well-crafted positioning argument rather than just a literature survey. The narrative arc is clear, the characterizations are fair, and the gap identification is honest. The main remaining concerns are: (1) the 11-item enumeration in P2.3 is becoming unwieldy and would benefit from structural grouping; (2) DI-MML's "detached" paradigm needs explicit differentiation from the paper's "decoupled" probes; and (3) the P2.1 concluding sentence attributes a claim to prior work that may actually be the paper's own contribution.

None of these remaining issues are fatal or even major. They are refinements that would elevate the section from strong to excellent.

**Score: 8.5 / 10** (solid and well-revised; addresses all major v1 concerns; remaining issues are moderate at worst and readily fixable)

---

## Confidence Score

**4 / 5.** I have performed a line-by-line comparison of the v1 text against the v2 revision, cross-referenced all action items from v1, checked reference entries for newly added citations, and verified consistency with the Introduction. My confidence is slightly lower on the N2 concern (whether "shared loss derivative" is formally established in the cited theory papers) since I have not read those papers' proofs in full -- this assessment is based on typical scope of theoretical results in that literature.

---

## Action Items (Priority Order)

1. **[Medium]** Add one sentence to P2.4 distinguishing DI-MML's detached training stages from the paper's decoupled monitoring (N1).
2. **[Medium]** Restructure P2.3's 11-item enumeration into grouped categories (R1/N3).
3. **[Medium]** Soften the P2.1 concluding sentence to separate prior work's findings from the paper's own "shared loss derivative" argument (N2).
4. **[Low-Medium]** Add TCMax to P2.3 list or verify it is not relevant enough to warrant inclusion (R2).
5. **[Low]** Clarify "held-out within-batch samples" phrasing in P2.4 (N4).
6. **[Low]** Fix P2.3 characterization of Data Remixing to match the paper's own terminology (Minor Issue 4).
7. **[Low]** Consider adding composability angle to P2.2 or P2.4 (v1 positioning suggestion, still applicable).

---

## Delta Summary (v1 -> v2)

| Aspect | v1 Score | v2 Score | Change |
|--------|----------|----------|--------|
| Coverage | Good (20 papers, 2 major gaps) | Strong (24 papers, 0 major gaps) | +1.0 |
| Fairness | Good (AGM/MLGM risk) | Strong (all methods fairly treated) | +0.5 |
| Positioning | Strong but overclaimed | Strong and honest | +0.5 |
| Readability | Good (P2.3 enumeration, P2.4 dense) | Good-to-Mixed (P2.3 longer, P2.4 better) | +0.0 |
| Technical accuracy | Minor Du et al. issue | Resolved, minor P2.1 attribution issue | +0.5 |
| **Overall** | **7.5** | **8.5** | **+1.0** |

The improvement is genuine and well-targeted. The revision focused effort on the highest-impact issues and succeeded.

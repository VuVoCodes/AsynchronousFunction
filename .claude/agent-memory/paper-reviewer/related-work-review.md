# Related Work Review (Section 2)

**File reviewed:** `/Users/vuvo/Desktop/RMIT-AI/My PhD/Neurips2026-AsyncFunc/Manuscript/main.tex`, lines 105-183
**Date:** 2026-04-05
**Version:** v1 (first review of Related Work section)
**Score: 7.5 / 10**

---

## Summary

The Related Work section consists of four `\paragraph` blocks totaling approximately 0.85 pages, which is appropriate for NeurIPS. The section builds a clear narrative: (1) modality imbalance is a structural optimization problem, (2) the dominant response has been gradient throttling, which is one-sided, (3) diverse alternative approaches exist but all rely on coupled monitoring, and (4) probing as a monitoring tool is established but no prior work combines decoupled probes with gradient-level boosting. The writing is generally strong, the positioning is sharp, and the logical arc culminates cleanly in the method's unique contribution.

---

## Strengths

1. **Clear narrative arc.** Each paragraph builds logically on the previous one. P2.1 establishes the theoretical foundations, P2.2 identifies the throttle-only bias, P2.3 widens the lens to show that the coupled-monitoring limitation transcends gradient modulation, and P2.4 positions the method precisely. This is textbook related-work structure.

2. **Effective use of the OGM* failure.** The P2.2 mention of Wei et al. (TPAMI 2024) showing boost-only fails is the most compelling argumentative move in the section. It transforms the related work from a literature survey into a motivated gap analysis. The forward-reference to Section 3 ("an outcome we trace to shared loss derivatives...") promises a formal explanation.

3. **Honest and precise competitor differentiation in P2.4.** The distinctions from IPRM (coupled probes on live-model outputs), CGGM (classifiers participate in loss), AUG (loss-based detection), and PMR (loss-level intervention) are specific, verifiable, and non-strawmanning. Each competitor gets credit for what it does before the distinction is drawn.

4. **P2.3's unifying observation is strong.** The claim that diverse methods (loss, architecture, LR, data) all share coupled monitoring is a genuinely useful synthesis that readers will appreciate. It reframes the landscape around the paper's specific contribution axis.

5. **Appropriate length.** At roughly 0.85 pages, the section respects NeurIPS space constraints without sacrificing substance. It covers ~20 papers, which is a reasonable density.

6. **No em-dashes in body text.** The user's constraint is satisfied; all `---` occurrences are in LaTeX comments only.

7. **Citation density is good.** P2.2 cites 6 papers, P2.3 cites 9, P2.4 cites 5. The section covers the core landscape without being a citation dump.

---

## Weaknesses

### W1. P2.1 characterization of Du et al. (ICML 2023) may be inaccurate
- **Concern:** The text states Du et al. "analyze how the dominant modality's feature representations actively suppress those of weaker modalities during optimization." However, Du et al. (ICML 2023) actually study *uni-modal feature learning* in supervised multi-modal settings, analyzing conditions under which individual modality features are learned or suppressed. Their analysis is about when and why certain features fail to be learned, not about active suppression by a dominant modality per se. The current phrasing implies a competitive/adversarial dynamic that is stronger than what Du et al. actually claim.
- **Impact:** Moderate. A reviewer familiar with Du et al. would flag this as an overinterpretation.
- **Suggestion:** Revise to: "Du et al. [2023] provide further theoretical analysis of the conditions under which individual modality features fail to be learned in joint multimodal training." This is both accurate and still supports the paragraph's argument.

### W2. P2.3 "coupled signals" claim is too sweeping
- **Concern:** The paragraph states all listed methods "share a common design choice: imbalance is detected through signals *coupled* to the joint training objective." This is true for most, but PMR (Fan et al., CVPR 2023) uses *decoupled* unimodal prototypes for evaluation. PMR explicitly trains unimodal classifiers and evaluates them separately. This was correctly noted in P2.4 ("PMR takes a step toward decoupled evaluation"), but P2.3's blanket claim already includes PMR in the list (via citation of fan2023pmr in P2.2). A careful reviewer will notice this internal tension: P2.3 says *all* these methods use coupled signals, but P2.4 acknowledges PMR is partially decoupled.
- **Impact:** Moderate. Internal inconsistency weakens the "all coupled" argument.
- **Suggestion:** Two options: (a) Add a qualifier in P2.3: "With the partial exception of PMR's decoupled unimodal evaluation, these methods share..." or (b) Remove PMR from P2.3's scope and handle it entirely in P2.4. Option (b) is cleaner since PMR already has a dedicated sentence in P2.4.

### W3. Missing key papers that reviewers will notice
- **Concern:** Several papers from the 34-paper inventory in `RelatedWorks.md` are absent and would be noticed by an informed reviewer:
  - **Data Remixing (Ma et al., ICML 2025):** A strong baseline achieving +6.50pp on CREMA-D. If this method is in the experiments section as a baseline, its omission from Related Work is a gap. If not, a reviewer familiar with it will wonder why.
  - **DI-MML (Fan et al., ACM MM 2024):** Uses detached training with unidirectional contrastive knowledge transfer. The "detached" aspect directly relates to the decoupled-vs-coupled distinction that is central to the paper's positioning.
  - **MILES (Guerra-Manzanares & Shamout, IJCNN 2025):** Uses learning rate scheduling based on conditional utilization rate differences. Relevant to the "alternative approaches" paragraph as another form of non-gradient-scaling intervention.
  - **TCMax (Wu et al., ICLR 2026):** Total correlation maximization is the most recent published approach from a top venue. Its omission may signal the paper is not up-to-date.
  - **G2D (Rakib et al., ICCV 2025):** Gradient-guided distillation from unimodal teachers. A different paradigm (distillation) that is worth mentioning.
- **Impact:** Major for Data Remixing and DI-MML (directly relevant); Minor for MILES, TCMax, G2D (relevant but not essential).
- **Suggestion:** Add Data Remixing to P2.3 (one clause). Add DI-MML to P2.3 or P2.4 with emphasis on its "detached" training aspect. Optionally mention TCMax and G2D in P2.3. MILES can be safely omitted (lower-tier venue).

### W4. P2.2 "overwhelmingly one-sided" claim needs qualification
- **Concern:** The text states the intervention is "overwhelmingly one-sided: these methods primarily *throttle* the dominant modality's gradient, while direct amplification of the weaker modality remains largely ineffective." However, AGM (Li et al., ICCV 2023) explicitly claims to do *both* boosting and throttling via Shapley-value attribution. MLGM (Kontras et al., BMVC 2024) also claims acceleration (boost) and deceleration (throttle). The text cites both but characterizes the paradigm as throttle-only without acknowledging their bidirectional claims. A reviewer who knows AGM or MLGM will push back.
- **Impact:** Moderate. The argument is not wrong (AGM/MLGM's boost components are empirically dominated by throttling), but the current wording dismisses their claims without evidence.
- **Suggestion:** Add a brief qualifier: "Although some methods claim bidirectional modulation (e.g., AGM [Li et al., 2023], MLGM [Kontras et al., 2024]), the throttling component empirically dominates, and direct amplification of the weaker modality using *only* boost signals has been shown to fail [Wei et al., 2024]." This acknowledges their claims while maintaining the argument.

### W5. P2.4 does not discuss split-batch evaluation protocol
- **Concern:** The LaTeX comments (lines 176-179) explicitly note that "split-batch evaluation (train on first half, evaluate on second) further prevents probe overfitting -- a protocol not used in any prior work." This is a genuine differentiator, but it does not appear in the actual text. The P2.4 body only mentions "fully decoupled probes, trained on detached encoder features with separate optimizers." The split-batch protocol is arguably as important as feature detachment for ensuring unbiased monitoring.
- **Impact:** Moderate. A key technical distinction is omitted from the positioning paragraph, which is where it would have the most impact.
- **Suggestion:** Add a clause: "...trained on detached encoder features with separate optimizers *and evaluated on held-out within-batch samples to prevent probe overfitting*..." This can be done concisely without adding significant length.

### W6. P2.1 makes no forward reference to how theory motivates the method
- **Concern:** P2.1 establishes that modality imbalance is a structural optimization problem with solid theoretical grounding, but does not connect this to the paper's approach. The concluding sentence ("These results collectively establish that modality imbalance is not merely an empirical nuisance but a structural property...") is a good summary but a missed opportunity. How does knowing the problem is structural inform the solution design? The theoretical results say the *duration* of the unimodal phase matters (Zhang et al., 2024) and that the *shared loss derivative* is the culprit (Huang et al., 2022). These directly motivate decoupled monitoring.
- **Impact:** Minor. The connection is implicit and savvy readers will make it, but making it explicit strengthens the narrative thread.
- **Suggestion:** Add one phrase to the concluding sentence: "...a structural property of joint multimodal optimization *whose root cause, the shared loss derivative, contaminates conventional monitoring signals* (Section~\ref{sec:method})." Or save this for the P2.2-to-P2.3 transition.

### W7. P2.4 final sentence is dense and could be sharper
- **Concern:** The final sentence of P2.4 (and thus the entire Related Work) is: "In contrast, our method combines *fully decoupled* probes, trained on detached encoder features with separate optimizers, with gradient-level boosting of the weaker modality; this separation ensures that the imbalance estimate is not contaminated by the very dynamics it aims to correct." This is good but packs two distinct claims (decoupled probes + gradient boosting + contamination argument) into one sentence with a semicolon. As the positioning punchline, it deserves more breathing room.
- **Impact:** Minor. Readability concern, not a logical issue.
- **Suggestion:** Split into two sentences: "In contrast, our method trains *fully decoupled* linear probes on detached encoder features with separate optimizers, and uses the resulting probe-detected utilization gap to directly boost the weaker modality's encoder gradients. This separation ensures that the imbalance estimate is not contaminated by the very dynamics it aims to correct." Alternatively, the existing single-sentence version is defensible for space reasons.

### W8. ARL citation and characterization may be incomplete
- **Concern:** P2.3 lists ARL (Wei et al., ICCV 2025) with a brief description ("bias-variance decomposition for inverse-variance weighting") as the last item in a list-style sentence. However, ARL is cited in the Introduction (P3) as a significant recent paper that "challenges the assumption that balanced learning is optimal." The Related Work characterization is much weaker than the Introduction's framing. If ARL is important enough to be singled out in the Introduction, it deserves more attention in Related Work, or at minimum a consistent level of characterization.
- **Impact:** Minor. Inconsistency between sections, not a factual error.
- **Suggestion:** Either give ARL its own sentence in P2.3 (consistent with the Intro's treatment) or reduce the Intro mention to match.

---

## Questions for Authors

1. **On the "coupled monitoring" unifying claim (P2.3):** Can you provide a formal definition of "coupled" vs. "decoupled" monitoring? Currently the distinction is intuitive but not precise. Does "coupled" mean (a) gradients flow from the monitoring signal to the encoder, (b) the monitoring signal depends on the current joint-training loss, or (c) both? PMR arguably satisfies (b) but not (a). CGGM satisfies both. AUG satisfies (b). Making this precise would significantly strengthen the argument.

2. **On Data Remixing (ICML 2025):** Is Data Remixing included as a baseline in your experiments? If so, it must be discussed in Related Work. If not, why not? It achieves +6.50pp on CREMA-D and +3.41pp on KS, making it a strong competitor.

3. **On DI-MML (ACM MM 2024):** DI-MML uses "detached training" with separate unimodal and multimodal stages. How does this relate to your notion of "decoupled" probes? The terminology overlap may confuse readers.

4. **On the OGM* characterization:** The text says OGM* "fails to reduce the modality performance gap." Is the precise claim in Wei et al. (TPAMI 2024) that OGM* fails *entirely* (no improvement over baseline) or that it fails *relative to OGM-GE* (some improvement but less)? This distinction matters for the strength of the argument. If OGM* partially works but is suboptimal, the "corrupted signal" explanation needs nuancing.

5. **On the theoretical connection:** P2.1 cites Zhang et al. (ICML 2024) on architecture-dependent unimodal phase duration. Does your method explicitly leverage this theory (e.g., predicting when intervention is needed based on architecture), or is the citation purely contextual?

---

## Minor Issues

1. **P2.1, line 121:** "na\"ive" -- while technically correct LaTeX, consider using `\usepackage[utf8]{inputenc}` and writing "na\"{i}ve" or just "naive" (the accent is optional in English). The current rendering should be checked in the compiled PDF to ensure the diaeresis appears correctly.

2. **P2.2, line 141:** "Early multi-task methods such as GradNorm and PCGrad" -- these are not multimodal methods; they are multi-task methods. The distinction is acknowledged ("subsequently adapted to the multimodal setting") but the phrasing "early multi-task methods" could imply they are early chronologically within the multimodal imbalance literature, when they actually predate it. Consider: "Multi-task gradient balancing methods such as GradNorm and PCGrad, originally developed for shared-backbone multi-task learning, were subsequently adapted..."

3. **P2.3, line 163:** The sentence listing nine methods is syntactically a 7-line enumeration connected by commas. While grammatically correct, it is a heavy cognitive load. Consider grouping by intervention type with semicolons: "These include feed-forward feature modulation [Wei et al., 2024] and Pareto-optimal gradient aggregation [Wei & Hu, 2024]; modality-alternating strategies such as reconcilement regularization [Hua et al., 2024] and alternating unimodal adaptation [Zhang et al., 2024]; and diagnostic approaches including representation separability diagnosis [Wei et al., 2024b], Fisher information regulation [Huang et al., 2025], and bias-variance decomposition [Wei et al., 2025]." This imposes structure on what is currently a flat list.

4. **P2.3:** MSLR (Yao & Mihalcea, ACL 2022) is cited but receives no characterization at all; it is buried in the list as "modality-specific learning rates." Either add a brief descriptor ("which adjust per-modality learning rates based on validation performance") or, if space is tight, consider moving the citation to a parenthetical alongside related methods.

5. **P2.2:** The citation `\citet{wei2024opm}` for the TPAMI paper appears as the OPM paper. Verify that the bib entry for `wei2024opm` correctly refers to the TPAMI 2024 paper (not to be confused with OGM-GE from CVPR 2022, which is `peng2022ogmge`). The entry looks correct in references.bib, but the text says "OGM*" while the citation key says "opm" -- this could cause confusion during review if a reviewer searches for "OPM" vs. "OGM*."

6. **P2.4:** The parenthetical in "Linear probes, lightweight classifiers trained on frozen intermediate representations, were introduced by Alain & Bengio (2017)" -- Alain & Bengio 2017 is a workshop paper (ICLR Workshop). While foundational and widely cited, a reviewer may question citing a workshop paper as the primary reference. Consider adding a note or a secondary reference (e.g., the linear evaluation protocol from SimCLR, Chen et al. 2020, which popularized probe-based evaluation).

7. **P2.4:** "IPRM performs a two-pass forward strategy in which the first pass probes modality strength via KL divergence between unimodal and fused predictions; however, these probes operate on live-model outputs and thus remain coupled to the joint objective." -- The semicolon-however construction is slightly unusual after a "in which" clause. Consider splitting: "IPRM performs a two-pass forward strategy, probing modality strength in the first pass via KL divergence between unimodal and fused predictions. However, these probes operate on live-model outputs..."

---

## Coverage Audit Against 34-Paper Inventory

Papers cited in Related Work (20 papers):
- wang2020gblending, peng2022ogmge, wei2024opm, guo2024cggm, wei2024mmpareto, wei2025arl (core imbalance)
- chen2018gradnorm, yu2020pcgrad (multi-task)
- huang2021multimodal, huang2022modality, zhang2024unimodal, du2023suppression (theory)
- li2023agm, fan2023pmr, kontras2024mlgm, gao2025arm, huang2025inforeg, wei2024dr (imbalance methods)
- yang2025iprm, jiang2025aug (closest competitors)
- alain2017understanding (probing)
- yao2022mslr, hua2024reconboost, zhang2024mla (alternative approaches)
- baltrusaitis2019multimodal (survey, in Intro only)

Notable papers from inventory NOT cited:
| Paper | Venue | Relevance | Urgency |
|-------|-------|-----------|---------|
| Data Remixing (Ma et al.) | ICML 2025 | Strong CREMA-D baseline | **High** |
| DI-MML (Fan et al.) | ACM MM 2024 | "Detached" training paradigm | **High** |
| TCMax (Wu et al.) | ICLR 2026 | Most recent top-venue method | Medium |
| G2D (Rakib et al.) | ICCV 2025 | Distillation paradigm | Low-Medium |
| MILES (Guerra-Manzanares) | IJCNN 2025 | LR scheduling, lower venue | Low |
| CMoB | NeurIPS 2025 | Causal valuation, tangential | Low |
| GOAL | Under review | Not yet published | Low |
| M-SAM (Nowdeh et al.) | arXiv 2025 | Preprint only | Low |
| AIM (Shen et al.) | arXiv 2025 | Preprint only | Low |
| DynCIM | arXiv 2025 | Preprint only | Low |
| GMML | ACM MM 2025 | Convergence guarantees | Low-Medium |
| Greedy Modality Selection | UAI 2022 | Modality selection, tangential | Low |

**Verdict on coverage:** The section covers the most important papers but has two notable gaps (Data Remixing and DI-MML) that a well-prepared reviewer would flag. TCMax from ICLR 2026 is also conspicuous by its absence given that it is the most recently published method at a top venue. Adding these three would bring the coverage from good to comprehensive.

---

## Positioning Assessment

The positioning is **strong but relies on one axis**. The paper positions itself by claiming the unique combination of (1) fully decoupled probes and (2) gradient-level boosting. This is a clean, defensible position. However, there are two risks:

1. **The "coupled vs. decoupled" axis may not be perceived as sufficiently novel.** A skeptical reviewer could argue that `.detach()` is a single line of code, and the real question is whether the empirical improvements justify a paper. The Related Work sets up the argument well, but the theory needs to deliver on the promise of explaining *why* decoupling matters quantitatively, not just qualitatively.

2. **The positioning is entirely against "what others do wrong" (coupled monitoring) rather than "what we enable that is new."** The composability claim from the Introduction (combining boost with throttle for two-sided pressure) is barely mentioned in Related Work. Adding a sentence about composability in P2.2 or P2.4 would strengthen the "what we enable" angle.

---

## Fairness Assessment

The characterizations of competing methods are **generally fair**. Specific evaluations:

- **CGGM:** Fairly described as using coupled auxiliary classifiers. The text does not strawman CGGM's direction modulation capability.
- **IPRM:** Fairly described. The "live-model outputs" characterization is accurate.
- **AUG:** Briefly but accurately characterized as using "loss-based signals."
- **PMR:** Fairly described. The text gives PMR credit ("takes a step toward decoupled evaluation") before noting the limitation (loss-level intervention).
- **OGM-GE:** Fairly described as introducing the paradigm.
- **AGM/MLGM:** Cited but under-characterized. The text does not acknowledge their bidirectional claims (see W4 above).
- **ARL:** Under-characterized relative to its importance (see W8 above).

No methods are strawmanned. The main fairness risk is W4 (implying AGM/MLGM are throttle-only when they claim bidirectional modulation).

---

## Logical Flow Assessment

| Transition | Quality | Notes |
|------------|---------|-------|
| P2.1 -> P2.2 | Good | Theory -> practical response. Could be tighter with explicit bridge sentence. |
| P2.2 -> P2.3 | Excellent | "Despite this variety" signals expansion. "overwhelmingly one-sided" in P2.2 + "beyond gradient scaling" in P2.3 creates clean progression. |
| P2.3 -> P2.4 | Good | The "coupled signals" theme unifies P2.2 and P2.3, setting up the "decoupled" solution in P2.4. |

The flow is logical and reads well. The one potential issue is that P2.3's concluding sentence ("these coupled signals are precisely the quantities distorted by dominant-modality dynamics") makes a strong theoretical claim that is not backed by a citation or formal argument in P2.3 itself. It leans on P2.1's theoretical papers, but the connection is implicit. The sentence works rhetorically but a theory-minded reviewer may want a more precise argument here or a forward reference to Section 3.

---

## Balance Assessment

The section is approximately 0.85 pages, which is within the typical NeurIPS range (0.75-1.25 pages for Related Work). The four paragraphs are reasonably balanced:

- P2.1 (Theory): ~5 sentences -- appropriate
- P2.2 (Gradient modulation): ~7 sentences -- slightly long but justified by the OGM* argument
- P2.3 (Alternative approaches): ~3 sentences -- feels compressed due to the 9-method enumeration
- P2.4 (Probing & positioning): ~6 sentences -- appropriate

P2.3 is the weakest paragraph in terms of balance. It tries to cover 9 methods in 3 sentences, resulting in a dense enumeration that is hard to parse. The unifying observation at the end is valuable but arrives after a taxing list. See Minor Issue 3 for a restructuring suggestion.

---

## Overall Assessment

This is a well-structured Related Work section that successfully establishes the paper's positioning. The narrative arc from theoretical foundations through the throttle-dominant paradigm to the coupled-monitoring limitation and finally to the decoupled-probe solution is clear and compelling. The OGM* failure argument in P2.2 is the section's strongest move and provides genuine motivation for the method.

The main weaknesses are: (1) two to three important papers are missing (Data Remixing, DI-MML, possibly TCMax), which a well-prepared reviewer will notice; (2) the "all coupled" claim in P2.3 is slightly too sweeping given PMR's partial decoupling; (3) the characterization of AGM/MLGM in P2.2 dismisses their bidirectional claims without acknowledgment; and (4) the split-batch evaluation protocol, which is a genuine differentiator, is mentioned only in comments but not in the text.

None of these weaknesses are fatal. With the suggested fixes, this section would be at 8.5-9.0/10 and fully competitive for NeurIPS.

**Score: 7.5 / 10** (borderline strong; solid structure and narrative but coverage gaps and some overclaims prevent a higher score)

---

## Confidence Score

**4 / 5.** I have read the full related work text, cross-referenced against the 34-paper inventory, verified citation keys against references.bib, and checked for formatting issues. My confidence is high on the structural and positioning assessment. It is slightly lower on the characterization accuracy of Du et al. (ICML 2023) and the bidirectional claims of AGM/MLGM, as I have not read those papers' full text -- only the descriptions in RelatedWorks.md and the paper's own characterizations.

---

## Action Items (Priority Order)

1. **[High]** Add Data Remixing (Ma et al., ICML 2025) to P2.3. One clause is sufficient.
2. **[High]** Add DI-MML (Fan et al., ACM MM 2024) to P2.3 or P2.4 (its "detached training" is directly relevant to the decoupled theme).
3. **[High]** Fix the internal inconsistency about PMR: either qualify the "all coupled" claim in P2.3 or handle PMR entirely in P2.4.
4. **[Medium]** Acknowledge AGM/MLGM bidirectional claims in P2.2 before arguing the paradigm is throttle-dominated.
5. **[Medium]** Add split-batch evaluation to the P2.4 positioning sentence.
6. **[Medium]** Restructure P2.3's enumeration for readability (group by intervention type).
7. **[Low]** Add TCMax (ICLR 2026) to demonstrate up-to-date coverage.
8. **[Low]** Revise Du et al. characterization for accuracy.
9. **[Low]** Consider splitting P2.4's final sentence for readability.
10. **[Low]** Add composability angle to P2.2 or P2.4 closing.

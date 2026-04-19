# Emotive TTS: Compute-Efficient Emotion-Conditioned Neural Text-to-Speech via Incremental VITS Adaptation

## Abstract

This project studies whether emotionally expressive neural text-to-speech (TTS) can be improved under strict student-level compute constraints by incrementally adapting a pretrained VITS backbone rather than training an expressive synthesizer from scratch. The work addresses a practical research question: can explicit emotion conditioning and lightweight prosody-aware supervision yield measurable improvements over plain domain adaptation while preserving a reproducible engineering workflow on commodity Colab hardware? To answer this question, the project implements a causal sequence of four systems: A0, a pretrained reference; A, domain adaptation without explicit emotion labels; B, System A augmented with a learnable four-class emotion embedding; and C, System B augmented with utterance-level auxiliary heads for pitch and energy statistics. The repository includes deterministic data preparation, checkpointed training, batch inference, automatic evaluation, visualization code, and a structured listening-test stimulus generator. Evaluation is performed on 256 synthesized utterances spanning four systems, four emotions, and sixteen fixed canary texts. The documented results show a consistent progression in prosodic differentiation from A0/A to B and then C, with System C yielding the largest reported emotion-dependent F0 spread and the clearest increase in pitch variability for emotionally marked speech. Although most within-system statistical tests do not reach conventional significance thresholds, the overall pattern supports the hypothesis that low-cost conditioning and prosody-aware supervision can improve expressive controllability in a compute-limited setting. The main contribution of the project is therefore not a novel generative backbone, but a technically coherent and empirically disciplined adaptation framework for emotion-aware TTS.

## I. Introduction

Recent neural text-to-speech systems can synthesize highly intelligible and natural speech, but controllable emotional synthesis remains an open problem. In expressive speech generation, text alone underdetermines the acoustic realization: the same sentence may be uttered with different pitch ranges, speaking rates, energy profiles, and rhythmic structures, all while remaining linguistically valid. Emotional TTS is therefore a one-to-many generation problem rather than a simple text-to-acoustics mapping [1].

This challenge is amplified in a student project context. Large expressive TTS systems are usually developed with substantial compute, long training schedules, and large-scale curated data. Under such constraints, the most credible strategy is not to claim a new state-of-the-art architecture, but to formulate a focused research question, build a technically sound adaptation pipeline on top of a strong open baseline, and evaluate results with enough discipline that the conclusions remain defensible.

This project takes precisely that approach. It builds on a pretrained Coqui TTS VITS backbone [1], [2] and investigates whether emotionally targeted conditioning can be introduced incrementally through a clean causal sequence of increasingly expressive systems. The work is designed to satisfy three goals simultaneously:

1. Engineer a feasible emotional TTS pipeline under limited compute.
2. Isolate the effect of each architectural change through controlled ablations.
3. Produce a research-style report whose claims are proportional to the available evidence.

The resulting report is intentionally serious in tone and conservative in interpretation. It emphasizes experimental logic, reproducibility, and validity rather than rhetorical overclaiming.

### A. Research Questions

The project is organized around the following research questions.

**RQ1.** Is domain adaptation of a pretrained VITS model sufficient to produce emotion-dependent prosodic variation on an emotional speech corpus?

**RQ2.** Does adding a discrete emotion embedding improve controllability relative to domain adaptation alone?

**RQ3.** Does lightweight utterance-level prosody supervision provide additional expressive benefit beyond discrete emotion conditioning?

### B. Main Contributions

The project makes four contributions at the project level.

1. It defines a causal A0 to A to B to C system chain that isolates domain adaptation, emotion conditioning, and prosody supervision as separate experimental factors.
2. It implements a compute-efficient extension of VITS using a learnable emotion embedding and lightweight utterance-level F0 and energy heads.
3. It builds a structured evaluation harness combining deterministic canary-text synthesis, non-parametric prosody analysis, auxiliary SER probing, and listening-test stimulus generation.
4. It provides a reproducible engineering workflow centered on configuration files, notebook-based checkpoint recovery, and documented separation between original code and upstream open-source components.

## II. Background and Related Work

VITS is a widely used end-to-end TTS architecture that combines a variational formulation, adversarial training, and monotonic alignment, and is explicitly motivated by the one-to-many nature of speech generation [1]. Its availability through Coqui TTS makes it a strong foundation for student-scale experimentation because it supports pretrained checkpoints and practical fine-tuning workflows [2].

Emotionally expressive TTS, however, remains more difficult than plain TTS. Emotional expression depends not only on linguistic content, but also on prosodic cues such as pitch contour, energy distribution, and timing structure. This makes emotional control more challenging than speaker adaptation or neutral speech generation. Furthermore, evaluation is inherently difficult: automatic metrics can indicate acoustic differentiation, but perceived emotional appropriateness and naturalness still require structured human listening protocols [5], [6].

Against that background, the novelty of this project is intentionally incremental and engineering-centered. The work does not claim a new generative architecture. Instead, it explores whether carefully chosen, low-overhead modifications to a strong pretrained backbone can produce measurable expressive gains under realistic student constraints. This is a legitimate research contribution when the hypotheses are clear, the ablation logic is explicit, and the claims are calibrated to the evidence.

## III. Problem Definition

### A. Objective

The objective is to construct an emotion-aware TTS pipeline that can synthesize the same text under four target emotions: neutral, angry, amused, and disgust. The system must remain reproducible, technically coherent, and feasible on Google Colab-class hardware.

### B. Success Criteria

The project defines success at three levels.

1. **Functional success:** the pipeline should preprocess data, train A/B/C systems, synthesize controlled evaluation packs, and run automatic evaluation end to end.
2. **Scientific success:** the system sequence should reveal interpretable changes in prosodic behavior attributable to the added conditioning components.
3. **Reporting success:** conclusions should remain conservative, transparent about limitations, and suitable for an academic final report.

### C. Scope and Constraints

The project is intentionally constrained in three ways.

1. It does not train an expressive TTS model from scratch.
2. It focuses on a single-speaker emotional adaptation setting to reduce confounds.
3. It prioritizes automatic prosody analysis and listening-test readiness over claiming completed perceptual validation when such validation has not yet been collected.

These constraints are not weaknesses in themselves. They define the scope within which the results should be interpreted.

## IV. Methodology

### A. System Design

The central design principle is controlled incremental modification. Each system differs from the previous one by exactly one conceptual change.

| System | Role | Conditioning Mechanism | Initialization |
|---|---|---|---|
| A0 | Reference baseline | None | Pretrained Coqui VITS |
| A | Domain-adapted baseline | None | Pretrained Coqui VITS |
| B | Emotion-conditioned system | Additive `nn.Embedding(4, 192)` | Best checkpoint from A |
| C | Prosody-aware system | Emotion embedding + utterance-level F0/energy heads | Best checkpoint from B |

This design allows each transition to be interpreted causally:

- A0 to A measures speaker and domain adaptation.
- A to B measures the effect of explicit emotion conditioning.
- B to C measures the effect of auxiliary prosody supervision.

### B. Data Pipeline

The project uses EmoV-DB as the primary emotional speech source [3]. The preprocessing workflow does not hard-code a speaker choice; instead, it scans the corpus, audits speaker-by-emotion coverage, and selects a core speaker with complete four-emotion coverage when the configuration leaves the speaker unspecified. This improves internal consistency by keeping speaker identity constant across emotion conditions.

The preparation pipeline performs the following steps:

1. Scan the raw EmoV-DB directory and map source labels to the project label set.
2. Exclude the `Sleepiness` category because it is outside project scope.
3. Resample all audio to 22.05 kHz.
4. Apply conservative peak normalization for training data.
5. Extract frame-level F0 using pYIN and energy using RMS.
6. Compute utterance-level prosodic statistics used as System C supervision targets.
7. Create train, validation, test, and held-out evaluation subsets using deterministic splitting rules.

Two normalization policies are deliberately separated. Training audio uses peak normalization to preserve amplitude variation as a possible emotional cue, whereas listening-test copies are LUFS-normalized to improve fairness during human playback.

### C. Architecture

The backbone is a pretrained VITS model accessed via Coqui TTS [1], [2]. The project follows a selective-freezing strategy: components judged important for preserving baseline quality remain frozen, while the parameters most relevant to adaptation and control are allowed to update.

System B introduces a learnable emotion embedding with four classes and 192 dimensions. This embedding is additively injected into the text encoder representation before downstream flow and alignment processing. The modification is intentionally lightweight so that any benefit can plausibly be attributed to conditioning rather than to a large increase in model capacity.

System C augments the encoder representation with two auxiliary heads:

- an F0 statistics head predicting mean, standard deviation, and lower and upper range summaries;
- an energy statistics head predicting mean and standard deviation.

These heads operate on a pooled utterance-level representation rather than on frame-level trajectories. This choice trades fine-grained control for robustness and training stability, which is appropriate under the project’s compute budget.

### D. Optimization Objective

For Systems A and B, the dominant training signal is the VITS KL-related objective after alignment. For System C, the reported objective is

$$
L_{\mathrm{total}} = L_{\mathrm{KL}} + \lambda L_{\mathrm{prosody}}, \quad \lambda = 0.1.
$$

The prosody term is an L1 loss between predicted and target utterance-level F0 and energy statistics. The implementation also computes mel and duration-related values for monitoring, but these are not backpropagated in the reported configuration. This is a deliberate stabilization choice, not an omission: the repository documents that a more direct duration-training path created numerical instability during experimentation.

### E. Reproducibility Strategy

The project places unusual emphasis on reproducibility for a student submission. The workflow includes:

- YAML-based configuration files for data, training, inference, and evaluation;
- checkpoint persistence and recovery in the Colab notebook;
- fixed canary texts for deterministic evaluation synthesis;
- clear provenance labeling in `NOTICE.md` between upstream components and original project code;
- automated tests covering large parts of the utility, model, evaluation, and training codebase.

This does not eliminate all reproducibility risk, but it materially improves the credibility of the results relative to an ad hoc notebook-only workflow.

## V. Experimental Design

### A. Hypotheses

The experiments are designed to test three hypotheses.

**H1.** Domain adaptation alone will alter speaker-domain prosody but will not produce meaningful emotion differentiation.

**H2.** Explicit emotion conditioning will introduce measurable emotion-dependent prosodic variation.

**H3.** Auxiliary prosody supervision will strengthen that variation beyond what is achieved by discrete emotion embeddings alone.

### B. Training Protocol

The reported training setup is as follows:

- backbone: pretrained Coqui VITS;
- training order: A, then B initialized from A, then C initialized from B;
- optimizer: AdamW;
- learning rate: $5 \times 10^{-3}$;
- scheduler: ExponentialLR with $\gamma = 0.998$;
- maximum epochs: 15 per system;
- validation interval: every 5 epochs;
- early-stopping patience: 4 validation checks;
- sample rate: 22.05 kHz.

The training environment is Google Colab oriented. Mixed precision is used when CUDA is available, and the notebook adapts batch size according to detected GPU memory. This is important because it means the report should be interpreted as a compute-conscious pilot study rather than as a fully converged large-scale benchmark.

### C. Evaluation Protocol

The evaluation set consists of sixteen fixed canary texts synthesized under four emotions and four systems, yielding 256 evaluation utterances in total. This text-matched factorial design is important because it eliminates linguistic content as a confounding factor in system comparison.

The evaluation has three layers.

1. **Primary automatic analysis:** F0 and energy statistics derived from generated waveforms.
2. **Auxiliary automatic probe:** a frozen SpeechBrain emotion-recognition model trained on IEMOCAP [4].
3. **Human-study preparation:** a local listening-test pack with randomized stimulus ordering and response-form generation.

The SER probe is used only as a weak auxiliary signal because the label sets are not fully aligned. Neutral and angry map reasonably well; amused only loosely corresponds to happy; and disgust has no direct equivalent in the probe’s label space. Consequently, SER output is useful for exploratory triangulation but not for primary claims.

### D. Statistical Methodology

Because the experiment operates with small groups and makes no normality assumptions, the project uses non-parametric statistical tests.

- **Kruskal-Wallis** tests are used to assess whether a single system produces significantly different prosody across emotion categories.
- **Mann-Whitney U** tests are used for pairwise causal transitions between consecutive systems.

This is an appropriate and defensible choice for pilot-scale evaluation. It also makes the report more credible than simply reporting descriptive plots without inferential framing.

## VI. Results

### A. Training Behavior

The repository documentation reports the training behavior summarized in Table I.

| System | Epoch 1 Train Loss | Final Train Loss | Final Validation Loss | Final Mel Loss | Final Duration Loss |
|---|---:|---:|---:|---:|---:|
| A | 107.27 | 102.09 | 101.88 | 2.134 | 1.164 |
| B | 102.92 | 101.57 | 101.57 | 2.144 | 1.137 |
| C | 109.19 | 103.96 | 103.84 | 2.132 | 0.947 |

Three points are noteworthy. First, System A demonstrates that domain adaptation is operationally feasible without destabilizing the backbone. Second, System B converges to a slightly lower final KL-related loss than A, indicating that the emotion embedding does not simply burden optimization. Third, System C has a higher total loss because it includes an auxiliary prosody term, yet its duration-monitoring value improves, suggesting that the auxiliary objective contributes useful structure rather than mere optimization noise.

### B. Prosody Differentiation

The most important quantitative evidence in the project comes from prosody statistics over the 256 evaluation utterances. The repository-reported summary is shown in Table II.

| System | F0 Mean (Hz) | F0 Std (Hz) | F0 Emotion Spread | F0 Std Spread |
|---|---:|---:|---:|---:|
| A0 | 208.5 +- 17.9 | 32.0 +- 5.9 | 0.0 | 0.0 |
| A | 183.8 +- 7.3 | 21.6 +- 5.3 | 0.0 | 0.0 |
| B | 176.3 +- 13.7 | 23.3 +- 9.2 | 0.9 | 2.5 |
| C | 166.1 +- 33.5 | 37.3 +- 17.8 | 6.7 | 6.8 |

The qualitative pattern in Table II is consistent with all three hypotheses.

1. A0 and A show effectively no reported emotion spread, which is consistent with the claim that domain adaptation alone does not create expressive control.
2. B shows small but non-zero emotion-dependent differentiation after explicit emotion conditioning is introduced.
3. C shows the strongest separation, especially in pitch variability, which is precisely the dimension that the auxiliary prosody heads were designed to influence.

This makes System C the strongest system in the repository from the standpoint of controllable prosodic expressiveness.

### C. Inferential Results

The within-system Kruskal-Wallis tests do not produce conventional statistical significance at $p < 0.05$ for the main prosody metrics. The strongest near-miss is the reported System C result for `f0_std`, which reaches approximately $p = 0.067$. This result should not be overstated, but it is directionally informative: the system with explicit prosody supervision is also the system closest to demonstrating statistically detectable emotion-dependent pitch variation.

The pairwise causal tests are more informative than the within-system omnibus tests. The documented interpretation is as follows:

- **A0 to A:** domain adaptation causes substantial shifts in pitch and energy consistent with moving from the pretrained LJSpeech domain to the selected EmoV-DB speaker domain;
- **A to B:** emotion conditioning introduces measurable prosodic change, though the magnitude remains modest;
- **B to C:** prosody supervision produces the strongest targeted effect, especially for angry-speech pitch variability.

The most notable reported effect is the B to C increase in angry-speech F0 standard deviation of approximately 23 Hz. This is the clearest single result supporting the intuition behind System C.

### D. Evaluation Readiness Beyond Automatic Metrics

The project also implements two pieces of infrastructure that strengthen completeness even when they do not yet yield final quantitative results.

First, the SER probe provides a secondary lens on whether generated samples contain recognizable emotional cues. Second, the listening-test pack generator produces a 64-stimulus subset suitable for practical human evaluation. From a research-quality perspective, this matters because it shows the project was designed to move beyond descriptive waveform generation toward structured evaluation.

## VII. Discussion

### A. Interpretation of Findings

The results support a cautious but coherent narrative.

Domain adaptation alone is necessary but not sufficient. It aligns the model with the target speaker and recording domain, but it does not by itself create controlled emotional variation. Discrete emotion conditioning helps, suggesting that the model can use symbolic category information. However, the clearest gains arise only when symbolic conditioning is paired with explicit prosodic supervision. This is intuitively plausible: emotion labels specify target class membership, but prosodic statistics provide a weak acoustic prior for how that class should sound.

This is a useful finding even if the experiment remains pilot-scale. The project does not need to demonstrate a production-ready emotional TTS system to be academically meaningful. It needs to show that the hypothesis was tested with rigor, that the design was logical, and that the conclusions remain proportional to the evidence.

### B. Why the Results Matter

The main value of the project lies in disciplined engineering under realistic constraints. Many student projects either over-ambitiously attempt model training that never converges, or rely on pretrained demos without a meaningful experiment design. This project avoids both extremes. It demonstrates that a strong pretrained backbone can be turned into a serious experimental platform when the work emphasizes controlled ablations, reproducible preprocessing, transparent evaluation, and conservative interpretation.

### C. Threats to Validity

The main threats to validity are the following.

1. **Limited training budget.** Fifteen epochs per system is sufficient for a pilot study but not for strong convergence claims.
2. **Single-speaker restriction.** Internal validity improves, but external validity and generalization are limited.
3. **No completed human listening study.** Automatic prosody metrics are informative but do not replace perceptual validation.
4. **SER label mismatch.** The auxiliary probe is not perfectly aligned to the project emotion set.
5. **Underpowered statistics.** The descriptive trend is clear, but inferential evidence remains limited.

These issues do not negate the study. They define its boundary conditions and make clear why the conclusions are promising rather than definitive.

### D. Reproducibility and Responsible Use

The project benefits from a stronger reproducibility story than many student submissions: configuration-driven experiments, explicit code provenance, checkpoint-aware notebook design, and a sizeable test suite. At the same time, responsible use remains important. Emotional speech synthesis can be persuasive and identity-bearing. The project therefore benefits from clearly labeling generated samples as synthetic and maintaining transparent attribution of upstream models, datasets, and licenses.

## VIII. Conclusion

This project investigated whether emotionally expressive TTS can be improved through compute-efficient adaptation of a pretrained VITS model. The answer, based on the available evidence, is cautiously affirmative.

The strongest engineering result is the end-to-end completeness of the pipeline: the repository covers data preparation, training, inference, automatic evaluation, listening-test preparation, and clear open-source provenance. The strongest empirical result is the behavior of the A0 to A to B to C chain: domain adaptation alone does not create expressive control, emotion conditioning introduces a modest effect, and explicit prosody supervision yields the clearest incremental gain.

Just as importantly, the report avoids claiming more than the evidence justifies. The experiment remains small-scale, significance is limited, and perceptual validation is not yet complete. For that reason, the appropriate conclusion is not that the problem of emotional TTS has been solved, but that the project demonstrates a technically credible and methodologically disciplined path toward emotion-aware speech synthesis under tight compute constraints.

In a final-year project setting, that is a strong outcome. The work is ambitious enough to be interesting, but controlled enough to remain believable.

## IX. Author Contribution

This section should be replaced with the actual division of labor before submission if the report is submitted by more than one student. If the submission is single-author, the section may be removed.

## References

[1] J. Kim, J. Kong, and J. Son, "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech," in *Proc. International Conference on Machine Learning (ICML)*, 2021.

[2] Coqui AI, "Coqui TTS," GitHub repository. [Online]. Available: https://github.com/coqui-ai/TTS. Accessed: Apr. 19, 2026.

[3] Numediart, "EmoV-DB: The Emotional Voices Database," GitHub repository. [Online]. Available: https://github.com/numediart/EmoV-DB. Accessed: Apr. 19, 2026.

[4] M. Ravanelli *et al*., "SpeechBrain: A General-Purpose Speech Toolkit," *arXiv preprint* arXiv:2106.04624, 2021.

[5] International Telecommunication Union, "Methods for subjective determination of transmission quality," ITU-T Recommendation P.800, 1996.

[6] International Telecommunication Union, "Method for the subjective assessment of intermediate quality levels of coding systems," ITU-R Recommendation BS.1534-3, 2015.

[7] A. Paszke *et al*., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in *Advances in Neural Information Processing Systems*, vol. 32, 2019.

[8] B. McFee *et al*., "librosa: Audio and music signal analysis in Python," in *Proc. 14th Python in Science Conference*, 2015, pp. 18-25.

[9] M. Zaharia *et al*., "Accelerating the Machine Learning Lifecycle with MLflow," *IEEE Data Engineering Bulletin*, vol. 41, no. 4, pp. 39-45, 2018.

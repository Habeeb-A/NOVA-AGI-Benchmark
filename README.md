# NOVA Benchmark

**Novel domain Onboarding and Verification of Adaptation**

**🚧 Status: Active Work in Progress. This repository contains the core design and procedural generation engine for the NOVA Benchmark. The generation of the full 400-domain dataset and the human baseline evaluations are currently pending compute funding.**

---

## What is NOVA?

NOVA is a benchmark for the **Learning** cognitive faculty, submitted to the [Kaggle Measuring AGI: Cognitive Abilities](https://www.kaggle.com/competitions/kaggle-measuring-agi) competition and developed as part of an AI Safety rapid grant application.

NOVA addresses the single most important unresolved question in AI evaluation:

> **Do large language models learn, or do they retrieve?**

Every existing benchmark for AI learning evaluates performance on tasks whose underlying structure exists somewhere in the model's training data. This conflates retrieval with genuine acquisition. NOVA resolves this conflation by testing learning exclusively within **novel, procedurally-generated micro-domains** that cannot have been seen during pretraining — because they are invented fresh for each evaluation instance using a controlled vocabulary of non-English words.

---

## The Core Scientific Contribution: The Degradation Curve

Each NOVA instance presents a model with an invented micro-domain (e.g., a fictional physics system, a made-up social obligation ruleset) and tests it across three sequential phases:

| Phase | Name | What it tests | Score weight |
|---|---|---|---|
| **C** | Near Transfer | Apply a single rule to a new surface instance | 20% |
| **D** | Composition | Combine two rules in a sequence never shown in examples | 35% |
| **E** | Structural Transfer | Apply the abstract rule structure to a completely different surface domain | 30% |
| — | Calibration (ECE) | Does the model know when it is right? | 15% |

**The key metric is the shape of the C→D→E degradation curve, not endpoint accuracy.**

- A system that genuinely *learned* the abstract rule will show a **relatively flat curve**: it can apply the rule in new combinations (Phase D) and in new surface contexts (Phase E) almost as well as it can on near-transfer items.
- A system that *retrieved* surface patterns from training data will show a **steep downward curve**: good Phase C performance collapses at Phase D and approaches chance at Phase E.

Human learners are expected to produce a flatter curve than current LLMs. This difference is the benchmark's primary scientific output.

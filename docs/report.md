# Structured tool-call reliability under agent-harness load

**Author:** Andrew Vilenchik
**Date:** _[fill in]_
**Status:** draft

> How to use this file: the Question, Setup, and Limitations sections describe
> the experiment as built and can stand as written. The **Results** and
> **Discussion** sections are deliberately left for you to complete from your
> own run data, in your own words. Do not paste numbers you have not produced.
> This matches Anthropic's candidate AI-use guidance: the analysis and
> conclusions should be yours.

## 1. Question

An earlier informal observation: an LLM agent that passed isolated tool-call
tests emitted **zero** structured tool calls across 25 consecutive turns once
placed in a full agent harness. This study asks whether that degradation
reproduces under controlled conditions, how large it is, and whether it varies
by model and task.

## 2. Setup

- **Tasks.** Three tasks, each answerable by exactly one deterministic,
  side-effect-free tool (`add_numbers`, `lookup_record`, `format_date`). The
  correct tool and arguments are fixed and known, so scoring is unambiguous.
- **Conditions.** Each task is presented under two conditions that differ
  *only* in surrounding context:
  - _Isolated_ — one tool registered, short system prompt, no prior turns.
  - _Full-harness_ — all tools registered, a long operator-style system
    prompt, and several unrelated prior conversation turns.
- **Models.** _[list the exact models and versions you ran]_ — Claude via the
  Anthropic API and local models served over an OpenAI-compatible endpoint.
- **Trials.** _[N]_ repetitions per (model, task, condition) cell.
- **Primary metric.** Structured tool-call emission rate: the fraction of
  trials in which the model emitted a well-formed call to any tool. Secondary:
  correct-tool rate, fully-correct (tool + arguments) rate, error rate, median
  latency. Emission rates are reported with 95% Wilson score intervals.

Code and exact prompts: this repository (commit _[hash]_).

## 3. Results

_[To complete from `results/run.summary.json`. Suggested contents:]_

- A table of emission rate (with CI) per model × task × condition.
- The isolated → full-harness delta per model × task.
- Any qualitative failure modes you saw (answered in prose instead of calling
  the tool; called the wrong tool; emitted malformed arguments; looped).

| Model | Task | Isolated | Full-harness | Δ |
|-------|------|----------|--------------|---|
| _…_   | _…_  | _…_      | _…_          | _…_ |

## 4. Discussion

_[Your interpretation, in your own words. Questions worth addressing:]_

- Did the 25-turn zero-emission observation reproduce, partially reproduce, or
  not reproduce? Under what conditions?
- Which of the three context changes (prompt length, tool count, prior turns)
  seems to carry the effect? What follow-up would isolate it? (The current
  design varies all three together and cannot separate them — state that
  plainly.)
- How does Claude compare to the local models, and what would you *not*
  conclude from this small task set?

## 5. Limitations

- Three deterministic tasks keep scoring clean but are not a general capability
  estimate; emission rates here should not be read as model quality scores.
- The full-harness condition bundles three variables (prompt length, tool
  count, history) and so measures their combined effect, not their individual
  contributions.
- One long prompt stands in for "a real harness"; real orchestration frameworks
  differ. A natural extension is a factorial design that varies each context
  dimension independently.

## 6. Reproducing

See the repository README. The mock backend reproduces the degradation pattern
with no API key for pipeline testing; real numbers require the model configs in
`configs/models.yaml`.

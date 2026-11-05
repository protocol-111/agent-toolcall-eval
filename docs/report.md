# Structured tool-call reliability under agent-harness load

**Author:** Andrew Vilenchik
**Date:** _[fill in]_
**Status:** draft — Results are observed data; Discussion is to be written in the author's own words.

> Note on authorship (per Anthropic's candidate AI-use guidance): the
> experimental code and the observed numbers below are real outputs of runs the
> author executed. The **Discussion** is deliberately left to the author to
> write; the analysis and conclusions should be the author's own, and are what
> a research discussion would probe.

## 1. Question

An earlier production observation: a local-model agent that passed isolated
tool-call tests emitted **zero** accepted structured tool calls across 25
consecutive turns once placed in a full agent harness. This study asks whether
that collapse reproduces under controlled conditions, how large it is, whether
it varies by model, and — ultimately — *where in the stack* it originates.

## 2. Setup

- **Tasks.** Three tasks, each answerable by exactly one deterministic,
  side-effect-free tool (`add_numbers`, `lookup_record`, `format_date`), so
  scoring is unambiguous.
- **Models.** Claude (Opus, Haiku) via the Anthropic API; three local models
  served over an OpenAI-compatible endpoint — Llama-3.1-8B, Qwen3-Coder-30B,
  and a production-tuned Qwen variant.
- **Conditions / sweeps.** (a) single-turn isolated vs full-harness; (b)
  multi-turn agentic loop to depth 25 with 10 tools registered; (c) raw-output
  parsing, strict vs sanitizing.
- **Primary metric.** Structured tool-call emission rate, with 95% Wilson
  intervals; plus correct-tool rate, error rate, latency.

Code and exact prompts: this repository (commit _[hash]_).

## 3. Results

**3.1 Single-turn, isolated vs full-harness** (20 trials/cell, 0 errors).
Every model emitted a correct tool call on every trial in both conditions:

| Model | Isolated | Full-harness | Δ |
|-------|----------|--------------|---|
| Claude Opus | 1.00 | 1.00 | 0.00 |
| Claude Haiku | 1.00 | 1.00 | 0.00 |
| Llama-3.1-8B | 1.00 | 1.00 | 0.00 |
| Qwen3-Coder-30B | 1.00 | 1.00 | 0.00 |
| Qwen (prod-tuned) | 1.00 | 1.00 | 0.00 |

**3.2 Multi-turn depth sweep** (depth 25, 10 tools, 5 trials, 0 errors).
Emission held at 1.00 at every turn index for all five models; mean
first-failure turn = 25 (i.e. no failure before the end).

| Model | turn 0 | turn 12 | turn 24 | mean first-fail |
|-------|--------|---------|---------|-----------------|
| all five | 1.00 | 1.00 | 1.00 | 25.0 |

**3.3 Raw-output parse, strict vs sanitizing** (local models, 10 trials/task,
temperature 0; Claude excluded — native tool use emits no raw text).

| Model | strict | sanitized | recovered |
|-------|--------|-----------|-----------|
| Llama-3.1-8B | 1.00 | 1.00 | +0.00 |
| Qwen3-Coder-30B | 1.00 | 1.00 | +0.00 |
| Qwen (prod-tuned) | 1.00 | 1.00 | +0.00 |

**3.4 Instrument check (mock dirty-output generator).** To confirm the
raw-output instrument *detects* malformation when present, a generator that
reproduces the leaked-token phenomenon (control tokens, code fences, a second
JSON object, prose preambles) was parsed both ways: strict **0.28**, sanitizing
**1.00**, recovered **+0.72**, with repairs dominated by control tokens and
extra content. This verifies the parser, not the models.

## 4. Discussion

_[To be written by the author, in their own words. The data above supports a
specific, falsifiable reading; develop and defend it yourself:]_

- The collapse did **not** reproduce in any clean-room condition (single-turn,
  depth-25, raw single-shot). In isolation the models — frontier and local
  alike — emit valid tool calls reliably.
- The instrument check (3.4) shows the measurement *would* catch the production
  phenomenon if it were present in the output under test. Its absence here is
  therefore informative, not a blind spot.
- Candidate conclusion to argue: the production collapse is not a model
  generation failure but a **harness parsing / observability** failure — valid
  tool-call JSON wrapped in template control tokens (or followed by a second
  object) that a strict parser discards, turning an actively-acting agent into
  one that appears inert. What is the evidence for and against this, and what
  would falsify it?
- What you would *not* claim from this data, and why.

## 5. Methods and extensions (implemented)

- **Depth axis** (`harness.depth_eval`): real multi-turn agentic loop; emission
  by turn index; `--depth` and `--n-tools` scale independently.
- **Raw-parsing axis** (`harness.raw_probe`, `harness.parsers`): strict vs
  sanitizing parse of raw generation output; reports the recovery gap and the
  specific repairs. The honest next step is to ingest **real captured outputs**
  from the production sanitizing proxy and compute strict-vs-sanitized on them,
  rather than synthesizing dirty output.
- **Execution fencing** (`run_eval --execute --sandbox`): each emitted call runs
  in an egress-fenced user+network namespace, failing closed and reporting
  verified (not assumed) isolation. Opens a separate honesty question: when a
  network tool legitimately fails inside the fence, does the model report the
  failure or fabricate a result?

## 6. Limitations

- Three deterministic tasks keep scoring clean but are not a capability
  estimate; the 1.00 rates are not model quality scores.
- The clean-room conditions are, by construction, simpler than the production
  harness whose template and streaming are the suspected trigger; absence of
  the collapse here does not mean it cannot occur there.
- The strict→sanitized gap has been demonstrated only on synthetic dirty output
  (3.4). The production claim in §4 requires analysis of real captured outputs,
  which is pending.

## 7. Reproducing

See the repository README. The `--mock` paths reproduce every pipeline with no
API key or GPU; real numbers require model access configured via environment
variables (never committed).

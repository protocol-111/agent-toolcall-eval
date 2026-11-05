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

## 5a. Depth axis: multi-turn agentic loop

The single-turn study (above) found no isolated→full-harness degradation on the
toy tasks for any model tested. This is reported as a null result: under light,
single-turn load, structured tool-call emission is robust. The original
25-turn collapse was not reproduced because the single-turn design cannot
exercise conversation depth — a turn-25 effect is unreachable in a one-turn
test.

`harness.depth_eval` runs a real multi-turn agentic loop and measures emission
rate by turn index, sweeping two stressors independently: conversation depth
(`--depth`) and registered tool count (`--n-tools`). The research questions to
develop from your own runs:

- Is there a turn depth at which emission falls, and does it differ by model?
  (Report the emission-by-turn curve and the mean first-failure turn.)
- Does tool count shift that threshold? (Sweep `--n-tools` at fixed depth.)
- Do your Strix-tuned local models collapse earlier than frontier models, and
  does the reduced-harness profile you already use delay the collapse? This
  connects the harness to the behavior you observed in production.

## 5b. The mechanism: strict vs sanitizing parsing of raw output

Neither the single-turn study nor the depth sweep reproduced the collapse,
because both reach models through an OpenAI-compatible endpoint that normalizes
tool calls before scoring. The production failure lives in the raw output: the
model emits a valid tool call wrapped in stray control tokens or followed by a
second JSON object, and a strict harness parser discards it. Across 25 turns
that reads as zero tool calls — an agent that appears inert while actively
trying to act.

`harness.raw_probe` isolates this: it takes raw generation output and parses it
with a strict parser and a sanitizing parser side by side, reporting both
emission rates and the specific repairs the sanitizer had to make. The
strict→sanitized gap quantifies the collapse, and also quantifies how much a
sanitizing proxy recovers.

Claims to develop from your own runs (keep them yours):

- The reframed finding: the collapse is a **parsing/observability** failure in
  the harness, not a generation failure in the model. Support it with the
  per-model strict vs sanitized rates and the repair breakdown (control tokens,
  code fences, second objects).
- The safety angle: a strict harness silently converts "model attempted an
  action" into "model did nothing," which is the dangerous direction for an
  agent you are monitoring — it hides activity rather than surfacing it.
- The fix you already built: report how much the sanitizer recovers, and note
  the residual cases it cannot (genuinely malformed or absent calls), because
  "recovered everything" would itself be an overclaim.

## 5a. Second axis: execution fencing and honesty under tool failure

The harness can execute each emitted tool call inside an egress-fenced sandbox
(`--execute --sandbox`): a process in its own user+network namespace, no
outbound network, resource-limited, failing closed if isolation can't be
established. This reproduces the "network restrictions" factor from the
original observation as a controlled variable rather than an anecdote.

The research question it opens, in your own words to develop: when a
network-dependent tool *legitimately fails* inside the fence, does the model
(a) report the failure honestly, (b) retry/adapt, or (c) fabricate a plausible
result it never obtained? Case (c) is the same failure class as the unsupported
vulnerability claim you observed earlier — a grounding/honesty failure under
tool error. The fence makes that condition reproducible on demand.

Measurement note: distinguishing "reached" from "blocked" requires a network
where the target host is actually reachable in the unfenced baseline. Record
the baseline reachability alongside results so a blocked result is attributed
to the fence, not to an already-unreachable host.

## 6. Reproducing

See the repository README. The mock backend reproduces the degradation pattern
with no API key for pipeline testing; real numbers require the model configs in
`configs/models.yaml`.

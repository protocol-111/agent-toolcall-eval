# Structured tool-call reliability under agent-harness load

**Author:** Andrew Vilenchik
**Date:** November 5, 2026

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

Code and exact prompts: this repository (`main`).

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
Emission held at 1.00 at every turn index for all five models. No failures were observed through turn 25; first-failure time was therefore not observed within the evaluation horizon.

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

### 4.1 Failure to reproduce the original observation

The original investigation identified a substantial discrepancy between
isolated model performance and behavior within a production agent framework.
Although the evaluated model successfully generated structured tool calls under
simplified conditions, the production harness recorded zero accepted tool calls
across 25 consecutive turns.

The controlled experiments conducted in this study did not reproduce that
failure. Across five models, including Claude Opus, Claude Haiku, and three
locally hosted models, structured tool-call emission remained at 100% in both
isolated and full-harness single-turn evaluations. Additional multi-turn
experiments extending to 25 turns with 10 registered tools similarly produced
no observed degradation.

These findings do not establish that agent-harness complexity has no effect on
tool-use reliability. Rather, they demonstrate that the specific complexity
introduced by the controlled evaluation was insufficient to reproduce the
original behavior. The controlled harness omits three features of the
production environment that are the leading suspects: its exact prompt
template, its streaming output path, and the strict parser that processed raw
model output.

### 4.2 Distinguishing model generation from harness interpretation

A central implication of these results is the importance of distinguishing
model-generated behavior from the behavior ultimately recorded by an agent
framework.

An agent may generate an intended tool invocation, yet the surrounding system
may fail to recognize or execute it because of formatting, parsing, or
protocol-handling requirements. Consequently, the absence of accepted tool
calls in an execution log does not necessarily establish that the underlying
model failed to attempt tool use.

The raw-output experiments investigated this distinction by comparing strict
parsing against a sanitizing parser capable of recovering structured calls from
outputs containing extraneous formatting or control tokens. Under the tested
real-model conditions, both parsers accepted 100% of outputs across the three
local models. A separate synthetic malformed-output experiment, however,
produced acceptance rates of 28% under strict parsing and 100% under sanitizing
parsing.

This establishes that parser behavior can materially influence measured
tool-call reliability when malformed output is present. It does not establish
that parsing caused the original production failure, because that mechanism has
not yet been verified against captured production outputs.

### 4.3 Implications for AI agent reliability and safety evaluation

The findings suggest that evaluating autonomous AI agents solely through
successful tool execution can obscure important distinctions between model
capability, interface compatibility, and execution-system reliability.

A reported tool-use failure may originate at several different stages:

- Failure of the model to select or generate the appropriate tool call.
- Generation of a tool call that does not conform to the required interface.
- Rejection or misinterpretation of the generated call by the harness.
- Failure of the execution environment after a valid call has been accepted.

These failure modes have different implications for model evaluation and system
design. Treating them as equivalent risks attributing infrastructure failures
to model behavior, or overlooking genuine model limitations.

For safety-critical agent systems, evaluation should therefore preserve enough
information to distinguish raw model output, parser decisions, accepted tool
invocations, execution results, and the agent's subsequent interpretation of
those results. This matters most when agents drive security tools or external
systems: a harness that silently drops a rejected command can leave an agent
reporting success for an action it never completed, so a parsing-layer gap
becomes a correctness-and-safety problem in the agent's own output.

### 4.4 Limitations and future research

The present study used a limited set of deterministic tasks, selected to permit
unambiguous measurement of tool-call correctness. These tasks do not capture
the full complexity of extended autonomous workflows. The controlled harness
also did not reproduce every aspect of the original production environment,
including its precise prompt templates, streaming behavior, and output-
processing mechanisms.

The most important next experiment is therefore to evaluate actual raw outputs
captured from the original production framework using both strict and
sanitizing parsers. This would permit direct measurement of whether apparently
missing tool calls were generated but rejected during processing.

A second direction is to examine how agents respond when valid tool calls fail
during execution. Network-isolated execution environments could be used to
determine whether models accurately acknowledge unavailable information or
instead generate unsupported claims about actions they could not complete. Both
investigations would help distinguish failures of model behavior from failures
introduced by the systems surrounding the model.

## 5. Conclusion

This study investigated whether structured tool-call failures observed in a
production AI agent could be reproduced under controlled experimental
conditions.

**The principal finding is that the original failure did not reproduce across
the evaluated models, tasks, and harness configurations.** Models that
successfully generated tool calls in isolation continued to do so under the
tested multi-turn and expanded-tool conditions.

A separate parser experiment demonstrated that strict output validation can
reject otherwise recoverable structured calls when additional formatting or
control tokens are present. This mechanism was demonstrated using synthetic
malformed outputs and remains an unconfirmed explanation for the original
production observation.

The broader contribution is methodological: **agent reliability cannot be
evaluated accurately without distinguishing model generation, output
interpretation, and tool execution.** Future work should prioritize
production-output replication, parser-level observability, and controlled
evaluations of agent behavior following genuine execution failures — all of
which support more reliable agent evaluation and guard against misleading
conclusions about both model capabilities and the safety of autonomous AI
systems.

## 6. Reproducing

See the repository README. The `--mock` paths reproduce every pipeline with no
API key or GPU; real numbers require model access configured via environment
variables (never committed).

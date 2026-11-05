# Agent tool-call reliability under harness load

![tests](https://github.com/protocol-111/agent-toolcall-eval/actions/workflows/ci.yml/badge.svg)

A small, reproducible study of a single question:

> When an LLM agent reliably emits a well-formed structured tool call in
> isolation, does it keep doing so once the same task is wrapped in a realistic
> agent harness — a long operator prompt, many registered tools, and prior
> conversation turns?

This rebuilds, as a controlled experiment, an earlier production observation of
**zero accepted tool calls across 25 full-harness turns despite successful
isolated tool-call tests**. The effect is measured across models and tasks with
repeated trials and confidence intervals rather than a single run.

## Findings so far

- **In isolation, models are robust.** Across single-turn, 25-turn depth, and
  raw single-shot conditions, Claude (Opus/Haiku) and three local models all
  emit valid tool calls at ~1.00 — the collapse does **not** reproduce in any
  clean-room condition.
- **The likely mechanism is parsing, not generation.** A strict harness parser
  discards valid tool-call JSON that carries stray control tokens or a trailing
  second object, so an actively-acting agent records zero calls. The
  raw-output probe demonstrates the instrument catches this when present
  (strict 0.28 → sanitizing 1.00 on synthetic dirty output).
- **Status:** the remaining step is to run the strict-vs-sanitizing analysis on
  *real* captured outputs from a production harness, rather than synthetic ones.
  See [`docs/report.md`](docs/report.md) for the full write-up and the open
  question. This repo is kept as a reference for that work.

## Design

| | Isolated | Full-harness |
|---|---|---|
| Tools registered | 1 (the relevant one) | all |
| System prompt | short | long, operator-style |
| Prior turns | none | several unrelated |
| Task & correct answer | identical across conditions | identical across conditions |

Only the surrounding context changes, so a drop in the tool-call emission rate
between conditions isolates the effect of harness load. Each (model, task,
condition) cell is repeated `--trials` times; the summary reports the emission
rate with a 95% Wilson interval.

## Run it

No API key needed for the mock sweep (it reproduces the degradation pattern so
the pipeline can be tested and demoed):

```bash
pip install -r requirements.txt
python -m harness.run_eval --config configs/mock.yaml --trials 30 --out results/mock.jsonl
```

Real models — Claude via the Anthropic API alongside local models served over
an OpenAI-compatible endpoint (vLLM / Ollama / LiteLLM):

```bash
export ANTHROPIC_API_KEY=...            # never committed; read at run time
export LOCAL_OPENAI_BASE_URL=http://localhost:8000/v1
python -m harness.run_eval --config configs/models.yaml --trials 20 --out results/run.jsonl
```

## Multi-turn depth sweep (the turn-depth stressor)

The single-turn study above found no degradation on easy tasks. The original
observation — zero tool calls across 25 *consecutive* turns — lives in a deep
agentic loop, which a single-turn test cannot reproduce. This sweep runs a real
loop of `--depth` turns (the model's own tool calls and their results
accumulate in context) with `--n-tools` tools registered, and reports emission
rate **by turn index** plus the mean turn at which emission first fails.

```bash
python -m harness.depth_eval --config configs/mock.yaml \
    --depth 20 --n-tools 10 --trials 5 --out results/depth.jsonl
```

Two knobs to scale the stressor until emission breaks, which localizes the
cause: `--depth` (conversation depth) and `--n-tools` (how many tools the model
must choose among). Sweep each while holding the other fixed.

## Raw-output probe: strict vs sanitizing parser (the real mechanism)

The depth sweep found no collapse, because it calls models through the
OpenAI-compatible endpoint, which pre-normalizes tool calls. The production
failure was upstream of that: the model emitted tool-call JSON wrapped in stray
control tokens (`<|im_end|>`, `<|tool_call_end|>`) or followed by a second JSON
object, and a strict harness parser rejected it — so an actively-acting agent
recorded zero tool calls.

This probe hits the model's RAW generation endpoint and parses each output two
ways — strict (whole output must be one clean JSON call) and sanitizing (strip
control tokens/fences, take the first balanced JSON object) — reporting both
emission rates. The gap is what a sanitizing proxy recovers.

```bash
python -m harness.raw_probe --mock --trials 40 --out results/raw.jsonl   # no network
python -m harness.raw_probe --config configs/models.yaml --trials 10 \
    --out results/raw.jsonl                                              # real local models
```

Reads Ollama's native `/api/generate` (host from `OLLAMA_URL`, or derived from
`LOCAL_OPENAI_BASE_URL`). Anthropic models are skipped — native tool use emits
no raw text to parse. The result reframes the finding: the collapse is a
harness **parsing/observability** failure, not model incapacity.

## Egress-fenced tool execution (sandbox)

Tool calls can be executed inside a "sacrificial box" style fence: a child
process in its own user+network namespace (rootless `unshare -rn`), under
CPU/memory/wall-clock limits, with no outbound network. This models running
agent tools where execution is isolated from everything else.

```bash
python -m harness.sandbox          # self-check: proves the fence blocks outbound
python -m harness.run_eval --config configs/mock.yaml --trials 10 \
    --out results/run.jsonl --execute --sandbox   # fence each emitted call
```

Two deliberate design choices, both load-bearing:

- **Fail closed.** With `--sandbox`, if the namespace cannot be created the run
  *refuses* rather than silently executing a tool unfenced.
- **Verified, not assumed.** Each run records `net_isolation_enforced` as a
  checked fact, and `harness.sandbox.verify_fence()` actually attempts an
  outbound connection and confirms it is blocked. The harness never reports an
  isolation it did not achieve.

This also sets up a second study: when a network-dependent tool legitimately
fails inside the fence, does the model report the failure honestly or fabricate
a result? That ties the harness-load finding to grounding/honesty behavior.

> Note: in this project's cloud container, outbound egress is already
> restricted, so the *unsandboxed* baseline also cannot reach an arbitrary host
> — the reach-vs-blocked contrast is only visible on a network where the target
> host is actually reachable. The fence's effect (namespace created, outbound
> blocked) is still verified independently.

## On k3s

Built to run as a batch `Job` on a k3s cluster; see [`k8s/`](k8s/). Results are
written to a `local-path` PVC. Secrets are mounted from a Kubernetes secret and
never baked into the image.

## Layout

```
harness/     run_eval.py (sweep + stats), clients.py (Claude/local/mock), tasks.py
tools/       three deterministic, side-effect-free tools under test
configs/     model lists (mock.yaml needs no key; models.yaml for real runs)
k8s/         k3s Job + PVC manifests
docs/        write-up / technical report
```

## Limitations

The task set is small and deterministic by design, which keeps scoring
unambiguous but means emission rates are not a general capability estimate.
The long system prompt is a stand-in for one real harness, not a survey of
harness designs. See [`docs/report.md`](docs/report.md) for the full list.

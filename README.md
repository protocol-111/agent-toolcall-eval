# Agent tool-call reliability under harness load

A small, reproducible study of a single question:

> When an LLM agent reliably emits a well-formed structured tool call in
> isolation, does it keep doing so once the same task is wrapped in a realistic
> agent harness — a long operator prompt, many registered tools, and prior
> conversation turns?

This rebuilds, as a controlled experiment, an earlier observation of **zero
structured tool calls across 25 full-harness turns despite successful isolated
tool-call tests**. Here the effect is measured across models and tasks with
repeated trials and confidence intervals rather than a single run.

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

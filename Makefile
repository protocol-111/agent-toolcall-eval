.PHONY: setup test fence lint clean
# Scope pytest to this project's own tests so a stray repo elsewhere on the
# machine can never be collected.
TESTS = harness tools tests

setup:
	python3 -m venv .venv && . .venv/bin/activate && \
	pip install -r requirements.txt -r requirements-dev.txt

test:
	python -m pytest -q $(TESTS)

fence:
	python -m harness.sandbox

clean:
	rm -rf results/*.jsonl results/*.summary.json results/figs __pycache__ \
	       harness/__pycache__ tools/__pycache__ tests/__pycache__

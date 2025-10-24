PYTHON ?= python
PIP ?= $(PYTHON) -m pip
VENV_DIR ?= .venv
ACTIVATE = . $(VENV_DIR)/bin/activate

.PHONY: dev test docker clean run-sft run-rl run-funsearch run-numerics fmt lint sbom

dev:
	$(PYTHON) -m venv $(VENV_DIR)
	$(ACTIVATE) && $(PIP) install --upgrade pip
	$(ACTIVATE) && $(PIP) install -e .[dev]

test:
	$(ACTIVATE) && pytest -q

fmt:
	$(ACTIVATE) && ruff format .

lint:
	$(ACTIVATE) && ruff check .

run-sft:
	$(ACTIVATE) && python prover/train/sft_train.py --config orchestration/configs/sft_tiny.yaml --max-steps 100

run-rl:
	$(ACTIVATE) && python prover/rl/run_selfplay.py --config orchestration/configs/rl_tiny.yaml --walltime 60

run-funsearch:
	$(ACTIVATE) && python conjecture/funsearch/funsearch_loop.py --config orchestration/configs/funsearch_loop.yaml

run-numerics:
	$(ACTIVATE) && python numerics/zeros/run_zero_checks.py --config orchestration/configs/numeric_verify.yaml

docker:
	docker build -f docker/base.Dockerfile -t archon-rh/base .
	docker build -f docker/trainer.Dockerfile -t archon-rh/trainer .
	docker build -f docker/lean.Dockerfile -t archon-rh/lean .
	docker build -f docker/numerics.Dockerfile -t archon-rh/numerics .
	mkdir -p artifacts/sbom
	syft packages archon-rh/base -o json > artifacts/sbom/base.json
	syft packages archon-rh/trainer -o json > artifacts/sbom/trainer.json
	syft packages archon-rh/lean -o json > artifacts/sbom/lean.json
	syft packages archon-rh/numerics -o json > artifacts/sbom/numerics.json

clean:
	rm -rf $(VENV_DIR) artifacts checkpoints

# Reproducible pipeline. `make all` regenerates every number in the paper.
PY ?= python3
EPISODES ?= 8
STEPS ?= 500
SEEDS ?= 12

.PHONY: all setup dataset train experiments sweep figures test clean docker

all: dataset heldout train experiments robustness figures paper

setup:
	$(PY) -m pip install -r requirements.txt

dataset:
	$(PY) -m python.simulator.dataset_generator --episodes $(EPISODES) --steps $(STEPS) \
		--out data/processed/telemetry_dataset.parquet

train:
	$(PY) -m python.predictor.train --data data/processed/telemetry_dataset.parquet

experiments:
	$(PY) -m python.experiments.run_experiments --seeds $(SEEDS) --steps $(STEPS)

sweep:
	$(PY) -m python.experiments.run_experiments --sweep --seeds 3 --steps 400

robustness:
	$(PY) -m python.experiments.robustness ablation
	$(PY) -m python.experiments.robustness topology --seeds 8
	$(PY) -m python.experiments.robustness ood
	$(PY) -m python.experiments.robustness noise
	$(PY) -m python.experiments.robustness horizon --seeds 6
	$(PY) -m python.experiments.robustness migration --seeds 6

heldout:
	$(PY) -m python.simulator.dataset_generator --episodes $(EPISODES) --steps $(STEPS) \
		--scenarios S5_leo_handover S6_common_cause \
		--out data/processed/heldout_dataset.parquet

figures:
	$(PY) -m python.experiments.make_figures2

paper:
	$(PY) -m python.paper.build_paper_ress --out "../Predictive_Resilience_Hybrid_5G6G_LEO.docx"
	$(PY) -m python.paper.build_paper_ua   --out "../submission/Прогнозована_стійкість_UA.docx"
	$(PY) -m python.paper.build_latex --docx "../Predictive_Resilience_Hybrid_5G6G_LEO.docx" \
		--out ../submission/latex
	$(PY) -m python.paper.build_bib --out ../submission/latex/references.bib

test:
	$(PY) -m pytest tests -q

docker:
	docker compose up --build

clean:
	rm -rf data/processed/*.parquet data/results/*.csv paper-assets/figures/*.png

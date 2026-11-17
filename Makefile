# Reproducible pipeline. `make all` regenerates every number in the paper.
PY ?= python3
EPISODES ?= 8
STEPS ?= 500
SEEDS ?= 12

.PHONY: all setup dataset train experiments sweep figures test clean docker campaign sweeps confirmation paper qa

all: dataset heldout train campaign sweeps confirmation figures paper qa

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

SWEEPS ?= detection_delay frontier migration_cost_sweep misspec_closed_loop large_topology \
	feedback_ablation_fixed design_sensitivity topology_sweep parameter_sensitivity \
	controller_weight_sensitivity accounting_check endogeneity_check protection_modes \
	p6_decomposition refresh_control
WORKERS ?= 4
OUT ?= ../submission_R1

campaign:
	$(PY) -m python.experiments.campaign_chunks run 0 168 $(WORKERS)
	$(PY) -m python.experiments.campaign_chunks merge

sweeps:
	for e in $(SWEEPS); do n=$$($(PY) -m python.experiments.revision_sweeps count $$e | tail -1); \
		$(PY) -m python.experiments.revision_sweeps run $$e 0 $$n $(WORKERS); \
		$(PY) -m python.experiments.revision_sweeps merge $$e; done
	$(PY) -m python.experiments.robustness horizon --seeds 12

confirmation:
	$(PY) -m python.experiments.revision_sweeps run confirmation 0 256 $(WORKERS)
	$(PY) -m python.experiments.revision_sweeps merge confirmation
	$(PY) -m python.experiments.confirmation_analysis

figures:
	$(PY) -m python.experiments.make_figures_r1

paper:
	mkdir -p $(OUT)
	$(PY) -m python.paper.build_paper_r1 --out $(OUT)/Manuscript_R1.docx
	$(PY) -m python.paper.build_response_r1 --out $(OUT) --captions $(OUT)/Manuscript_R1.docx.captions.json
	$(PY) -m python.paper.set_metadata $(OUT)
	$(PY) -m python.paper.build_latex --docx $(OUT)/Manuscript_R1.docx --out $(OUT)/els-article-R1

qa:
	$(PY) -m python.paper.qa_r1 --docx $(OUT)/Manuscript_R1.docx --pkg $(OUT)

test:
	$(PY) -m pytest tests -q

docker:
	docker compose up --build

clean:
	rm -rf data/processed/*.parquet data/results/*.csv paper-assets/figures/*.png

# SIGINT-Fusion developer entry points. Python targets expect an activated venv (see README "Setup").
COMPOSE ?= docker compose
GPU ?= 0
ifeq ($(GPU),1)
COMPOSE += -f docker-compose.yml -f docker-compose.gpu.yml
endif

.PHONY: help venv data train lint test test-java web-build up down logs ingest sim bench rag-eval agent-eval smoke report clean

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*##' '{printf "  %-12s %s\n", $$1, $$2}'

venv:            ## create .venv (python 3.11) with CUDA torch + all dev deps
	uv venv --python 3.11 .venv && . .venv/bin/activate && uv pip install --index-url https://download.pytorch.org/whl/cu128 torch \
	&& uv pip install -r ml-service/requirements-train.txt -r agent/requirements.txt && uv pip install "numpy<2"

data:            ## generate the synthetic RadioML-layout datasets
	cd ml-service && python -m data.synth_mod --out data/synth_2018.hdf5 --length 1024 --per-class 4000 --workers 16 \
	&& python -m data.synth_mod --out data/synth_2016.pkl --length 128 --per-class 1000 --format 2016 --workers 16 --seed 1

train:           ## full experiment ladder (GPU) -> ml-service/results/*.json + ckpt/ft.onnx
	scripts/run_experiments.sh

lint:            ## ruff
	ruff check ml-service rag agent scripts

test:            ## python unit/service tests (CPU)
	cd ml-service && CUDA_VISIBLE_DEVICES= python -m pytest -q tests fusion
	cd rag && python -m pytest -q tests

test-java:       ## gateway unit tests via the maven container
	cd gateway-java && docker run --rm -v "$$PWD":/src -w /src -v sigint-m2:/root/.m2 -e HTTPS_PROXY -e HTTP_PROXY \
	  -e MAVEN_OPTS="$$MAVEN_OPTS" maven:3.9-eclipse-temurin-21 mvn -q -B package

web-build:       ## typecheck + build the console
	cd web && npm ci && npm run typecheck && npm run build

up:              ## build and start the platform (GPU=1 for Ollama on GPU)
	$(COMPOSE) up -d --build

down:            ## stop everything (keeps volumes)
	$(COMPOSE) down

logs:            ## follow logs
	$(COMPOSE) logs -f --tail=100

ingest:          ## (re)ingest the RAG corpus into pgvector
	$(COMPOSE) run --rm rag-ingest

sim:             ## start the sensor simulator profile
	$(COMPOSE) --profile sim up -d sensor-sim

bench:           ## /classify load test against the running ml-service
	cd ml-service && python benchmark/load.py --url http://localhost:8000 --n 2000 --conc 32 --label "$(BENCH_LABEL)"

rag-eval:        ## retrieval hit@5 / MRR on rag/questions.jsonl
	cd rag && python eval.py questions.jsonl

agent-eval:      ## agent task success on agent/tasks.jsonl (needs the LLM)
	cd agent && python eval_agent.py tasks.jsonl

smoke:           ## end-to-end smoke through nginx on :$(WEB_PORT)
	scripts/smoke.sh

report:          ## render docs/results.md from results JSON
	python scripts/report.py

clean:
	rm -rf ml-service/results/smoke_* .pytest_cache .ruff_cache

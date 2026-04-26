.PHONY: help install preprocess train submit

# Default parameters
CONFIG ?= configs/config.yaml
CKPT ?= ""
AUG_COPIES ?= 0
WORKERS ?= $(shell nproc 2>/dev/null || sysctl -n hw.logicalcpu)

help:
	@echo "Available commands:"
	@echo "  make install                                     - Install dependencies using uv"
	@echo "  make preprocess [AUG_COPIES=N] [WORKERS=N]      - Preprocess; N augmented copies, N parallel workers"
	@echo "  make train                                       - Start model training (requires wandb login)"
	@echo "  make submit CKPT=path/to/checkpoint.ckpt        - Generate Kaggle submission"

install:
	uv sync

preprocess:
	PYTHONPATH=$PYTHONPATH:./src .venv/bin/python src/dataset/preprocess.py --train_csv data/train.csv --dev_csv data/dev.csv --out_dir data/preprocessed --aug_copies $(AUG_COPIES) --workers $(WORKERS)

train:
	PYTHONPATH=$PYTHONPATH:./src .venv/bin/python src/train/train.py --config $(CONFIG)

submit:
	@if [ -z "$(CKPT)" ]; then \
		echo "Error: You must provide a checkpoint path. Usage: make submit CKPT=path/to/model.ckpt"; \
		exit 1; \
	fi
	PYTHONPATH=$PYTHONPATH:./src .venv/bin/python src/utils/submission.py --config $(CONFIG) --ckpt $(CKPT)


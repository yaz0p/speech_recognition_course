.PHONY: help install preprocess sample train submit

# Default parameters
CONFIG ?= configs/config.yaml
CKPT ?= ""

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies using uv"
	@echo "  make preprocess - Run data preprocessing (extract waveforms to .pt files)"
	@echo "  make sample     - Generate 10 examples of augmented audio for listening"
	@echo "  make train      - Start model training (requires wandb login)"
	@echo "  make submit CKPT=path/to/checkpoint.ckpt - Generate Kaggle submission"

install:
	uv sync

preprocess:
	.venv/bin/python src/dataset/preprocess.py --train_csv data/train.csv --dev_csv data/dev.csv --out_dir data/preprocessed

sample:
	.venv/bin/python src/dataset/sample_aug.py --csv data/train.csv --prep_dir data/preprocessed --out_dir data/samples --num 10

train:
	.venv/bin/python src/train/train.py --config $(CONFIG)

submit:
	@if [ -z "$(CKPT)" ]; then \
		echo "Error: You must provide a checkpoint path. Usage: make submit CKPT=path/to/model.ckpt"; \
		exit 1; \
	fi
	.venv/bin/python submission.py --config $(CONFIG) --ckpt $(CKPT)


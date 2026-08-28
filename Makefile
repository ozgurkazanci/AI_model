# ASIC-AI Makefile
# Common tasks for development

PYTHON = python
PYTHONPATH_CMD = PYTHONPATH=src
PYTEST = $(PYTHONPATH_CMD) $(PYTHON) -m pytest

.PHONY: test lint demo train analyze chat benchmark clean help

help:  ## Show this help
	@echo.
	@echo   ASIC-AI Development Tasks
	@echo   =========================
	@echo.
	@echo   make test        - Run all tests
	@echo   make demo        - Run pipeline demo
	@echo   make analyze     - Analyze SFT training data
	@echo   make prepare     - Prepare optimized training dataset
	@echo   make train-quick - Quick fine-tune test (15 min CPU)
	@echo   make train-full  - Full fine-tune (8 hours CPU)
	@echo   make chat        - Interactive chat with model
	@echo   make benchmark   - Benchmark inference speed
	@echo   make eval        - Run eval on 5 tasks
	@echo   make monitor     - Check training progress
	@echo   make merge       - Merge LoRA into base model
	@echo   make clean       - Clean outputs
	@echo.

test:  ## Run all tests
	$(PYTEST) tests/ -v --tb=short

demo:  ## Run full pipeline demo
	$(PYTHONPATH_CMD) $(PYTHON) scripts/demo_full_pipeline.py

analyze:  ## Analyze SFT training data
	$(PYTHONPATH_CMD) $(PYTHON) scripts/analyze_sft_data.py

prepare:  ## Prepare optimized training dataset
	$(PYTHONPATH_CMD) $(PYTHON) scripts/prepare_training_data.py

train-quick:  ## Quick fine-tune test (15 min)
	$(PYTHONPATH_CMD) $(PYTHON) scripts/finetune_local.py --quick-test

train-full:  ## Full fine-tune (8 hours CPU)
	$(PYTHONPATH_CMD) $(PYTHON) scripts/finetune_local.py --epochs 3

chat:  ## Interactive chat with model
	$(PYTHONPATH_CMD) $(PYTHON) scripts/chat.py --model outputs/sft_local/final

benchmark:  ## Benchmark inference speed
	$(PYTHONPATH_CMD) $(PYTHON) scripts/benchmark.py

eval:  ## Run eval (first 5 tasks)
	$(PYTHONPATH_CMD) $(PYTHON) scripts/run_eval.py --limit 5

monitor:  ## Check training progress
	$(PYTHONPATH_CMD) $(PYTHON) scripts/training_monitor.py

merge:  ## Merge LoRA adapter into base model
	$(PYTHONPATH_CMD) $(PYTHON) scripts/merge_lora.py --test

clean:  ## Clean training outputs
	rm -rf outputs/sft_local/checkpoint-*
	@echo "Cleaned checkpoints (kept final/)"

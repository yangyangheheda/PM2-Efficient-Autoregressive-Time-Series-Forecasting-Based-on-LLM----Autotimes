#!/usr/bin/env bash
set -e
python -m pip install -r requirements.txt
python src/run_experiment.py -c configs/etth1_gpt2.yaml
python src/run_experiment.py -c configs/lora_etth1.yaml
python src/run_experiment.py -c configs/weather_gpt2.yaml
python src/make_table.py checkpoints results.csv --latex results.tex
python src/generate_article_fig.py

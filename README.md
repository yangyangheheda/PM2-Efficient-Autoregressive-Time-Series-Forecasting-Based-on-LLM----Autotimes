# PM2-Efficient-Autoregressive-Time-Series-Forecasting-Based-on-LLM----Autotimes
# AutoTimes – DSCI-498 Final Project
Efficient Autoregressive Time Series Forecasting Based on LLM -- Autotimes

---

## 1. Project Description  
We freeze a decoder‑only LLM (default GPT‑2‑small, 124 M) and train two lightweight MLPs (SegmentEmbedding and SegmentProjection) to forecast multivariate time‑series.  
Extra contributions:  

LoRA(rank 8, α 16) for parameter‑efficient fine‑tuning  
Mixed‑variable training：flag for dense sensor data  
One‑click script 'run_all.sh' to reproduce all experiments, collect results, and generate publication‑ready tables / figures.

---

## 2. Data Source  

ETTh1, Weather: https://drive.google.com/file/d/1t7jOkctNJ0rt3VMwZaqmxSuA75TFEo96/view

All datasets files are already in the data folder.

---

## 3. Required Packages  

python  >= 3.9
torch   >= 2.1
transformers >= 4.40
peft    >= 0.8
pandas, numpy, tqdm, scikit-learn, pyyaml, matplotlib

Install all with

pip install -r requirements.txt

---

## 4. How to Run  

### Quick demo (one‑line)

bash scripts/run_all.sh

This will

1. install Python packages  
2. train **three** experiments (frozen GPT‑2, LoRA‑GPT‑2, Weather mixed‑var)  
3. write metrics to `checkpoints/*/result.txt`  
4. generate `results.csv`, `results.tex`, and `mse_bar.png`

### Custom experiment

python src/run_experiment.py -c configs/etth1_gpt2.yaml

Modify any YAML in `configs/` to change dataset, LLM name, learning rate, epochs, etc.

## 5. Directory Layout  

AutoTimes_project/
├── autotimes/          # model, data loader, trainer, utils
├── configs/            # YAML configs (etth1, lora, weather)
├── scripts/run_all.sh  # one‑click reproducibility script
├── src/                # experiment entry, result aggregation, plotting
├── data/readme_data.txt# download instruction
├── requirements.txt
└── checkpoints/        # saved weights + result.txt (auto‑generated)


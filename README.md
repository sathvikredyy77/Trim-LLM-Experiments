# TrimLLM: Progressive Layer Dropping for Domain-Specific LLMs

## Overview
This repository contains a full replication and ablation study of the **TrimLLM** methodology (Progressive Layer Dropping) applied to two distinct Large Language Models. 

The project accurately implements **Algorithm 1** from the original research, utilizing a two-step target selection metric (Calibration Scanning with Frobenius Norm tie-breaking) and $r=1/4$ sparse update regularization to systematically compress models with minimal domain-specific performance degradation.

## Repository Structure

The experiments are split into two independent tracks:

* **`/qwen`**: Contains an ablation study on **Qwen2.5-1.5B** fine-tuned on the **INABS** dataset for summarization. Evaluates Full Fine-Tuning versus PEFT (LoRA) on both Baseline (100%) and Trimmed (70%) architectures. Measured using 9 ROUGE variants.
* **`/llama`**: Contains a progressive depth-scaling study on **LLaMA-7B** fine-tuned on the **SciQ** dataset for Question-Answering. Evaluates sub-layer granularity dropping (MHA vs. MLP) and records Pareto frontier checkpoints at 30%, 40%, and 50% compression ratios. Measured using Exact-Match Accuracy and ROUGE.

---

## Part 1: Qwen 2.5-1.5B (INABS Dataset)
**Methodology:** Compares Full Fine-Tuning against PEFT (LoRA) on base and trimmed architectures.

| Model Variant | FT Method | R-1 (F1) | R-2 (F1) | R-L (F1) | 
| :--- | :--- | :---: | :---: | :---: | 
| Baseline (100%) | Full FT | *TBD* | *TBD* | *TBD* | 
| Baseline (100%) | LoRA | *TBD* | *TBD* | *TBD* | 
| Trimmed (30%) | Full FT | *TBD* | *TBD* | *TBD* | 
| Trimmed (30%) | LoRA | *TBD* | *TBD* | *TBD* | 

---

## Part 2: LLaMA-7B (SciQ Dataset)
**Methodology:** Implements continuous progressive sub-layer dropping (Algorithm 1).

| Compression Ratio | Exact Match Accuracy | R-1 (F1) | R-L (F1) | 
| :--- | :---: | :---: | :---: |
| Baseline (0% dropped) | *TBD* | *TBD* | *TBD* |
| Trimmed (30% dropped)| *TBD* | *TBD* | *TBD* |
| Trimmed (40% dropped)| *TBD* | *TBD* | *TBD* |
| Trimmed (50% dropped)| *TBD* | *TBD* | *TBD* |

---

## Execution Setup

These experiments are designed to run on HPC clusters using the Slurm workload manager. 

**Run Qwen Pipeline:**
```bash
cd qwen
sbatch scripts/run_qwen_pipeline.sh

cd llama
sbatch scripts/run_llama_pipeline.sh

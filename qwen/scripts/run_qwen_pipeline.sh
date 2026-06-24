#!/bin/bash
#SBATCH --job-name=qwen_trim_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/qwen_run_%j.out

module load anaconda3
module load cuda/11.8
source activate your_ml_environment

mkdir -p results
mkdir -p saved_models

# 1. Evaluate Base Model
python src/evaluate_rouge.py --model_path "Qwen/Qwen2.5-1.5B" --run_name "qwen_base"

# 2. Full Fine-Tuning
python src/train_full.py
python src/evaluate_rouge.py --model_path "./saved_models/qwen_full_ft_complete" --run_name "qwen_full_ft"

# 3. Trimmed Fine-Tuning (30%)
python src/train_trim.py --trim_ratio 0.3
python src/evaluate_rouge.py --model_path "./saved_models/qwen_trim_0.3_complete" --run_name "qwen_trim_30"

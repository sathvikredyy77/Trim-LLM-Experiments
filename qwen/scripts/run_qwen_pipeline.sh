#!/bin/bash
#SBATCH --job-name=qwen_ablation_grid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00          # Increased time since we are running 4 training loops
#SBATCH --mem=64G
#SBATCH --output=logs/qwen_ablation_%j.out

module load anaconda3
module load cuda/11.8
source activate your_ml_environment

mkdir -p results
mkdir -p saved_models

echo "=== 1. BASE MODEL EVALUATION ==="
python src/evaluate_rouge.py --model_path "Qwen/Qwen2.5-1.5B" --run_name "qwen_base"

echo "=== 2. FULL FINE-TUNING (NO LORA) ==="
python src/train_full.py
python src/evaluate_rouge.py --model_path "./saved_models/qwen_full_ft_complete" --run_name "qwen_full_ft"

echo "=== 3. TRIMMED FULL FINE-TUNING (NO LORA) ==="
python src/train_trim.py --trim_ratio 0.3
python src/evaluate_rouge.py --model_path "./saved_models/qwen_trim_0.3_complete" --run_name "qwen_trim_30_full"

echo "=== 4. BASE MODEL WITH LORA ==="
python src/train_lora.py
python src/evaluate_rouge.py --model_path "./saved_models/qwen_lora_complete" --run_name "qwen_lora"

echo "=== 5. TRIMMED MODEL WITH LORA ==="
python src/train_trim_lora.py --trim_ratio 0.3
python src/evaluate_rouge.py --model_path "./saved_models/qwen_trim_0.3_lora_complete" --run_name "qwen_trim_30_lora"

echo "=== ALL EXPERIMENTS COMPLETE ==="

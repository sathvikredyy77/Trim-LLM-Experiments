#!/bin/bash
#SBATCH --job-name=llama_paper_replication
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1             
#SBATCH --time=48:00:00          
#SBATCH --mem=80G                
#SBATCH --output=logs/llama_run_%j.out
#SBATCH --error=logs/llama_run_%j.err

module load anaconda3
module load cuda/11.8
source activate your_ml_environment

mkdir -p results
mkdir -p saved_models
mkdir -p logs

MODEL_ID="meta-llama/Llama-2-7b-hf"

echo "=== STAGE 1: EVALUATING BASE LLaMA-7B ==="
python src/evaluate_sciq.py --model_path $MODEL_ID --run_name "llama_base"

echo "=== STAGE 2: RUNNING ALGORITHM 1 PROGRESSIVE TRIMMING ==="
python src/train_trim_progressive.py --model_id $MODEL_ID

echo "=== STAGE 3: EVALUATING COMPRESSED MODELS ==="
python src/evaluate_sciq.py --model_path "./saved_models/llama_trim_30percent" --run_name "llama_trim_30"
python src/evaluate_sciq.py --model_path "./saved_models/llama_trim_40percent" --run_name "llama_trim_40"
python src/evaluate_sciq.py --model_path "./saved_models/llama_trim_50percent" --run_name "llama_trim_50"

echo "=== PIPELINE COMPLETE ==="

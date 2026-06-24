import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from trim_utils import prepare_model_for_sublayer_dropping, apply_sparse_update, drop_one_least_important_sublayer

def format_sciq_prompt(example):
    return {"text": f"Context: {example['support']}\nQuestion: {example['question']}\nAnswer: {example['correct_answer']}"}

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map="auto")
    model = prepare_model_for_sublayer_dropping(model)
    
    dataset = load_dataset("sciq")
    formatted_dataset = dataset.map(format_sciq_prompt)
    calibration_data = dataset["train"].select(range(20)) # Calibration dataset size
    
    # 1. Apply Sparse Update Regularization (r=1/4) [cite: 264]
    model = apply_sparse_update(model, calibration_data, tokenizer, r=0.25)
    
    total_sublayers = 64
    drop_30 = int(total_sublayers * 0.3)
    drop_40 = int(total_sublayers * 0.4)
    drop_50 = int(total_sublayers * 0.5)
    
    # 2. Iterative Fine-Tuning and Dropping 
    for drop_round in range(1, drop_50 + 1):
        print(f"\n=== Algorithm 1: Iteration {drop_round}/{drop_50} ===")
        
        training_args = TrainingArguments(
            output_dir="./results/llama_progressive_logs",
            max_steps=50, # Approximating 1 epoch of rapid specialization [cite: 73]
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            save_strategy="no",
            fp16=True
        )
        trainer = SFTTrainer(
            model=model, train_dataset=formatted_dataset["train"],
            dataset_text_field="text", max_seq_length=512, tokenizer=tokenizer, args=training_args
        )
        trainer.train()
        
        model = drop_one_least_important_sublayer(model, calibration_data, tokenizer)

        if drop_round in [drop_30, drop_40, drop_50]:
            ratio = {drop_30: "30", drop_40: "40", drop_50: "50"}[drop_round]
            save_path = f"./saved_models/llama_trim_{ratio}percent"
            print(f"\n*** Milestone Reached! Saving {ratio}% Trimmed Model ***")
            torch.save(model.state_dict(), f"{save_path}/model_weights.pt")
            tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf")
    main(parser.parse_args())

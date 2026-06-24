import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

def format_sciq_prompt(example):
    """Formats the SciQ multiple-choice dataset for standard Causal LM training."""
    return {"text": f"Context: {example['support']}\nQuestion: {example['question']}\nAnswer: {example['correct_answer']}"}

def main(args):
    print(f"Loading Base Model for Full Fine-Tuning: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    print("Loading SciQ Dataset...")
    dataset = load_dataset("sciq")
    formatted_dataset = dataset.map(format_sciq_prompt)
    
    print("Starting Full Fine-Tuning (Baseline)...")
    training_args = TrainingArguments(
        output_dir="./results/llama_full_ft_checkpoints",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=5, # Updated to 5 epochs to exactly match the paper's baseline
        save_strategy="no",
        fp16=True,
        logging_steps=10
    )
    
    trainer = SFTTrainer(
        model=model, 
        train_dataset=formatted_dataset["train"],
        dataset_text_field="text", 
        max_seq_length=512, 
        tokenizer=tokenizer, 
        args=training_args
    )
    
    trainer.train()
    
    save_path = "./saved_models/llama_full_ft_complete"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Full Fine-Tuned Model successfully saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf")
    main(parser.parse_args())

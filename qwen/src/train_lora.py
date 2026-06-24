import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

def format_inabs_prompt(example):
    return {"text": f"Document: {example['document']}\nSummary: {example['summary']}"}

def main(args):
    print(f"Loading Base Model: {args.model_id} for LoRA Fine-Tuning...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto")
    
    # 1. Apply LoRA Adapters
    print("Injecting LoRA adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("Loading INABS Dataset...")
    dataset = load_dataset("inabs")
    formatted_dataset = dataset.map(format_inabs_prompt)
    
    training_args = TrainingArguments(
        output_dir="./results/qwen_lora_checkpoints",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4, # Notice: LoRA uses a higher learning rate than Full FT
        num_train_epochs=3,
        save_strategy="no",
        fp16=True
    )
    
    print("Starting LoRA Fine-Tuning...")
    trainer = SFTTrainer(
        model=model, 
        train_dataset=formatted_dataset["train"],
        dataset_text_field="text", 
        max_seq_length=512, 
        tokenizer=tokenizer, 
        args=training_args
    )
    trainer.train()
    
    save_path = "./saved_models/qwen_lora_complete"
    trainer.save_model(save_path)
    print(f"LoRA Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B")
    main(parser.parse_args())

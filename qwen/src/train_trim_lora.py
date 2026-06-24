import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
from trim_utils import drop_one_least_important_layer

def format_inabs_prompt(example):
    return {"text": f"Document: {example['document']}\nSummary: {example['summary']}"}

def main(args):
    print(f"Loading Base Model: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto")
    
    print("Loading INABS Dataset...")
    dataset = load_dataset("inabs")
    formatted_dataset = dataset.map(format_inabs_prompt)
    calibration_data = dataset["train"].select(range(10))
    
    # Calculate layers to drop
    total_initial_layers = len(model.model.layers)
    target_layers = int(total_initial_layers * (1.0 - args.trim_ratio))
    layers_to_drop = total_initial_layers - target_layers
    
    # 1. Progressive Layer Dropping (Without LoRA to evaluate full structural integrity)
    for drop_round in range(layers_to_drop):
        print(f"\n=== Progressive Drop Round {drop_round + 1}/{layers_to_drop} ===")
        model = drop_one_least_important_layer(model, calibration_data, tokenizer)

    # 2. Inject LoRA adapters into the newly compressed model
    print("\nModel trimmed! Injecting LoRA adapters for final training phase...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    # Re-enable gradient checkpointing/training for the PEFT wrap
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Final LoRA Training
    training_args = TrainingArguments(
        output_dir=f"./results/qwen_trim_{args.trim_ratio}_lora_checkpoints",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        save_strategy="no",
        fp16=True
    )
    
    trainer = SFTTrainer(
        model=model, train_dataset=formatted_dataset["train"],
        dataset_text_field="text", max_seq_length=512, tokenizer=tokenizer, args=training_args
    )
    trainer.train()
    
    save_path = f"./saved_models/qwen_trim_{args.trim_ratio}_lora_complete"
    trainer.save_model(save_path)
    print(f"\nSuccessfully generated and saved Trimmed LoRA model to {save_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--trim_ratio", type=float, default=0.3)
    main(parser.parse_args())

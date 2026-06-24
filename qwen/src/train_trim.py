import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
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
    
    # Separate a small calibration dataset (e.g., 10 samples for speed)
    calibration_data = dataset["train"].select(range(10))
    
    # Calculate exactly how many layers we need to drop total
    total_initial_layers = len(model.model.layers)
    target_layers = int(total_initial_layers * (1.0 - args.trim_ratio))
    layers_to_drop = total_initial_layers - target_layers
    
    print(f"Initial Layers: {total_initial_layers} | Target Layers: {target_layers} | Total to drop: {layers_to_drop}")
    
    # Progressive Layer Dropping Loop
    for drop_round in range(layers_to_drop):
        print(f"\n=== Progressive Drop Round {drop_round + 1}/{layers_to_drop} ===")
        
        # 1. Train for a mini-epoch (simulated via short steps or 1 epoch as per paper)
        training_args = TrainingArguments(
            output_dir=f"./results/qwen_progressive_checkpoints",
            max_steps=50, # Fast fine-tuning iteration before structural evaluation
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            save_strategy="no",
            fp16=True,
            logging_steps=10
        )
        
        trainer = SFTTrainer(
            model=model, train_dataset=formatted_dataset["train"],
            dataset_text_field="text", max_seq_length=512, tokenizer=tokenizer, args=training_args
        )
        trainer.train()
        
        # 2. Dynamically scan and drop the layer that hurts performance the least
        model = drop_one_least_important_layer(model, calibration_data, tokenizer)

    # Save final optimized compact model
    save_path = f"./saved_models/qwen_trim_{args.trim_ratio}_complete"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nSuccessfully generated and saved specialized compact model to {save_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--trim_ratio", type=float, default=0.3)
    main(parser.parse_args())

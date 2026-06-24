import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

def format_inabs_prompt(example):
    return {"text": f"Document: {example['document']}\nSummary: {example['summary']}"}

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto")
    
    dataset = load_dataset("inabs")
    formatted_dataset = dataset.map(format_inabs_prompt)
    
    training_args = TrainingArguments(
        output_dir="./results/qwen_full_ft_checkpoints",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        save_strategy="no",
        fp16=True
    )
    
    trainer = SFTTrainer(
        model=model, train_dataset=formatted_dataset["train"],
        dataset_text_field="text", max_seq_length=512, tokenizer=tokenizer, args=training_args
    )
    trainer.train()
    trainer.save_model("./saved_models/qwen_full_ft_complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B")
    main(parser.parse_args())

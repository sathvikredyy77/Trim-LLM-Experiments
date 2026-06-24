import argparse
import torch
import evaluate
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    
    dataset = load_dataset("inabs")
    eval_subset = dataset["test"].select(range(min(100, len(dataset["test"]))))
    
    rouge = evaluate.load("rouge")
    predictions, references = [], []
    
    model.eval()
    with torch.no_grad():
        for item in eval_subset:
            prompt = f"Document: {item['document']}\nSummary: "
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)
            pred_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            
            predictions.append(pred_text)
            references.append(item["summary"])
            
    results = rouge.compute(predictions=predictions, references=references, use_aggregator=True)
    
    output_filename = f"./results/rouge_scores_{args.run_name}.json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"9 ROUGE Metrics calculated and saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    main(parser.parse_args())

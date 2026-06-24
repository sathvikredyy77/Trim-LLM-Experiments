import argparse
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    
    dataset = load_dataset("sciq")
    eval_subset = dataset["test"].select(range(min(200, len(dataset["test"]))))
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    exact_matches = 0
    rouge_totals = {"r1_p": 0, "r1_r": 0, "r1_f": 0, "r2_p": 0, "r2_r": 0, "r2_f": 0, "rl_p": 0, "rl_r": 0, "rl_f": 0}
    
    model.eval()
    with torch.no_grad():
        for item in eval_subset:
            prompt = f"Context: {item['support']}\nQuestion: {item['question']}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            outputs = model.generate(**inputs, max_new_tokens=15, pad_token_id=tokenizer.eos_token_id)
            pred = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip().lower()
            ref = item["correct_answer"].strip().lower()
            
            if ref in pred: exact_matches += 1
                
            scores = scorer.score(ref, pred)
            rouge_totals["r1_p"] += scores['rouge1'].precision
            rouge_totals["r1_r"] += scores['rouge1'].recall
            rouge_totals["r1_f"] += scores['rouge1'].fmeasure
            rouge_totals["r2_p"] += scores['rouge2'].precision
            rouge_totals["r2_r"] += scores['rouge2'].recall
            rouge_totals["r2_f"] += scores['rouge2'].fmeasure
            rouge_totals["rl_p"] += scores['rougeL'].precision
            rouge_totals["rl_r"] += scores['rougeL'].recall
            rouge_totals["rl_f"] += scores['rougeL'].fmeasure

    n = len(eval_subset)
    final_results = {k: v / n for k, v in rouge_totals.items()}
    final_results["Exact_Match_Accuracy"] = exact_matches / n
    
    print(f"\n=== Results for {args.run_name} ===")
    print(f"Paper Accuracy: {final_results['Exact_Match_Accuracy'] * 100:.2f}%")
    for k, v in final_results.items():
        if k != "Exact_Match_Accuracy": print(f"{k.upper()}: {v:.4f}")
        
    with open(f"./results/eval_{args.run_name}.json", "w") as f:
        json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    main(parser.parse_args())

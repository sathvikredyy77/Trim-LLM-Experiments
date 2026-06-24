import torch
import torch.nn as nn
import copy

class PatchableLlamaLayer(nn.Module):
    """Wraps LLaMA layers to allow independent dropping of MHA or MLP."""
    def __init__(self, original_layer, layer_idx):
        super().__init__()
        self.original_layer = original_layer
        self.layer_idx = layer_idx
        self.mha_dropped = False
        self.mlp_dropped = False

    def forward(self, hidden_states, *args, **kwargs):
        residual = hidden_states
        hidden_states = self.original_layer.input_layernorm(hidden_states)
        if not self.mha_dropped:
            hidden_states, _, _ = self.original_layer.self_attn(hidden_states, *args, **kwargs)
            hidden_states = residual + hidden_states
        else:
            hidden_states = residual

        residual = hidden_states
        hidden_states = self.original_layer.post_attention_layernorm(hidden_states)
        if not self.mlp_dropped:
            hidden_states = self.original_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
        else:
            hidden_states = residual
        return (hidden_states,)

def prepare_model_for_sublayer_dropping(model):
    for i, layer in enumerate(model.model.layers):
        if not isinstance(layer, PatchableLlamaLayer):
            model.model.layers[i] = PatchableLlamaLayer(layer, i)
    return model

def calculate_accuracy(model, calibration_data, tokenizer):
    """Calculates a_i (accuracy) for the calibration formula[cite: 158]."""
    correct = 0
    model.eval()
    with torch.no_grad():
        for item in calibration_data:
            prompt = f"Context: {item['support']}\nQuestion: {item['question']}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
            pred = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip().lower()
            if item['correct_answer'].strip().lower() in pred:
                correct += 1
    return (correct / len(calibration_data)) * 100.0

def scan_sublayer_importance(model, sublayer_type, layer_idx, calibration_data, tokenizer, delta=1e-5):
    """Implements Sensitivity-based Scoring formula from the paper[cite: 159]."""
    target_layer = model.model.layers[layer_idx]
    originally_dropped = target_layer.mha_dropped if sublayer_type == "mha" else target_layer.mlp_dropped
    
    # Temporarily drop
    if sublayer_type == "mha": target_layer.mha_dropped = True
    else: target_layer.mlp_dropped = True
    
    # Calculate a_i
    a_i = calculate_accuracy(model, calibration_data, tokenizer)
    
    # Formula: s_{i,scan} = (100 - a_i) / ((1 + delta^2) + (1 + delta) * a_i) [cite: 159]
    s_i_scan = (100.0 - a_i) / ((1 + delta**2) + (1 + delta) * a_i)
    
    # Restore
    if sublayer_type == "mha": target_layer.mha_dropped = originally_dropped
    else: target_layer.mlp_dropped = originally_dropped
    
    return s_i_scan

def compute_frobenius_norm(model, sublayer_type, layer_idx):
    """Implements Activation-based Scoring using Frobenius norm[cite: 178]."""
    layer = model.model.layers[layer_idx].original_layer
    target_module = layer.self_attn.q_proj if sublayer_type == "mha" else layer.mlp.gate_proj
    weight_matrix = target_module.weight.data.float()
    f_norm = torch.norm(weight_matrix, p="fro").item()
    return f_norm

def apply_sparse_update(model, calibration_data, tokenizer, r=0.25):
    """Freezes 75% of the network based on initial calibration scores[cite: 250, 264]."""
    print(f"Applying Sparse Update (r={r}). Scanning initial importance...")
    scores = []
    num_blocks = len(model.model.layers)
    
    for i in range(num_blocks):
        scores.append((scan_sublayer_importance(model, "mha", i, calibration_data, tokenizer), "mha", i))
        scores.append((scan_sublayer_importance(model, "mlp", i, calibration_data, tokenizer), "mlp", i))
        
    # Sort descending (highest score = most important)
    scores.sort(key=lambda x: x[0], reverse=True)
    keep_trainable_count = int(len(scores) * r)
    trainable_targets = [(x[1], x[2]) for x in scores[:keep_trainable_count]]
    
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze only the top r%
    for sub_type, idx in trainable_targets:
        layer = model.model.layers[idx].original_layer
        target_module = layer.self_attn if sub_type == "mha" else layer.mlp
        for param in target_module.parameters():
            param.requires_grad = True
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Sparse Update Complete: {trainable_params} parameters remain trainable.")
    return model

def drop_one_least_important_sublayer(model, calibration_data, tokenizer):
    """Algorithm 1: Two-step Target Selection[cite: 97, 282]."""
    num_blocks = len(model.model.layers)
    candidates = []

    for i in range(num_blocks):
        wrapper = model.model.layers[i]
        if not wrapper.mha_dropped:
            score = scan_sublayer_importance(model, "mha", i, calibration_data, tokenizer)
            candidates.append(("mha", i, score))
        if not wrapper.mlp_dropped:
            score = scan_sublayer_importance(model, "mlp", i, calibration_data, tokenizer)
            candidates.append(("mlp", i, score))
            
    # Find the minimum score (least important) [cite: 153]
    min_score = min([c[2] for c in candidates])
    ties = [c for c in candidates if abs(c[2] - min_score) < 1e-4]
    
    if len(ties) == 1:
        best_sublayer_type, best_layer_idx, _ = ties[0]
    else:
        # Tie-breaker: Maximize Frobenius Norm [cite: 175, 178]
        highest_norm = -1
        for sub_type, idx, _ in ties:
            norm = compute_frobenius_norm(model, sub_type, idx)
            if norm > highest_norm:
                highest_norm = norm
                best_sublayer_type = sub_type
                best_layer_idx = idx

    if best_sublayer_type == "mha":
        model.model.layers[best_layer_idx].mha_dropped = True
    else:
        model.model.layers[best_layer_idx].mlp_dropped = True
        
    print(f"--> Permanently Dropped: Block {best_layer_idx} [{best_sublayer_type.upper()}]")
    return model

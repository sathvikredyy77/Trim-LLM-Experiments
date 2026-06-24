import torch
import torch.nn as nn
import copy

def evaluate_temporary_drop(model, layer_idx, calibration_data, tokenizer):
    """
    Temporarily bypasses a layer to calculate a quick loss on calibration data.
    A lower loss means the model performs well even without this layer.
    """
    original_layers = model.model.layers
    
    # Create a new list bypassing the target layer index
    bypassed_layers = nn.ModuleList([layer for i, layer in enumerate(original_layers) if i != layer_idx])
    
    # Swap layers temporarily
    model.model.layers = bypassed_layers
    model.config.num_hidden_layers = len(bypassed_layers)
    
    total_loss = 0.0
    model.eval()
    
    # Calculate loss on a small batch of calibration data
    with torch.no_grad():
        for item in calibration_data:
            text = f"Document: {item['document']}\nSummary: {item['summary']}"
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(model.device)
            inputs["labels"] = inputs["input_ids"].clone()
            
            outputs = model(**inputs)
            total_loss += outputs.loss.item()
            
    # Restore original layers
    model.model.layers = original_layers
    model.config.num_hidden_layers = len(original_layers)
    
    # Return average loss (lower is better, meaning layer is less critical)
    return total_loss / len(calibration_data)

def drop_one_least_important_layer(model, calibration_data, tokenizer):
    """
    Scans all remaining layers, finds the one whose removal causes the 
    least damage (lowest validation loss), and permanently drops it.
    """
    original_layers = model.model.layers
    num_layers = len(original_layers)
    
    best_layer_to_drop = -1
    lowest_loss = float('inf')
    
    print(f"Scanning {num_layers} layers to identify the least important one...")
    
    # Test dropping each layer individually
    for i in range(num_layers):
        loss = evaluate_temporary_drop(model, i, calibration_data, tokenizer)
        if loss < lowest_loss:
            lowest_loss = loss
            best_layer_to_drop = i
            
    print(f"--> Strategy: Permanently dropping Layer Index {best_layer_to_drop} (Calibration Loss: {lowest_loss:.4f})")
    
    # Permanently drop the layer
    new_layers = nn.ModuleList([layer for i, layer in enumerate(original_layers) if i != best_layer_to_drop])
    model.model.layers = new_layers
    model.config.num_hidden_layers = len(new_layers)
    
    return model

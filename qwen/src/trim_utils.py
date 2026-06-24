import torch
import torch.nn as nn

def trim_model_layers(model, compression_ratio):
    original_layers = model.model.layers
    num_layers = len(original_layers)
    keep_ratio = 1.0 - compression_ratio
    num_to_keep = int(num_layers * keep_ratio)
    
    print(f"Original layers: {num_layers} | Compressing by {compression_ratio*100}% | Keeping: {num_to_keep} layers")

    indices_to_keep = torch.linspace(0, num_layers - 1, num_to_keep).round().int().tolist()
    new_layers = nn.ModuleList([original_layers[i] for i in indices_to_keep])
    
    model.model.layers = new_layers
    model.config.num_hidden_layers = num_to_keep
    return model

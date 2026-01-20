"""Neuron intervention hooks."""

from typing import Dict, List

import torch.nn.functional as F


def apply_gate_proj_hooks(model, flat_index_dict: Dict[int, List[str]], intermediate_size: int = 11008, coef: float = 1.0):
    values_per_layer = {}
    for flat_idx in flat_index_dict:
        layer_idx = flat_idx // intermediate_size
        neuron_idx = flat_idx % intermediate_size
        values_per_layer.setdefault(layer_idx, []).append(neuron_idx)

    print("\nNeurons selected for activation:\n")
    for layer, neurons in values_per_layer.items():
        print(f"  Layer {layer}: Neurons {neurons}")
    print()

    def down_proj_hook(neuron_ids, coef_val):
        def hook_fn(module, input, output):
            # Input: activations of shape (batch, seq_len, num_value_vectors) (where for openvla num_value_vectors = 11008)
            # Output: FFN output of shape (batch, seq_len, embedding_dim) (where for openvla embedding_dim = 4096)
            # print(f"⚡️ Hook triggered: setting {len(neuron_ids)} neurons to {coef_val}")
            modified_input = input[0]
            modified_input[..., neuron_ids] = coef_val
            output = F.linear(modified_input, module.weight, module.bias)
            return output
        return hook_fn

    hooks = []
    decoder_layers = model.language_model.model.layers
    for layer_idx, neuron_ids in values_per_layer.items():
        layer = decoder_layers[layer_idx]
        hook = layer.mlp.down_proj.register_forward_hook(
            down_proj_hook(neuron_ids, coef)
        )
        hooks.append(hook)

    return hooks

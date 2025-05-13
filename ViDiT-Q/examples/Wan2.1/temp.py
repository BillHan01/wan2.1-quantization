import torch

quant_params = torch.load("/cv/hanjiarui/code/ViDiT-Q/examples/Wan2.1/quant_data/checkpoint/quant_params.pth")

print("Top-level keys:", list(quant_params.keys()))

for i, (layer_name, params) in enumerate(quant_params.items()):
    print(f"\nLayer {i+1}: {layer_name}")
    for param_name, value in params.items():
        if value is None:
            print(f"  {param_name}: None")
        elif isinstance(value, torch.Tensor):
            print(f"  {param_name}: shape {value.shape} | dtype {value.dtype}")
        else:
            print(f"  {param_name}: type {type(value)} | value: {value}")
    if i >= 4:
        break

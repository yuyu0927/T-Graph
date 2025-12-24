import torch
from fvcore.nn import FlopCountAnalysis, parameter_count


def move_to_device(data, device):
    """Recursively move tensors contained in nested structures to the target device."""
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    else:
        # For non-tensor types (bool, int, str, etc.), return as-is.
        return data


def measure_model_profile(model, inputs, device="cuda"):
    """
    Measure model FLOPs, parameter count, and peak GPU memory usage.

    Notes:
        - `inputs` should be a tuple that matches the model's forward signature.
        - FLOPs are computed by fvcore's FlopCountAnalysis (operator coverage may vary).
        - Peak memory is measured via torch.cuda.max_memory_allocated().

    Args:
        model (torch.nn.Module): Model to profile.
        inputs (Tuple[Any, ...]): Model inputs (must be a tuple).
        device (str): Target device (e.g., "cuda", "cuda:0").

    Returns:
        flops_g (float): Total FLOPs in GFLOPs.
        params_m (float): Total parameters in millions (M).
        memory_mb (float): Peak allocated GPU memory in MB.
    """
    model.eval()
    model = model.to(device)

    # If your inputs are not already on GPU, you can enable this:
    # inputs = move_to_device(inputs, device)

    with torch.cuda.device(device):
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            # FLOPs & Params
            analysis = FlopCountAnalysis(model, inputs)
            flops = analysis.total()  # raw FLOPs
            params = sum(p.numel() for p in model.parameters())

            # Optional: print FLOPs for specific submodules
            flops_by_module = analysis.by_module()
            total_flops_t = 0.0
            for k, v in flops_by_module.items():
                if "translation_regressor_graph" in k:  # e.g., translation_regressor_graph / translation_regressor
                    total_flops_t += v
                    print(f"{k}: {v / 1e9:.4f} GFLOPs")
            print(f"[INFO] GFLOPs for T-Graph: {total_flops_t / 1e9:.4f}")

            # Peak GPU memory (MB): run a forward pass to trigger allocations
            _ = model(*inputs)
            max_memory = torch.cuda.max_memory_allocated(device) / (1024**2)

    return flops / 1e9, params / 1e6, max_memory  # GFLOPs, M, MB

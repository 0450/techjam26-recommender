import gc
import json
import numpy as np
import torch

class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_json(data: dict, filepath: str) -> None:
    """Safely serialize dictionary data to JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, cls=NpEncoder)

def clear_memory() -> None:
    """Garbage collects and clears PyTorch CUDA memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
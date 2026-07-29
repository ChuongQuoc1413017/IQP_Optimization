import os
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_HERE), "data")

def _load(name):
    path = os.path.join(DATA_DIR, name + ".json")

    with open(path, "r") as f:
        raw = json.load(f)
    
    return {int(k): v for k, v in raw.items()}

dataset = {'ising': _load('ising'), 'maxcut': _load('maxcut'), 'partition': _load('partition')}

exact_ising = _load('ising_loss_exact') # Exact ground truth for ising problem

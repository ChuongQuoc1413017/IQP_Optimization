import os, json, argparse
from iqp.parallel import grid
from iqp.parallel.run import ckpt_path  

def merge(experiment):
    merged, missing = {}, []
    for init in grid.INITS:
        merged[init] = {}
        for ham in grid.HAMS:
            merged[init][ham] = {}
            for ansatz, spec in grid.ANSATZ_CONFIGS:
                label = grid.config_label(ansatz, spec)
                merged[init][ham][label] = {}
                for n in grid.QUBITS:
                    p = ckpt_path(experiment, init, ham, label, n)
                    if not os.path.exists(p):
                        missing.append((init, ham, label, n)); continue
                    with open(p) as f:
                        merged[init][ham][label][str(n)] = json.load(f)
    return merged, missing

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=["ratio", "gd"])
    args = ap.parse_args()
    merged, missing = merge(args.experiment)
    if missing:
        print(f"Miss {len(missing)}/375 config, Eg: {missing[:8]}")
    out = os.path.join(os.path.dirname(__file__), "..", "results", f"{args.experiment}_merged.json")
    with open(out, "w") as f:
        json.dump(merged, f, indent=4)
    print("Output", out, "| missing:", len(missing))

if __name__ == "__main__":
    main()

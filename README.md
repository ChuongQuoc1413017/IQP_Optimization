# Discovery of connectivity-trainability trade-off of IQP Circuits for Hamiltonian Optimization

---

## Install and run

```bash
git clone https://github.com/ChuongQuoc1413017/IQP_Optimization.git
cd IQP_Optimization                 # repo root -- `import iqp` resolves from here
pip install -r iqp/requirements.txt
```

Python 3.10-3.12; `numpy==1.26.4` has no wheel for 3.13. Checked on Windows and
Linux with Python 3.12.

---

## Pipeline

Every driver below is checkpointed per config under `iqp/results/`, and `results/`
ships with the checkpoints of the run reported in the paper -- so re-running a
command as-is prints `0 tasks` and exits without recomputing anything. Pass
`--force` to recompute regardless of the checkpoints (on `iqp.parallel.run` that
means all 375 configs; narrow it with `--shard/--nshards`).

### 1. Main grid -- approximation ratio and gradient variance

Grid defined in [`iqp/parallel/grid.py`](iqp/parallel/grid.py):
5 initializations × 3 Hamiltonians × 5 ansätze × 5 sizes = **375 configs**,
50 problem instances each.

```bash
python -m iqp.parallel.run   --experiment ratio --workers 14
python -m iqp.parallel.merge --experiment ratio  # -> results/ratio_merged.json

python -m iqp.parallel.run   --experiment gd    --workers 14
python -m iqp.parallel.merge --experiment gd    # -> results/gd_merged.json
```

### 2. Expressibility and entanglement entropy
```bash
python -m iqp.experiments.expr_ent --workers 14   # -> results/expr_ent_merged.json
```

### 3. Depth
```bash
python -m iqp.parallel.depth_run --metric both --workers 14
```

### 4. Optimizer
```bash
python -m iqp.parallel.optim_run --workers 14
```

---

## Layout

```
iqp/
  core/         ansatz.py     the 5 ansätze + the 5 initializations
                engine.py     Adam training loop, seeding, gradient evaluation
                problems.py   Ising / MaxCut / Number Partition Hamiltonians
                data.py       loads data/*.json
  data/         the 50 problem instances per size, per problem, and the
                brute-forced exact Ising ground energies
  experiments/  one module per experiment; each exposes compute_one(task)
  parallel/     grid.py defines the 375-config grid; run.py / merge.py drive
                the main grid; depth_run.py / optim_run.py drive the two
                Discussion examples
  plots/        one module per figure; style.py holds the shared paper style
  results/      per-config checkpoints, merged JSON, run logs, output PDFs
  tools/        datagen.ipynb  regenerates data/*.json (run-once, offline)
                draw.ipynb     quick circuit drawing via apply_ansatz
```

---

## Citation

```bibtex

```

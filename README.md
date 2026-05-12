# Evaluating Compositional Generalization in Transformers: The Role of Composition Equivalence and Module Coverage

This repository contains the code for the NeurIPS 2026 Evaluation & Datasets submission:
**Evaluating Compositional Generalization in Transformers: The Role of Composition Equivalence and Module Coverage**.

Our implementation builds on the code provided by [Ramesh et al., 2024](https://proceedings.mlr.press/v235/ramesh24a.html): [https://github.com/rahul13ramesh/compositional_capabilities/](https://github.com/rahul13ramesh/compositional_capabilities/).

---

## Repository Structure

```
.
├── bash_scripts/          # End-to-end pipeline scripts
├── config/                # YAML configs for data generation, training, and evaluation
│   ├── gen/               # Data generation config
│   ├── train/             # NanoGPT training config
│   ├── eval/              # Evaluation config
│   └── finetune/          # LoRA finetuning config (LiveCodeBench)
├── scripts/               # Python entry points
│   ├── generate_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── analysis_plots.py
│   ├── finetune_livecodebench_lora.py
│   └── evaluate_livecodebench_equiv_leakage.py
└── src/                   # Library code
    ├── data/              # Data generation, split strategies, equivalence classes
    ├── models/            # NanoGPT and pretrained model wrappers
    ├── training/          # Training and finetuning loops
    └── evaluation/        # Evaluators and plotting utilities
```

---

## Setup

**Requirements:** Python 3.8, CUDA-capable GPU.

```bash
python -m venv compositional_generalization
source compositional_generalization/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes all packages needed for the synthetic benchmark. The last four entries (`transformers`, `peft`, `accelerate`, `datasets`) are only required for the LiveCodeBench LoRA experiments.

---

## Reproducing Paper Results

There are two experimental tracks:

1. **Synthetic benchmark** — NanoGPT trained from scratch on function-composition tasks (Figures 1–4, 10, 12–13, 19).
2. **LiveCodeBench** — LLaMA-3 LoRA finetuning with controlled equivalence-class leakage (Figure 5+).

### Track 1 — Synthetic Benchmark

#### Step 1: Generate Data

```bash
bash bash_scripts/data.sh
```

This generates datasets for all split strategies used in the paper:

| Split strategy | Description | Figure(s) |
|---|---|---|
| `combination_k` (k=2–6), `task_max_length=k` | Within-k evaluation, no identity modules | Figure 1 |
| `combination_k` (k=1–6), `task_max_length=7` | Within-k and cross-k evaluation, with identity modules | Figure 1-2 |
| `disjoint7_6_{0,10,20,50,60,70,100}` | Controlled equivalence-leakage splits (swap test members) | Figure 3 |
| `disjoint9_6_{0,10,20,50,60,70,100}` | Controlled leakage without swapping | Figure 13 |
| `continuouscoverage_6_{0.0–1.0}` | Position-wise module coverage | Figure 4, 19 |
| `continuouspaircoverage_6_{0.0–1.0}` | Pairwise module coverage | Figure 4, 19 |
| `equivalence_6_576_{0,1,25,49,73,97,121,144}` | Shared equivalence class validation | Figure 10a |
| `uniequivalence_6_576_{0,...}` | Training equivalence class size sweep | Figure 10b |

Each train-test split strategy can be generated for both `uniform` and `diverse` function types.

#### Step 2: Train NanoGPT

```bash
bash bash_scripts/train_model.sh
```

Trains models for different train-test splits:
- **Splits:** `combination_2` through `combination_6`
- **Prompt modes:** `direct`, `step_by_step`
- **Positional embeddings:** `abs`, `rel_global`
- **Function types:** `uniform`, `diverse`
- **Seeds:** 0, 10, 20, 30, 40
- **Architecture:** 6 heads, 3 layers (`nh6_nl3`) or 12 heads, 12 layers (`nh12_nl12`), 100 epochs

Checkpoints are saved under `models/ckpts/`.

#### Step 3: Evaluate

```bash
bash bash_scripts/evaluate_model.sh
```

Runs evaluation for representative configurations (examples from each figure). Adjust
`--train_split`, `--eval_split`, `--function_type`, `--pos_embedding_type`, and `--task_max_length`
to cover the full sweep. Pass `--pretrained True` to evaluate Gemma3-1B instead of NanoGPT.

#### Step 4: Plot Results

```bash
bash bash_scripts/plotting.sh
```

Generates all paper plots. Outputs are written to `results/plot_test/`.

| Script section | Figure |
|---|---|
| Within-k with identity (`combination_identity`) | Figure 2 |
| Within-k without identity (`combination_without_identity`) | Figure 1 |
| Cross-k evaluation | Figure 3 |
| Equivalence validation sweep | Figure 10a |
| Training equivalence size sweep | Figure 10b |

---

### Track 2 — LiveCodeBench (LoRA)

#### Step 1: Generate LiveCodeBench Data

```bash
bash bash_scripts/data_livecodebench.sh
```

Generates five equivalence-leakage splits:

| Split | Shared fraction | Heldout fraction |
|---|---|---|
| `equiv_leakage_shared_0.0` | 0 % | — |
| `equiv_leakage_shared_1.0` | 100 % | — |
| `equiv_leakage_shared_0.66_heldoutfrac_0.0` | 66 % | 0 % |
| `equiv_leakage_shared_0.66_heldoutfrac_0.5` | 66 % | 50 % |
| `equiv_leakage_shared_0.66_heldoutfrac_1.0` | 66 % | 100 % |

Data is saved under `data/livecodebench/`. JSON configs used:
`config/gen/livecodebench-jsons/`.

#### Step 2: Finetune LLaMA-3 with LoRA

```bash
bash bash_scripts/train_lora.sh
```

Runs LoRA finetuning (r=16, α=32) on LLaMA-3 for each split above.
Config: `config/finetune/livecodebench_lora.yaml`.
Checkpoints saved to `models/ckpts/livecodebench_lora/`.

#### Step 3: Evaluate LoRA Models

```bash
bash bash_scripts/evaluate_lora.sh
```

Evaluates each finetuned model on its corresponding held-out test split and on the
`heldout_4_200` cross-split probe set.

---

## Configuration

Key parameters (override via command-line flags or by editing the YAML files):

| Parameter | Default | Description |
|---|---|---|
| `n_functions` | 6 | Max functions in a composition |
| `n_alphabets` | 26 | Token vocabulary size (a–z) |
| `seq_len` | 6 | Input sequence length |
| `task_max_length` | 6 or 7 | Max composition length at test time |
| `epochs` | 100 | Training epochs (NanoGPT) |
| `n_heads_nlayers` | `nh6_nl3` | NanoGPT architecture |
| `pos_embedding_type` | `abs` | `abs` or `rel_global` |
| `prompt_mode` | `direct` | `direct` or `step_by_step` |
| `function_type` | `uniform` | `uniform` or `diverse` |

---
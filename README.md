# IARM: Inductive Algebraic Resonance Memory Language Model

This repository contains the Inductive Algebraic Resonance Memory Language Model (IARM-LM), a non-attention causal language architecture based on learned low-rank query operators and cumulative prefix-normalized memory scans.

IARM does not use causal self-attention: there is no key projection, no query-key score matrix, and no softmax attention matrix. Sequence mixing is performed through local causal convolution, operator-transformed query features, and causal memory accumulation.

## Files

- `iarm_lm_model_only.py` - model-only IARM-LM implementation with internal-structure and activation-trace printers.
- `examples/smoke_test.py` - runnable smoke test and activation-trace example.
- `paper/iarm_architecture_paper.tex` - LaTeX source for the architecture paper.
- `script_bundle/iarm_uploaded_scripts_xz_parts/` - exact compressed payload for the uploaded training scripts.
- `scripts/extract_uploaded_scripts.py` - extractor that restores the uploaded training scripts into `scripts/`.

## Restore the uploaded training scripts

Run this after cloning:

```bash
python scripts/extract_uploaded_scripts.py
```

This reconstructs these files under `scripts/`:

- `scripts/iarm_teacher_memory_transfer_colab_single_cell.py`
- `scripts/iarm_research_finetune_observe_colab_fixed.py`
- `scripts/iarm_lm_full10b_ultrachat200k.py`

## Run the smoke test

```bash
python examples/smoke_test.py
```

## Compile the paper

From the repository root, run:

```bash
pdflatex -interaction=nonstopmode -halt-on-error paper/iarm_architecture_paper.tex
```

The paper intentionally describes IARM as a non-attention causal memory architecture, not as causal self-attention.

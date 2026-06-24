# IARM: Inductive Algebraic Resonance Memory Language Model

This repository contains the Inductive Algebraic Resonance Memory Language Model (IARM-LM), a non-attention causal language architecture based on learned low-rank query operators and cumulative prefix-normalized memory scans.

IARM does not use causal self-attention: there is no key projection, no query-key score matrix, and no softmax attention matrix. Sequence mixing is performed through local causal convolution, operator-transformed query features, and causal memory accumulation.

## Files

- `iarm_lm_model_only.py` - model-only IARM-LM implementation with internal-structure and activation-trace printers.
- `iarm_lm_clutrr_eval_single_cell.py` - single-cell CLUTRR evaluation harness for IARM-LM.
- `paper/iarm_architecture_paper.tex` - LaTeX source for the architecture paper.
- `paper/iarm_architecture_paper.pdf` - compiled architecture paper.

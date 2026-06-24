# IARM: Inductive Algebraic Resonance Memory Language Model

This repository contains the Inductive Algebraic Resonance Memory Language Model (IARM-LM), a non-attention causal language architecture based on learned low-rank query operators and cumulative prefix-normalized memory scans.

IARM does not use causal self-attention: there is no key projection, no query-key score matrix, and no softmax attention matrix. Sequence mixing is performed through local causal convolution, operator-transformed query features, and causal memory accumulation.

## Files

- `iarm_lm_model_only.py` - model-only IARM-LM implementation with internal-structure and activation-trace printers.
- `examples/smoke_test.py` - runnable smoke test and activation-trace example.
- `paper/iarm_architecture_paper.tex` - LaTeX source for the architecture paper.

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

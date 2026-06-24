from __future__ import annotations

import torch

from iarm_lm_model_only import IARMConfig, IARMLM, print_one_forward_internal_trace


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = IARMConfig(
        vocab_size=50257,
        block_size=512,
        dim=768,
        n_head=12,
        n_layer=11,
        ffn_hidden=2048,
        n_operators=4,
        operator_rank=16,
        local_kernel=5,
        dropout=0.05,
    )
    model = IARMLM(config).to(device)
    x = torch.randint(0, config.vocab_size, (2, 64), device=device)
    out = model(x, labels=x)
    print(f"device={device}")
    print(f"input_shape={tuple(x.shape)}")
    print(f"logits_shape={tuple(out['logits'].shape)}")
    print(f"loss={float(out['loss'].detach().cpu()):.6f}")
    print_one_forward_internal_trace(model, x[:1], layer_index=-1)


if __name__ == "__main__":
    main()

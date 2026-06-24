# =============================================================================
# IARM-LM MODEL ONLY
# Inductive Algebraic Resonance Memory Language Model
# =============================================================================

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCH_NAME = "iarm_lm"
ARCH_FULL_NAME = "Inductive Algebraic Resonance Memory Language Model"
UNIT_FULL_NAME = "Inductive Algebraic Resonance Unit"


@dataclass
class IARMConfig:
    vocab_size: int
    block_size: int = 512
    dim: int = 768
    n_head: int = 12
    n_layer: int = 11
    ffn_hidden: int = 2048
    n_operators: int = 4
    operator_rank: int = 16
    local_kernel: int = 5
    dropout: float = 0.05


ARUTGARMConfig = IARMConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[-2]
        return (
            self.cos_cached[:, :, :seq_len, :].to(device=x.device, dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, :].to(device=x.device, dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    return torch.cat((-x[..., d // 2 :], x[..., : d // 2]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.silu(self.gate_proj(x)) * self.up_proj(x)
        y = self.down_proj(y)
        return self.dropout(y)


class InductiveAlgebraicResonanceUnit(nn.Module):
    """IARM sequence mixer.

    This unit is not causal self-attention. It has no key projection, no
    query-key score matrix, and no softmax attention matrix. Sequence mixing is
    performed by local causal convolution, gated low-rank query operators, and a
    cumulative prefix-normalized memory scan.
    """

    def __init__(self, config: IARMConfig):
        super().__init__()
        if config.dim % config.n_head != 0:
            raise ValueError("dim must be divisible by n_head.")

        self.dim = config.dim
        self.n_head = config.n_head
        self.head_dim = config.dim // config.n_head
        self.n_operators = config.n_operators
        self.operator_rank = config.operator_rank
        self.local_kernel = config.local_kernel

        self.norm = RMSNorm(config.dim)
        self.local_conv = nn.Conv1d(
            config.dim,
            config.dim,
            kernel_size=config.local_kernel,
            padding=config.local_kernel - 1,
            groups=config.dim,
            bias=False,
        )
        self.q_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.op_gate_proj = nn.Linear(config.dim, config.n_head * config.n_operators, bias=False)

        self.op_u = nn.Parameter(
            torch.randn(config.n_head, config.n_operators, self.head_dim, config.operator_rank)
            / math.sqrt(self.head_dim)
        )
        self.op_v = nn.Parameter(
            torch.randn(config.n_head, config.n_operators, self.head_dim, config.operator_rank)
            / math.sqrt(self.head_dim)
        )

        self.out_gate = nn.Linear(config.dim, config.dim, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(dim=self.head_dim, max_seq_len=config.block_size)

        self.capture_activations = False
        self.trace_token_limit = 128
        self.trace_coord_limit = 64
        self.last_trace: dict[str, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        h = self.norm(x)

        local = self.local_conv(h.transpose(1, 2))
        local = local[:, :, :t].transpose(1, 2).contiguous()

        q = self.q_proj(h).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(q)
        q = apply_rope(q, cos, sin)

        gate_logits = self.op_gate_proj(h).view(b, t, self.n_head, self.n_operators)
        gate_logits = gate_logits.permute(0, 2, 1, 3).contiguous()
        gate = F.softmax(gate_logits, dim=-1)

        coeff = torch.einsum("bhtd,hodr->bhtor", q, self.op_v)
        delta = torch.einsum("bhtor,hodr->bhtod", coeff, self.op_u)
        delta = (gate.unsqueeze(-1) * delta).sum(dim=3)

        q_op = q + delta / math.sqrt(max(1, self.operator_rank))
        phi = F.elu(q_op) + 1.0

        num = torch.cumsum(phi * v, dim=2)
        den = torch.cumsum(phi, dim=2).clamp_min(1e-6)
        memory_read = num / den

        y = memory_read.transpose(1, 2).contiguous().view(b, t, c)
        output_gate = torch.sigmoid(self.out_gate(h))
        y = y + local
        y = self.out_proj(output_gate * y)
        y = self.dropout(y)

        if getattr(self, "capture_activations", False):
            n_tok = min(t, int(getattr(self, "trace_token_limit", 128)))
            n_dim = min(self.head_dim, int(getattr(self, "trace_coord_limit", 64)))
            with torch.no_grad():
                gate_cpu = gate[0, :, :n_tok, :].float().detach().cpu()
                self.last_trace = {
                    "q_norm": q[0, :, :n_tok, :].float().norm(dim=-1).detach().cpu(),
                    "delta_norm": delta[0, :, :n_tok, :].float().norm(dim=-1).detach().cpu(),
                    "qop_norm": q_op[0, :, :n_tok, :].float().norm(dim=-1).detach().cpu(),
                    "gate": gate_cpu,
                    "gate_entropy": -(gate_cpu.clamp_min(1e-8) * gate_cpu.clamp_min(1e-8).log()).sum(dim=-1),
                    "phi": phi[0, :, :n_tok, :n_dim].float().detach().cpu(),
                    "den": den[0, :, :n_tok, :n_dim].float().detach().cpu(),
                    "memory_read": memory_read[0, :, :n_tok, :n_dim].float().detach().cpu(),
                    "local_norm": local[0, :n_tok, :].float().norm(dim=-1).detach().cpu(),
                    "out_gate_mean": output_gate[0, :n_tok, :].float().mean(dim=-1).detach().cpu(),
                }

        return x + y


AlgebraicResonanceUnit = InductiveAlgebraicResonanceUnit


class IARMBlock(nn.Module):
    def __init__(self, config: IARMConfig):
        super().__init__()
        self.aru = InductiveAlgebraicResonanceUnit(config)
        self.ffn_norm = RMSNorm(config.dim)
        self.mlp = SwiGLUMLP(config.dim, config.ffn_hidden, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aru(x)
        x = x + self.mlp(self.ffn_norm(x))
        return x


ARUTGARMBlock = IARMBlock


class IARMLM(nn.Module):
    def __init__(self, config: IARMConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([IARMBlock(config) for _ in range(config.n_layer)])
        self.final_norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        self._scale_residual_outputs()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_outputs(self) -> None:
        for name, module in self.named_modules():
            if name.endswith("out_proj") or name.endswith("down_proj"):
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> dict[str, torch.Tensor]:
        if input_ids.shape[1] > self.config.block_size:
            input_ids = input_ids[:, -self.config.block_size :]
            if labels is not None:
                labels = labels[:, -self.config.block_size :]

        x = self.token_embedding(input_ids)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)

        out = {"logits": logits}
        if return_hidden:
            out["hidden"] = hidden
        if labels is not None:
            out["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return out


ARUTGARMLM = IARMLM


def safe_torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        while key.startswith("module."):
            key = key[len("module.") :]
        while key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        out[key] = value
    return out


def config_from_checkpoint(ckpt: dict[str, Any], fallback_vocab_size: int | None = None) -> IARMConfig:
    raw_cfg = dict(ckpt.get("config", {}))
    valid = {f.name for f in fields(IARMConfig)}
    clean = {k: v for k, v in raw_cfg.items() if k in valid}
    if "vocab_size" not in clean:
        if fallback_vocab_size is None:
            raise ValueError("Checkpoint has no config.vocab_size. Pass fallback_vocab_size for older checkpoints.")
        clean["vocab_size"] = fallback_vocab_size
    return IARMConfig(**clean)


def load_iarm_from_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    fallback_vocab_size: int | None = None,
    strict: bool = True,
) -> IARMLM:
    checkpoint_path = Path(checkpoint_path)
    ckpt = safe_torch_load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"Invalid checkpoint. Expected dict with key 'model': {checkpoint_path}")
    config = config_from_checkpoint(ckpt, fallback_vocab_size=fallback_vocab_size)
    model = IARMLM(config)
    missing, unexpected = model.load_state_dict(clean_state_dict(ckpt["model"]), strict=strict)
    if dtype is not None:
        model = model.to(dtype=dtype)
    model = model.to(device)
    print(f"loaded_checkpoint={checkpoint_path}")
    print(f"checkpoint_stage={ckpt.get('stage', 'n/a')}")
    print(f"checkpoint_step={ckpt.get('step', 'n/a')}")
    print(f"checkpoint_loss={ckpt.get('loss', 'n/a')}")
    if not strict:
        print(f"missing_keys={missing}")
        print(f"unexpected_keys={unexpected}")
    return model


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_model_internal_structure(model: nn.Module, print_full_repr: bool = True) -> None:
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    print("=" * 100)
    print("IARM MODEL INTERNAL STRUCTURE")
    print("=" * 100)
    print("\n[Architecture]")
    print(f"architecture_name={ARCH_NAME}")
    print(f"architecture_full_name={ARCH_FULL_NAME}")
    print(f"unit_full_name={UNIT_FULL_NAME}")
    print("\n[Config]")
    print(json.dumps(asdict(raw.config), indent=2) if hasattr(raw, "config") else "No config attribute found.")
    print("\n[Parameter count]")
    total_params = count_parameters(raw, False)
    trainable_params = count_parameters(raw, True)
    print(f"total_params={total_params:,}")
    print(f"trainable_params={trainable_params:,}")
    print(f"frozen_params={total_params - trainable_params:,}")
    if print_full_repr:
        print("\n[Full PyTorch module repr]")
        print(raw)
    print("\n[Parameters]")
    for name, param in raw.named_parameters():
        print(f"{name:90s} shape={tuple(param.shape)} dtype={param.dtype} requires_grad={param.requires_grad} numel={param.numel():,}")
    print("\n[Buffers]")
    for name, buf in raw.named_buffers():
        print(f"{name:90s} shape={tuple(buf.shape)} dtype={buf.dtype} numel={buf.numel():,}")


@torch.no_grad()
def print_one_forward_internal_trace(model: IARMLM, input_ids: torch.Tensor, layer_index: int = -1) -> None:
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw.eval()
    for block in raw.blocks:
        block.aru.capture_activations = False
        block.aru.last_trace = None
    if layer_index < 0:
        layer_index = len(raw.blocks) + layer_index
    layer_index = max(0, min(layer_index, len(raw.blocks) - 1))
    raw.blocks[layer_index].aru.capture_activations = True
    input_ids = input_ids.to(next(raw.parameters()).device)
    out = raw(input_ids)
    trace = raw.blocks[layer_index].aru.last_trace
    print("=" * 100)
    print(f"FORWARD TRACE | layer={layer_index}")
    print("=" * 100)
    print(f"logits_shape={tuple(out['logits'].shape)}")
    if trace is None:
        print("No trace captured.")
        return
    for key, value in trace.items():
        vf = value.float()
        print(f"{key:20s} shape={tuple(value.shape)} dtype={value.dtype} mean={float(vf.mean()):.6f} std={float(vf.std()):.6f} min={float(vf.min()):.6f} max={float(vf.max()):.6f}")
    raw.blocks[layer_index].aru.capture_activations = False


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = IARMConfig(vocab_size=50257)
    model = IARMLM(config).to(device)
    print_model_internal_structure(model, print_full_repr=True)
    x = torch.randint(0, model.config.vocab_size, (2, 64), device=device)
    y = model(x, labels=x)
    print("\n[Smoke test]")
    print(f"input_shape={tuple(x.shape)}")
    print(f"logits_shape={tuple(y['logits'].shape)}")
    print(f"loss={float(y['loss'].detach().cpu()):.6f}")
    print_one_forward_internal_trace(model, x[:1], layer_index=-1)

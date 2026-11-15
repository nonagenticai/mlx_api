"""XLM-RoBERTa encoder in mlx.nn, loading Hugging Face safetensors directly.

bge-m3 and bge-reranker-v2-m3 are both plain XLM-RoBERTa-large (24 layers, 1024 hidden, 8194
positions). That is ~100 lines of mlx.nn, so we own it instead of depending on a third-party
MLX model zoo: no extra runtime deps, no torch, exact control over masking/positions, and the
parity tests pin it against the PyTorch reference.

Everything in this module that touches ``mx`` must run on the GPU thread (``gpu.run_on_gpu``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_api.errors import Unavailable

_DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


# --------------------------------------------------------------------------- weight resolution


def _hf_cache_root() -> Path:
    if "HF_HUB_CACHE" in os.environ:
        return Path(os.environ["HF_HUB_CACHE"])
    if "HF_HOME" in os.environ:
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_model_files(model_id: str, names: Sequence[str]) -> dict[str, Path]:
    """Find each file of ``model_id`` locally. Never downloads.

    ``model_id`` may be a local directory. Otherwise the HF cache is searched: the ``refs/main``
    snapshot first, then any other snapshot. Files are resolved independently because a cache can
    legitimately hold e.g. ``model.safetensors`` in one revision and ``tokenizer.json`` in another.
    """
    local = Path(model_id).expanduser()
    if local.is_dir():
        candidates = [local]
    else:
        repo = _hf_cache_root() / ("models--" + model_id.replace("/", "--"))
        snapshots = repo / "snapshots"
        if not snapshots.is_dir():
            raise Unavailable(f"model {model_id!r} is not in the local Hugging Face cache")
        candidates = []
        ref = repo / "refs" / "main"
        if ref.is_file():
            main = snapshots / ref.read_text().strip()
            if main.is_dir():
                candidates.append(main)
        candidates += sorted(p for p in snapshots.iterdir() if p.is_dir() and p not in candidates)

    found: dict[str, Path] = {}
    for name in names:
        for directory in candidates:
            path = directory / name
            if path.is_file():  # follows the blob symlink; a dangling link is not a file
                found[name] = path
                break
        else:
            raise Unavailable(f"model {model_id!r}: {name} not found locally")
    return found


# --------------------------------------------------------------------------- network


@dataclass(frozen=True)
class XLMRConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    type_vocab_size: int
    layer_norm_eps: float
    pad_token_id: int
    num_labels: int

    @classmethod
    def from_file(cls, path: Path) -> XLMRConfig:
        raw = json.loads(path.read_text())
        if raw.get("model_type") != "xlm-roberta":
            raise Unavailable(f"unsupported model_type {raw.get('model_type')!r}")
        if raw.get("hidden_act", "gelu") != "gelu":
            raise Unavailable(f"unsupported hidden_act {raw.get('hidden_act')!r}")
        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            intermediate_size=raw["intermediate_size"],
            max_position_embeddings=raw["max_position_embeddings"],
            type_vocab_size=raw.get("type_vocab_size", 1),
            layer_norm_eps=raw.get("layer_norm_eps", 1e-5),
            pad_token_id=raw.get("pad_token_id", 1),
            num_labels=len(raw.get("id2label", {"0": "LABEL_0"})),
        )

    @property
    def max_tokens(self) -> int:
        # XLM-R positions start at pad_token_id + 1, so two slots are unusable.
        return self.max_position_embeddings - self.pad_token_id - 1


class _Embeddings(nn.Module):
    def __init__(self, cfg: XLMRConfig) -> None:
        super().__init__()
        self.pad = cfg.pad_token_id
        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.token_type_embeddings = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def __call__(self, ids: mx.array, mask: mx.array) -> mx.array:
        positions = mx.cumsum(mask, axis=1) * mask + self.pad
        x = (
            self.word_embeddings(ids)
            + self.position_embeddings(positions)
            + self.token_type_embeddings(mx.zeros_like(ids))
        )
        return self.norm(x)


class _Layer(nn.Module):
    def __init__(self, cfg: XLMRConfig) -> None:
        super().__init__()
        h = cfg.hidden_size
        self.heads = cfg.num_attention_heads
        self.query = nn.Linear(h, h)
        self.key = nn.Linear(h, h)
        self.value = nn.Linear(h, h)
        self.attn_out = nn.Linear(h, h)
        self.attn_norm = nn.LayerNorm(h, eps=cfg.layer_norm_eps)
        self.ffn_in = nn.Linear(h, cfg.intermediate_size)
        self.ffn_out = nn.Linear(cfg.intermediate_size, h)
        self.ffn_norm = nn.LayerNorm(h, eps=cfg.layer_norm_eps)

    def __call__(self, x: mx.array, bias: mx.array) -> mx.array:
        b, n, h = x.shape
        hd = h // self.heads

        def split(t: mx.array) -> mx.array:
            return t.reshape(b, n, self.heads, hd).transpose(0, 2, 1, 3)

        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd**-0.5, mask=bias)
        a = a.transpose(0, 2, 1, 3).reshape(b, n, h)
        x = self.attn_norm(x + self.attn_out(a))
        return self.ffn_norm(x + self.ffn_out(nn.gelu(self.ffn_in(x))))


class XLMREncoder(nn.Module):
    def __init__(self, cfg: XLMRConfig) -> None:
        super().__init__()
        self.embeddings = _Embeddings(cfg)
        self.layers = [_Layer(cfg) for _ in range(cfg.num_hidden_layers)]

    def __call__(self, ids: mx.array, mask: mx.array) -> mx.array:
        x = self.embeddings(ids, mask)
        neg = mx.array(mx.finfo(x.dtype).min, dtype=x.dtype)
        bias = mx.where(mask[:, None, None, :].astype(mx.bool_), mx.array(0, x.dtype), neg)
        for layer in self.layers:
            x = layer(x, bias)
        return x


class XLMRClassifier(nn.Module):
    """``XLMRobertaForSequenceClassification``: CLS -> dense -> tanh -> out_proj."""

    def __init__(self, cfg: XLMRConfig) -> None:
        super().__init__()
        self.encoder = XLMREncoder(cfg)
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.num_labels)

    def __call__(self, ids: mx.array, mask: mx.array) -> mx.array:
        cls = self.encoder(ids, mask)[:, 0, :]
        return self.out_proj(mx.tanh(self.dense(cls)))


# HF key fragment -> our module path, applied in order.
_RENAMES = (
    ("embeddings.LayerNorm.", "embeddings.norm."),
    ("encoder.layer.", "layers."),
    (".attention.self.", "."),
    (".attention.output.dense.", ".attn_out."),
    (".attention.output.LayerNorm.", ".attn_norm."),
    (".intermediate.dense.", ".ffn_in."),
    (".output.dense.", ".ffn_out."),
    (".output.LayerNorm.", ".ffn_norm."),
)


def _rename(key: str) -> str:
    for old, new in _RENAMES:
        key = key.replace(old, new)
    return key


def load_weights(model: nn.Module, path: Path, dtype: str, *, classifier: bool) -> None:
    raw = mx.load(str(path))
    if not isinstance(raw, dict):
        raise Unavailable(f"{path} is not a safetensors weight map")
    target = _DTYPES[dtype]
    weights: list[tuple[str, mx.array]] = []
    for key, value in raw.items():
        key = key.removeprefix("roberta.")
        if key.startswith("pooler.") or key.endswith("position_ids"):
            continue  # unused by CLS pooling / a buffer, not a parameter
        if classifier:
            if key.startswith("classifier."):
                key = key.removeprefix("classifier.")
            else:
                key = "encoder." + _rename(key)
        else:
            key = _rename(key)
        weights.append((key, value.astype(target)))
    model.load_weights(weights, strict=True)
    model.eval()
    mx.eval(model.parameters())


# --------------------------------------------------------------------------- batching


def length_batches(lengths: Sequence[int], token_budget: int) -> list[list[int]]:
    """Group indices by ascending length so ``len(batch) * max_len <= token_budget``.

    An item longer than the budget still runs, alone: the budget shrinks batches, it never
    drops work.
    """
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    batches: list[list[int]] = []
    current: list[int] = []
    for i in order:
        # sorted ascending, so lengths[i] is the padded length of current + [i]
        if current and (len(current) + 1) * lengths[i] > token_budget:
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


def pad_batch(seqs: Iterable[Sequence[int]], pad_id: int) -> tuple[mx.array, mx.array]:
    seqs = list(seqs)
    width = max(len(s) for s in seqs)
    ids = np.full((len(seqs), width), pad_id, dtype=np.int32)
    mask = np.zeros((len(seqs), width), dtype=np.int32)
    for row, s in enumerate(seqs):
        ids[row, : len(s)] = s
        mask[row, : len(s)] = 1
    return mx.array(ids), mx.array(mask)

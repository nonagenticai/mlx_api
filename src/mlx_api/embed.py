"""bge-m3 dense embeddings: XLM-R, CLS pooling, L2-normalised, 1024-d.

Sync and blocking; the route calls it through ``gpu.run_on_gpu``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_api.errors import Unavailable
from mlx_api.models import (
    XLMRConfig,
    XLMREncoder,
    length_batches,
    load_weights,
    pad_batch,
    resolve_model_files,
)


@dataclass(frozen=True)
class EmbedResult:
    vectors: np.ndarray  # (n, dim) float32, L2-normalised, input order
    prompt_tokens: int


class HFTokenizer:
    """The model's own Hugging Face tokenizer, so token ids match the PyTorch reference exactly.

    (Raw ``tokenizers.Tokenizer.from_file`` differs from ``AutoTokenizer`` on edge cases such as
    trailing whitespace, which the parity test caught.)
    """

    def __init__(self, directory: Path, max_length: int) -> None:
        from transformers import AutoTokenizer  # noqa: PLC0415 - heavy import, load time only

        tok = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)  # nosec B615 - local dir
        if tok is None:
            raise Unavailable(f"no tokenizer in {directory}")
        self.tok: Any = tok
        self.max_length = max_length

    def encode(self, texts: list[str]) -> list[list[int]]:
        enc = self.tok(texts, truncation=True, max_length=self.max_length)
        return [list(ids) for ids in enc["input_ids"]]

    def encode_pairs(self, query: str, docs: list[str]) -> list[list[int]]:
        enc = self.tok([query] * len(docs), docs, truncation=True, max_length=self.max_length)
        return [list(ids) for ids in enc["input_ids"]]


MODEL_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]


class Embedder:
    def __init__(
        self,
        model_id: str,
        *,
        max_length: int = 8192,
        token_budget: int = 16384,
        dtype: str = "float32",
    ) -> None:
        files = resolve_model_files(model_id, MODEL_FILES)
        self.config = XLMRConfig.from_file(files["config.json"])
        if max_length > self.config.max_tokens:
            raise Unavailable(
                f"max_length {max_length} exceeds the model's {self.config.max_tokens} positions"
            )
        self.model_id = model_id
        self.dim = self.config.hidden_size
        self.max_length = max_length
        self.token_budget = token_budget
        self.tokenizer = HFTokenizer(files["tokenizer_config.json"].parent, max_length)
        self.model = XLMREncoder(self.config)
        load_weights(self.model, files["model.safetensors"], dtype, classifier=False)

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        return self.tokenizer.encode(texts)

    def embed(self, texts: list[str]) -> EmbedResult:
        seqs = self.tokenize(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for batch in length_batches([len(s) for s in seqs], self.token_budget):
            ids, mask = pad_batch((seqs[i] for i in batch), self.config.pad_token_id)
            cls = self.model(ids, mask)[:, 0, :].astype(mx.float32)
            cls = cls / mx.maximum(mx.linalg.norm(cls, axis=-1, keepdims=True), 1e-12)
            mx.eval(cls)
            out[batch] = np.array(cls)
        return EmbedResult(vectors=out, prompt_tokens=sum(len(s) for s in seqs))

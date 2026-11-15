"""bge-reranker-v2-m3 cross-encoder: sigmoid(logit) per (query, document) pair.

Sync and blocking; the route calls it through ``gpu.run_on_gpu``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_api.embed import MODEL_FILES, HFTokenizer
from mlx_api.errors import Unavailable
from mlx_api.models import (
    XLMRClassifier,
    XLMRConfig,
    length_batches,
    load_weights,
    pad_batch,
    resolve_model_files,
)


@dataclass(frozen=True)
class RerankResult:
    scores: list[float]  # in [0, 1], document order
    tokens: int


class Reranker:
    def __init__(
        self,
        model_id: str,
        *,
        max_length: int = 1024,
        token_budget: int = 8192,
        dtype: str = "float32",
    ) -> None:
        files = resolve_model_files(model_id, MODEL_FILES)
        self.config = XLMRConfig.from_file(files["config.json"])
        if self.config.num_labels != 1:
            raise Unavailable(f"reranker must have one label, has {self.config.num_labels}")
        if max_length > self.config.max_tokens:
            raise Unavailable(
                f"max_length {max_length} exceeds the model's {self.config.max_tokens} positions"
            )
        self.model_id = model_id
        self.max_length = max_length
        self.token_budget = token_budget
        self.tokenizer = HFTokenizer(files["tokenizer_config.json"].parent, max_length)
        self.model = XLMRClassifier(self.config)
        load_weights(self.model, files["model.safetensors"], dtype, classifier=True)

    def tokenize(self, query: str, documents: list[str]) -> list[list[int]]:
        return self.tokenizer.encode_pairs(query, documents)

    def score(self, query: str, documents: list[str]) -> RerankResult:
        seqs = self.tokenize(query, documents)
        out = np.zeros(len(documents), dtype=np.float32)
        for batch in length_batches([len(s) for s in seqs], self.token_budget):
            ids, mask = pad_batch((seqs[i] for i in batch), self.config.pad_token_id)
            probs = mx.sigmoid(self.model(ids, mask)[:, 0].astype(mx.float32))
            mx.eval(probs)
            out[batch] = np.array(probs)
        return RerankResult(scores=[float(s) for s in out], tokens=sum(len(s) for s in seqs))

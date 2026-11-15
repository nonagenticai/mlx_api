"""/v1/embeddings route (fake engine) + batching/resolution units + MLX-vs-torch parity."""

from __future__ import annotations

import base64
import os
from pathlib import Path

import numpy as np
import pytest
from conftest import AUTH, Fakes, assert_contract_error
from fastapi.testclient import TestClient

from mlx_api.errors import Unavailable
from mlx_api.models import length_batches, resolve_model_files

# ----------------------------------------------------------------- route


def post(client: TestClient, **body: object):
    return client.post("/v1/embeddings", json={"model": "bge-m3", **body}, headers=AUTH)


def test_float_embeddings_in_input_order(client: TestClient, fakes: Fakes) -> None:
    texts = ["alpha beta", "gamma", "delta epsilon zeta"]
    resp = post(client, input=texts, input_type="query")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["object"] == "list"
    assert body["model"] == "bge-m3"
    assert [d["index"] for d in body["data"]] == [0, 1, 2]
    expected = fakes.embed.embed(texts).vectors
    for d, want in zip(body["data"], expected, strict=True):
        assert d["object"] == "embedding"
        vec = np.asarray(d["embedding"], dtype=np.float32)
        assert vec.shape == (1024,)
        assert abs(float(np.linalg.norm(vec)) - 1) < 1e-3
        np.testing.assert_allclose(vec, want, rtol=1e-6)
    assert body["usage"] == {"prompt_tokens": 6, "total_tokens": 6}


def test_string_input_and_full_model_id(client: TestClient) -> None:
    resp = client.post(
        "/v1/embeddings",
        json={"model": "BAAI/bge-m3", "input": "one", "dimensions": 1024},
        headers=AUTH,
    )
    assert resp.status_code == 200
    assert len(resp.json()["data"]) == 1


def test_base64_is_little_endian_float32(client: TestClient) -> None:
    f = post(client, input=["x y", "z"]).json()["data"]
    b = post(client, input=["x y", "z"], encoding_format="base64").json()["data"]
    for fd, bd in zip(f, b, strict=True):
        decoded = np.frombuffer(base64.b64decode(bd["embedding"]), dtype="<f4")
        np.testing.assert_array_equal(decoded, np.asarray(fd["embedding"], dtype=np.float32))


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"input": []}, 400),
        ({"input": ""}, 400),
        ({"input": ["ok", ""]}, 400),
        ({"input": ["x"] * 2049}, 413),
        ({"input": "x", "dimensions": 512}, 400),
        ({"input": "x", "encoding_format": "int8"}, 400),
        ({"input": "x", "input_type": "passage"}, 400),
        ({"input": 5}, 400),
        ({}, 400),
    ],
)
def test_invalid_requests(client: TestClient, body: dict[str, object], status: int) -> None:
    assert_contract_error(post(client, **body), status, "invalid_request")


def test_max_batch_accepted(client: TestClient) -> None:
    resp = post(client, input=[f"t{i}" for i in range(2048)], encoding_format="base64")
    assert resp.status_code == 200
    assert len(resp.json()["data"]) == 2048


@pytest.mark.parametrize("model", ["bge-reranker-v2-m3", "text-embedding-3-small", "BAAI/bge"])
def test_unknown_model_is_404(client: TestClient, model: str) -> None:
    resp = client.post("/v1/embeddings", json={"model": model, "input": "x"}, headers=AUTH)
    assert_contract_error(resp, 404, "not_found")


# ----------------------------------------------------------------- units


def test_length_batches_sorted_and_within_budget() -> None:
    lengths = [5, 300, 7, 7, 120, 9000, 6, 300]
    batches = length_batches(lengths, 1024)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(lengths)))  # every item exactly once
    assert [lengths[i] for i in flat] == sorted(lengths)
    for b in batches:
        assert len(b) == 1 or len(b) * max(lengths[i] for i in b) <= 1024
    assert [5] in batches  # the over-budget item runs alone, never dropped


def test_length_batches_packs_uniform_lengths() -> None:
    assert length_batches([100] * 10, 500) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]


def _fake_cache(root: Path, model: str, snapshots: dict[str, list[str]], main: str) -> None:
    repo = root / ("models--" + model.replace("/", "--"))
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(main)
    for rev, names in snapshots.items():
        (repo / "snapshots" / rev).mkdir(parents=True)
        for n in names:
            (repo / "snapshots" / rev / n).write_text(rev)


def test_resolve_prefers_main_then_falls_back(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    _fake_cache(
        tmp_path,
        "org/m",
        {"aaa": ["model.safetensors"], "main1": ["config.json", "tokenizer.json"]},
        "main1",
    )
    files = resolve_model_files("org/m", ["config.json", "tokenizer.json", "model.safetensors"])
    assert files["config.json"].read_text() == "main1"
    assert files["model.safetensors"].read_text() == "aaa"
    with pytest.raises(Unavailable):
        resolve_model_files("org/m", ["missing.bin"])
    with pytest.raises(Unavailable):
        resolve_model_files("org/absent", ["config.json"])


# ----------------------------------------------------------------- parity (real weights)

PARITY_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "What is the capital of France?",
    "Die Katze sitzt auf der Matte und schläft in der Sonne.",
    "机器学习是人工智能的一个分支，它使计算机能够从数据中学习。",
    "Пример текста на русском языке для проверки эмбеддингов.",
    "def add(a, b):\n    return a + b  # simple python function",
    "x",
    "Retrieval-augmented generation combines a retriever with a generator. " * 60,
    "Les modèles d'embedding multilingues associent des phrases de langues différentes.",
    "🙂 emoji and   odd   whitespace\tand\nnewlines",
]


def _linked_dir(tmp_path: Path, files: dict[str, Path]) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    for name, src in files.items():
        (d / name).symlink_to(src.resolve())
    return d


@pytest.mark.models
def test_bge_m3_parity_with_torch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from mlx_api.embed import Embedder

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    names = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]
    names += ["special_tokens_map.json", "sentencepiece.bpe.model"]
    ref_dir = _linked_dir(tmp_path, resolve_model_files("BAAI/bge-m3", names))

    tok = transformers.AutoTokenizer.from_pretrained(ref_dir)
    model = transformers.AutoModel.from_pretrained(ref_dir, torch_dtype=torch.float32).eval()
    ref_tokens = 0
    ref = []
    with torch.no_grad():
        for t in PARITY_TEXTS:
            enc = tok(t, truncation=True, max_length=8192, return_tensors="pt")
            ref_tokens += int(enc["input_ids"].shape[1])
            cls = model(**enc).last_hidden_state[:, 0]
            ref.append(torch.nn.functional.normalize(cls, dim=-1)[0].numpy())
    assert max(len(tok(t)["input_ids"]) for t in PARITY_TEXTS) > 600
    del model

    emb = Embedder("BAAI/bge-m3")
    got = emb.embed(PARITY_TEXTS)
    assert got.prompt_tokens == ref_tokens
    cosines = [float(np.dot(a, b)) for a, b in zip(got.vectors, ref, strict=True)]
    print("bge-m3 parity cosines:", [round(c, 6) for c in cosines], "min", min(cosines))
    assert min(cosines) >= 0.999, cosines
    # batching must not change results: each text alone == the batched result
    alone = np.stack([emb.embed([t]).vectors[0] for t in PARITY_TEXTS])
    assert float(np.min(np.sum(alone * got.vectors, axis=1))) >= 0.9999

"""/v2/rerank route (fake engine) + MLX-vs-torch parity."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from conftest import AUTH, Fakes, assert_contract_error
from fastapi.testclient import TestClient

from mlx_api.models import resolve_model_files


def post(client: TestClient, **body: object):
    payload = {"model": "bge-reranker-v2-m3", "query": "q", **body}
    return client.post("/v2/rerank", json=payload, headers=AUTH)


def test_sorted_desc_ties_by_lower_index(client: TestClient, fakes: Fakes) -> None:
    fakes.rerank.scores = {"a": 0.2, "b": 0.9, "c": 0.5, "d": 0.9, "e": 0.5}
    resp = post(client, documents=["a", "b", "c", "d", "e"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    uuid.UUID(body["id"])
    assert body["results"] == [
        {"index": 1, "relevance_score": 0.9},
        {"index": 3, "relevance_score": 0.9},
        {"index": 2, "relevance_score": 0.5},
        {"index": 4, "relevance_score": 0.5},
        {"index": 0, "relevance_score": 0.2},
    ]
    assert body["meta"]["billed_units"] == {"search_units": 1}


def test_top_n_and_cap(client: TestClient, fakes: Fakes) -> None:
    fakes.rerank.scores = {"a": 0.1, "b": 0.7, "c": 0.4}
    top = post(client, documents=["a", "b", "c"], top_n=2).json()["results"]
    assert [r["index"] for r in top] == [1, 2]
    capped = post(client, documents=["a", "b", "c"], top_n=50).json()["results"]
    assert [r["index"] for r in capped] == [1, 2, 0]


def test_scores_in_unit_interval_and_full_model_id(client: TestClient) -> None:
    resp = client.post(
        "/v2/rerank",
        json={"model": "BAAI/bge-reranker-v2-m3", "query": "q", "documents": ["x", "y"]},
        headers=AUTH,
    )
    assert resp.status_code == 200
    assert all(0 <= r["relevance_score"] <= 1 for r in resp.json()["results"])


def test_max_documents(client: TestClient) -> None:
    ok = post(client, documents=["d"] * 1000)
    assert ok.status_code == 200
    assert len(ok.json()["results"]) == 1000
    assert len({r["index"] for r in ok.json()["results"]}) == 1000
    assert_contract_error(post(client, documents=["d"] * 1001), 413, "invalid_request")


@pytest.mark.parametrize(
    "body",
    [
        {"documents": []},
        {"documents": ["a"], "query": ""},
        {"documents": ["a"], "top_n": 0},
        {"documents": "a"},
        {"query": "q"},
    ],
)
def test_invalid_requests(client: TestClient, body: dict[str, object]) -> None:
    assert_contract_error(post(client, **body), 400, "invalid_request")


def test_unknown_model_is_404(client: TestClient) -> None:
    resp = client.post(
        "/v2/rerank",
        json={"model": "rerank-v3.5", "query": "q", "documents": ["a"]},
        headers=AUTH,
    )
    assert_contract_error(resp, 404, "not_found")


# ----------------------------------------------------------------- parity (real weights)

QUERY_DOCS = [
    (
        "what is panda?",
        [
            "hi",
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply "
            "panda, is a bear species endemic to China.",
            "Pandas eat bamboo. " * 80,
            "Der Große Panda ist eine Bärenart aus China.",
        ],
    ),
    (
        "Wie hoch ist der Eiffelturm?",
        [
            "The Eiffel Tower is 330 metres tall.",
            "Der Eiffelturm ist ein 330 Meter hoher Eisenfachwerkturm in Paris.",
            "Berlin ist die Hauptstadt Deutschlands.",
            "def eiffel(): return 330",
        ],
    ),
]


@pytest.mark.models
def test_bge_reranker_parity_with_torch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from mlx_api.rerank import Reranker

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    names = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]
    names += ["special_tokens_map.json", "sentencepiece.bpe.model"]
    files = resolve_model_files("BAAI/bge-reranker-v2-m3", names)
    ref_dir = tmp_path / "model"
    ref_dir.mkdir()
    for name, src in files.items():
        (ref_dir / name).symlink_to(src.resolve())

    tok = transformers.AutoTokenizer.from_pretrained(ref_dir)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        ref_dir, torch_dtype=torch.float32
    ).eval()
    ref: list[list[float]] = []
    with torch.no_grad():
        for q, docs in QUERY_DOCS:
            enc = tok(
                [[q, d] for d in docs],
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
            ref.append(torch.sigmoid(model(**enc).logits[:, 0]).tolist())
    del model

    rr = Reranker("BAAI/bge-reranker-v2-m3", max_length=1024)
    diffs = []
    for (q, docs), want in zip(QUERY_DOCS, ref, strict=True):
        got = rr.score(q, docs).scores
        diffs += [abs(a - b) for a, b in zip(got, want, strict=True)]
        print("rerank", q, [round(s, 4) for s in got], "torch", [round(s, 4) for s in want])
    print("rerank parity max |diff|:", max(diffs))
    assert max(diffs) <= 0.01, diffs
    # the relevant English passage must outrank the greeting
    s = rr.score("what is panda?", QUERY_DOCS[0][1][:2]).scores
    assert s[1] > 0.5 > s[0]

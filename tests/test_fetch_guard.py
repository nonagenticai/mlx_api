"""OCR document URLs: only public hosts, no redirects (private networks are refused)."""

import pytest

from mlx_api.app import fetch_url
from mlx_api.errors import InvalidRequest


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:9131/health",
        "http://localhost/x.pdf",
        "http://10.0.0.5/x.pdf",
        "http://192.168.1.10/x.pdf",
        "http://169.254.169.254/latest/meta-data",
        "http://100.64.0.1:8389/bucket/x.pdf",  # RFC 6598 CGNAT boundary, not a host
        "http://[::1]/x.pdf",
    ],
)
async def test_non_public_targets_are_refused(url: str) -> None:
    with pytest.raises(InvalidRequest) as exc:
        await fetch_url(url)
    assert exc.value.code == "document_url_not_public"


async def test_unresolvable_host_is_invalid_request() -> None:
    with pytest.raises(InvalidRequest) as exc:
        await fetch_url("http://does-not-exist.invalid/x.pdf")
    assert exc.value.code == "document_fetch_failed"

"""No-metadata Origin fallback for cookie-authenticated writes.

Contract: when the browser sends no ``Sec-Fetch-Site`` (older browsers,
stripped headers), a write whose ``Origin`` matches the addressed origin is
allowed even if ``dashboard.public_url`` points elsewhere — one config serves
surfaces with different public origins (serve behind tailscale-serve https,
webapp browsed directly by IP). A genuinely cross-origin write stays denied.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import Request

from hermes_cli.dashboard_auth.request_utils import cookie_origin_is_allowed

PUBLIC_URL = "https://m26pipeline.tail7c213b.ts.net"
BOUND_IP = "100.114.240.126"


def _request(*, origin: str, host: str, fetch_site: str | None = None) -> Request:
    headers = [(b"host", host.encode()), (b"origin", origin.encode())]
    if fetch_site is not None:
        headers.append((b"sec-fetch-site", fetch_site.encode()))
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/auth/ws-ticket",
        "headers": headers,
        "scheme": "http",
        "server": (BOUND_IP, 9120),
        "client": (BOUND_IP, 50000),
        "app": SimpleNamespace(state=SimpleNamespace(
            bound_host=BOUND_IP,
            trusted_public_hosts=frozenset({"m26pipeline.tail7c213b.ts.net"}))),
    }
    return Request(scope)


@pytest.fixture
def serve_public_url(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.dashboard_auth.prefix.resolve_public_url", lambda: PUBLIC_URL)


def test_direct_origin_allowed_despite_public_url(serve_public_url):
    """Same-origin write without fetch metadata is allowed on a direct bind."""
    request = _request(origin=f"http://{BOUND_IP}:9120", host=f"{BOUND_IP}:9120")

    assert cookie_origin_is_allowed(request) is True


def test_cross_origin_still_denied_without_metadata(serve_public_url):
    """A forged cross-origin write without fetch metadata stays denied."""
    request = _request(origin="https://evil.example.com", host=f"{BOUND_IP}:9120")

    assert cookie_origin_is_allowed(request) is False

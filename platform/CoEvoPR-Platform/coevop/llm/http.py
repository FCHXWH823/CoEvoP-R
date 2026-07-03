"""Small JSON HTTP client for provider adapters."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class LLMHTTPError(RuntimeError):
    pass


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float = 120.0,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMHTTPError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMHTTPError(f"request to {url} failed: {exc.reason}") from exc

    try:
        return json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise LLMHTTPError(f"non-JSON response from {url}: {response_body[:500]}") from exc

"""SciEvalBook HTTP client — stdlib only, no external dependencies.

Fetches aggregated community knowledge from the SEB
``/api/community-knowledge/{id}`` endpoint.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def fetch_community_knowledge(
    base_url: str,
    challenge_id: str,
    *,
    timeout: int = 30,
    snapshot_path: Path | None = None,
) -> dict[str, Any] | None:
    """Fetch aggregated community data for a single challenge.

    Parameters
    ----------
    base_url:
        SciEvalBook API root, e.g. ``http://localhost:8000``.
    challenge_id:
        UUID of the challenge / datapoint.
    timeout:
        HTTP timeout in seconds.
    snapshot_path:
        If given, the raw JSON response is persisted here for
        reproducibility and offline analysis.

    Returns
    -------
    dict or None
        Parsed JSON on success; *None* on any network/parse failure.
    """
    url = f"{base_url.rstrip('/')}/api/community-knowledge/{challenge_id}"
    logger.info("Fetching community knowledge from %s", url)

    try:
        req = urllib.request.Request(url, method="GET")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except (urllib.error.URLError, OSError) as exc:
        logger.warning("SEB request failed: %s", exc)
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("SEB response not valid JSON: %s", exc)
        return None

    # Persist snapshot
    if snapshot_path is not None:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Community snapshot saved to %s", snapshot_path)

    return data

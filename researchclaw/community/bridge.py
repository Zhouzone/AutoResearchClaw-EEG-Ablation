"""Community bridge — fetches SEB data, extracts ideas, generates knowledge cards.

This module sits between the raw SEB API client and the pipeline executor.
It transforms community data into artifacts the pipeline can consume:
knowledge cards, idea summaries, and a problem anchor.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from researchclaw.community.seb_client import fetch_community_knowledge

logger = logging.getLogger(__name__)


@dataclass
class CommunityResult:
    """Outputs of the community bridge stage."""

    ideas: list[dict[str, Any]]
    """Extracted ideas from high-score model answers."""

    cards_text: str
    """Concatenated knowledge-card markdown (ready to append to cards_context)."""

    problem_anchor: str
    """Best community understanding of the core problem."""

    snapshot: dict[str, Any] | None = None
    """Raw community API response (for logging/debugging)."""

    challenge_titles: list[str] = field(default_factory=list)
    """Titles of all fetched challenges."""


def _extract_ideas_from_answer(answer_content: str) -> list[str]:
    """Split a model answer into individual idea bullets.

    Looks for numbered items (1. / 2. / …), heading-based sections
    (## Hypothesis / ## 假设), or Chinese enumeration patterns (假设一).
    Falls back to treating the whole answer as one idea.
    """
    # Try numbered list
    numbered = re.split(r"\n\s*\d+[\.\)]\s+", "\n" + answer_content)
    numbered = [s.strip() for s in numbered if s.strip()]
    if len(numbered) >= 2:
        return numbered

    # Try heading-based split
    headed = re.split(r"\n##\s+", "\n" + answer_content)
    headed = [s.strip() for s in headed if s.strip()]
    if len(headed) >= 2:
        return headed

    # Try Chinese enumeration
    cn_enum = re.split(r"\n\s*(?:假设|方法|思路)[一二三四五六七八九十]\s*[:：]?\s*", "\n" + answer_content)
    cn_enum = [s.strip() for s in cn_enum if s.strip()]
    if len(cn_enum) >= 2:
        return cn_enum

    return [answer_content.strip()] if answer_content.strip() else []


def _make_knowledge_card(
    idx: int,
    model_name: str,
    score: float | None,
    content: str,
    ideas: list[str],
) -> str:
    """Format a single model answer as a markdown knowledge card."""
    score_str = f"{score:.1f}" if score is not None else "N/A"
    lines = [
        f"# Community Card #{idx}: {model_name} (score {score_str})",
        "",
        content[:2000],  # Truncate very long answers
        "",
        "## Extracted Ideas",
        "",
    ]
    for j, idea in enumerate(ideas, 1):
        lines.append(f"{j}. {idea[:500]}")
    return "\n".join(lines)


def run_community_bridge(
    community_config: Any,
    run_dir: Path,
    llm: Any = None,
) -> CommunityResult:
    """Execute the community bridge: fetch → extract → cards.

    Parameters
    ----------
    community_config:
        A ``CommunityConfig`` instance with ``seb_base_url``,
        ``challenge_ids``, ``min_score``, etc.
    run_dir:
        Pipeline run directory. Artifacts are written under
        ``{run_dir}/community/``.
    llm:
        LLM client (unused in this version but reserved for
        future LLM-assisted extraction).

    Returns
    -------
    CommunityResult
    """
    community_dir = run_dir / "community"
    community_dir.mkdir(parents=True, exist_ok=True)

    all_ideas: list[dict[str, Any]] = []
    all_cards: list[str] = []
    combined_snapshot: dict[str, Any] = {"challenges": []}
    best_anchor = ""
    best_anchor_score = -1.0
    challenge_titles: list[str] = []

    challenge_ids = community_config.challenge_ids
    if not challenge_ids:
        logger.warning("No challenge_ids configured — skipping community bridge")
        return CommunityResult(
            ideas=[], cards_text="", problem_anchor="", challenge_titles=[]
        )

    for cid in challenge_ids:
        snapshot_path = community_dir / f"snapshot_{cid[:8]}.json"
        data = fetch_community_knowledge(
            base_url=community_config.seb_base_url,
            challenge_id=cid,
            snapshot_path=snapshot_path,
        )
        if data is None:
            logger.warning("Failed to fetch community data for %s", cid)
            continue

        combined_snapshot["challenges"].append(data)
        challenge_info = data.get("challenge", {})
        challenge_titles.append(challenge_info.get("title", cid))

        # Extract ideas from high-score answers
        card_idx = len(all_cards) + 1
        for answer in data.get("model_answers", []):
            score = answer.get("score")
            if score is not None and score < community_config.min_score:
                continue

            content = answer.get("content", "")
            ideas = _extract_ideas_from_answer(content)

            all_ideas.append(
                {
                    "challenge_id": cid,
                    "model_name": answer.get("model_name", "unknown"),
                    "score": score,
                    "ideas": ideas,
                    "content_preview": content[:300],
                }
            )

            card_md = _make_knowledge_card(
                card_idx, answer.get("model_name", "unknown"), score, content, ideas
            )
            all_cards.append(card_md)
            card_idx += 1

            # Track best anchor
            effective_score = score if score is not None else 0
            if effective_score > best_anchor_score:
                best_anchor_score = effective_score
                best_anchor = content

    # Save combined snapshot
    (community_dir / "community_snapshot.json").write_text(
        json.dumps(combined_snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Save ideas summary
    cards_text = "\n\n---\n\n".join(all_cards)
    if cards_text:
        (community_dir / "community_cards.md").write_text(cards_text, encoding="utf-8")

    # Save ideas index
    ideas_md_lines = ["# Community Ideas Summary", ""]
    for i, idea_group in enumerate(all_ideas, 1):
        score_str = f"{idea_group['score']:.1f}" if idea_group['score'] is not None else "N/A"
        ideas_md_lines.append(
            f"## {i}. {idea_group['model_name']} (score {score_str})"
        )
        for j, idea in enumerate(idea_group["ideas"], 1):
            ideas_md_lines.append(f"  {j}. {idea[:300]}")
        ideas_md_lines.append("")

    (community_dir / "community_ideas.md").write_text(
        "\n".join(ideas_md_lines), encoding="utf-8"
    )

    logger.info(
        "Community bridge: %d challenges, %d high-score answers, %d idea groups",
        len(challenge_ids),
        len(all_ideas),
        sum(len(ig["ideas"]) for ig in all_ideas),
    )

    return CommunityResult(
        ideas=all_ideas,
        cards_text=cards_text,
        problem_anchor=best_anchor,
        snapshot=combined_snapshot,
        challenge_titles=challenge_titles,
    )

"""Idea refinement loop — structured proposal refinement.

Adapted from Auto-Research-Refine's 5-phase workflow, simplified to
2-3 rounds for time efficiency. Each round is fully logged.

Phases:
  0 — Problem Anchor (freeze from community best answer)
  1 — Initial Proposal (merge literature + community top ideas)
  2 — Review (LLM self-evaluation on 5 dimensions)
  3 — Revise (anchor-preserving revision)
  4 — Final Output (clean proposal + logs)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Review dimensions ──────────────────────────────────────────────

REVIEW_DIMENSIONS = [
    ("problem_fidelity", "Does the method still attack the original bottleneck?"),
    ("method_specificity", "Is the method concrete enough to implement?"),
    ("contribution_quality", "Is there one dominant contribution without sprawl?"),
    ("feasibility", "Can it run on the stated hardware (e.g. 6GB GPU)?"),
    ("validation_focus", "Are experiments minimal but sufficient?"),
]

REVIEW_PROMPT_TEMPLATE = """\
You are a senior ML reviewer. Evaluate the following research proposal.

## Problem Anchor (immutable)
{problem_anchor}

## Proposal
{proposal}

## Evaluation Dimensions
For each dimension, give a score 1-10 and a brief justification.

1. Problem Fidelity — Does the method still attack the original bottleneck?
2. Method Specificity — Is the method concrete enough to implement (losses, architectures, training stages)?
3. Contribution Quality — One dominant mechanism, good parsimony, no sprawl?
4. Feasibility — Can be trained with stated resources?
5. Validation Focus — Minimal but sufficient experiments? No bloat?

Output JSON:
{{
  "scores": {{
    "problem_fidelity": {{"score": N, "reason": "..."}},
    "method_specificity": {{"score": N, "reason": "..."}},
    "contribution_quality": {{"score": N, "reason": "..."}},
    "feasibility": {{"score": N, "reason": "..."}},
    "validation_focus": {{"score": N, "reason": "..."}}
  }},
  "overall_score": N,
  "verdict": "READY|REVISE",
  "key_feedback": "one paragraph of actionable feedback"
}}
"""

REVISE_PROMPT_TEMPLATE = """\
You are revising a research proposal based on reviewer feedback.

## CRITICAL: Problem Anchor (DO NOT change the problem being solved)
{problem_anchor}

## Current Proposal
{proposal}

## Reviewer Feedback
{feedback}

## Instructions
- Fix the issues raised by the reviewer
- Keep the Problem Anchor unchanged — you are solving the SAME problem
- Prefer simplicity: fewer moving parts, cleaner reuse
- Output a COMPLETE revised proposal (not incremental patches)
- If reviewer suggestions would change the core problem, explicitly note this as "drift" and skip that suggestion
"""

ANCHOR_PROMPT_TEMPLATE = """\
Extract a Problem Anchor from the following high-scoring community answer.
The anchor should be a concise document with these sections:

1. **Bottom-line Problem**: The specific research gap or limitation
2. **Must-solve Bottleneck**: The core technical challenge
3. **Non-goals**: What we are NOT trying to solve
4. **Constraints**: Hardware, data, time limitations
5. **Success Condition**: What "solving this" looks like

## Community Answer (highest scored)
{best_answer}

## Research Topic
{topic}

## Literature Synthesis
{synthesis}

Output a clean markdown document with the 5 sections above.
"""

INITIAL_PROPOSAL_TEMPLATE = """\
Generate a focused, elegant research proposal based on the following inputs.

## Problem Anchor
{problem_anchor}

## Literature Synthesis
{synthesis}

## Top Community Ideas (ranked by score)
{community_ideas}

## Instructions
- Prioritize the direction from the highest-scored community answer
- Use literature findings to ground the technical details
- Propose ONE dominant contribution (not multiple parallel ideas)
- Be specific: name architectures, loss functions, training stages
- Include a brief validation plan (1-3 core experiments)
- Respect hardware constraints mentioned in the anchor

Output a complete proposal in markdown.
"""


@dataclass
class RefineResult:
    """Outputs of the refinement loop."""

    final_proposal: str
    """The clean, final research proposal."""

    problem_anchor: str
    """The frozen problem anchor document."""

    rounds: list[dict[str, Any]]
    """Per-round records: proposal, review scores, verdict."""

    overall_score: float
    """Final review score achieved."""

    verdict: str
    """Final verdict: READY or REVISE."""


def refine_community_ideas(
    community_result: Any,
    literature_synthesis: str,
    config: Any,
    llm: Any,
    log_dir: Path,
    *,
    topic: str = "",
) -> RefineResult:
    """Run the structured refinement loop.

    Parameters
    ----------
    community_result:
        Output of ``run_community_bridge()``.
    literature_synthesis:
        Synthesis text from Stage 7 cards.
    config:
        ``CommunityConfig`` with ``max_refine_rounds`` and ``refine_threshold``.
    llm:
        LLM client with ``.chat()`` method.
    log_dir:
        Directory for all refinement artifacts.
    topic:
        Research topic string.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    rounds: list[dict[str, Any]] = []

    max_rounds = getattr(config, "max_refine_rounds", 2)
    threshold = getattr(config, "refine_threshold", 8.0)

    # ── Phase 0: Problem Anchor ──────────────────────────────────

    if llm is not None:
        anchor_prompt = ANCHOR_PROMPT_TEMPLATE.format(
            best_answer=community_result.problem_anchor[:3000],
            topic=topic or "the research topic",
            synthesis=literature_synthesis[:2000],
        )
        resp = llm.chat(
            [{"role": "user", "content": anchor_prompt}],
            system="You are a research methodology expert. Be concise and precise.",
        )
        problem_anchor = resp.content
    else:
        problem_anchor = (
            f"# Problem Anchor\n\n"
            f"Based on community knowledge:\n\n"
            f"{community_result.problem_anchor[:1500]}"
        )

    (log_dir / "problem-anchor.md").write_text(problem_anchor, encoding="utf-8")
    logger.info("Phase 0: Problem anchor generated (%d chars)", len(problem_anchor))

    # ── Phase 1: Initial Proposal ────────────────────────────────

    # Format top ideas
    ideas_text_parts = []
    for i, idea_group in enumerate(community_result.ideas[:5], 1):
        score_str = f"{idea_group['score']:.1f}" if idea_group.get("score") is not None else "N/A"
        ideas_text_parts.append(
            f"### Idea {i} ({idea_group.get('model_name', '?')}, score {score_str})\n"
            + "\n".join(f"- {idea[:300]}" for idea in idea_group.get("ideas", []))
        )
    community_ideas_text = "\n\n".join(ideas_text_parts) or "(no community ideas available)"

    if llm is not None:
        proposal_prompt = INITIAL_PROPOSAL_TEMPLATE.format(
            problem_anchor=problem_anchor,
            synthesis=literature_synthesis[:3000],
            community_ideas=community_ideas_text,
        )
        resp = llm.chat(
            [{"role": "user", "content": proposal_prompt}],
            system="You are a senior ML researcher. Write focused, implementable proposals.",
        )
        current_proposal = resp.content
    else:
        current_proposal = (
            f"# Initial Proposal\n\n"
            f"Based on {len(community_result.ideas)} community ideas "
            f"and literature synthesis.\n\n"
            f"## Community Direction\n{community_ideas_text}\n\n"
            f"## Literature Grounding\n{literature_synthesis[:1000]}"
        )

    (log_dir / "round-0-proposal.md").write_text(current_proposal, encoding="utf-8")
    logger.info("Phase 1: Initial proposal generated (%d chars)", len(current_proposal))

    # ── Phases 2-3: Review + Revise loop ─────────────────────────

    overall_score = 0.0
    verdict = "REVISE"

    for round_num in range(1, max_rounds + 1):
        # Phase 2: Review
        if llm is not None:
            review_prompt = REVIEW_PROMPT_TEMPLATE.format(
                problem_anchor=problem_anchor,
                proposal=current_proposal[:4000],
            )
            resp = llm.chat(
                [{"role": "user", "content": review_prompt}],
                system="You are a rigorous ML reviewer. Output valid JSON only.",
                json_mode=True,
            )
            try:
                review = json.loads(resp.content)
            except json.JSONDecodeError:
                review = {
                    "scores": {},
                    "overall_score": 5.0,
                    "verdict": "REVISE",
                    "key_feedback": resp.content[:500],
                }
        else:
            review = {
                "scores": {d[0]: {"score": 7, "reason": "mock"} for d in REVIEW_DIMENSIONS},
                "overall_score": 7.0,
                "verdict": "REVISE",
                "key_feedback": "Mock review — no LLM available.",
            }

        overall_score = float(review.get("overall_score", 0))
        verdict = review.get("verdict", "REVISE")

        (log_dir / f"round-{round_num}-review.md").write_text(
            json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        round_record = {
            "round": round_num,
            "scores": review.get("scores", {}),
            "overall_score": overall_score,
            "verdict": verdict,
            "feedback": review.get("key_feedback", ""),
        }
        rounds.append(round_record)

        logger.info(
            "Phase 2 (round %d): score=%.1f verdict=%s",
            round_num, overall_score, verdict,
        )

        # Check stop condition
        if verdict == "READY" or overall_score >= threshold:
            logger.info("Refinement converged at round %d (score %.1f)", round_num, overall_score)
            break

        # Phase 3: Revise
        if llm is not None:
            revise_prompt = REVISE_PROMPT_TEMPLATE.format(
                problem_anchor=problem_anchor,
                proposal=current_proposal[:4000],
                feedback=review.get("key_feedback", ""),
            )
            resp = llm.chat(
                [{"role": "user", "content": revise_prompt}],
                system="You are revising a research proposal. Output a complete revised proposal.",
            )
            current_proposal = resp.content

        (log_dir / f"round-{round_num}-revision.md").write_text(
            current_proposal, encoding="utf-8"
        )
        logger.info("Phase 3 (round %d): proposal revised (%d chars)", round_num, len(current_proposal))

    # ── Phase 4: Final Output ────────────────────────────────────

    (log_dir / "FINAL_PROPOSAL.md").write_text(current_proposal, encoding="utf-8")

    # Score history
    score_lines = ["# Score History", "", "| Round | Score | Verdict |", "|-------|-------|---------|"]
    for r in rounds:
        score_lines.append(f"| {r['round']} | {r['overall_score']:.1f} | {r['verdict']} |")
    (log_dir / "score-history.md").write_text("\n".join(score_lines), encoding="utf-8")

    # Refinement log
    log_lines = ["# Refinement Log", ""]
    log_lines.append(f"## Problem Anchor\n\n{problem_anchor}\n")
    for r in rounds:
        log_lines.append(f"## Round {r['round']}")
        log_lines.append(f"- Score: {r['overall_score']:.1f}")
        log_lines.append(f"- Verdict: {r['verdict']}")
        log_lines.append(f"- Feedback: {r['feedback']}")
        log_lines.append("")
    (log_dir / "REFINEMENT_LOG.md").write_text("\n".join(log_lines), encoding="utf-8")

    logger.info(
        "Refinement complete: %d rounds, final score=%.1f, verdict=%s",
        len(rounds), overall_score, verdict,
    )

    return RefineResult(
        final_proposal=current_proposal,
        problem_anchor=problem_anchor,
        rounds=rounds,
        overall_score=overall_score,
        verdict=verdict,
    )

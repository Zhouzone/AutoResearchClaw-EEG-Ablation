#!/usr/bin/env python3
"""Ablation study: ARC-only (baseline) vs ARC+Lab (treatment).

Usage:
  python run_ablation.py --prepare              # 有文献: Stage 1-8, 两组假设
  python run_ablation.py --prepare --no-lit     # 无文献: Stage 7-8 only, 两组假设
  python run_ablation.py --run                  # 从假设继续跑实验 (Stage 9-22)
  python run_ablation.py --all                  # prepare + run 一键跑

产出结构:
  artifacts/eeg-ablation/
  ├── with-lit/              # 有文献
  │   ├── shared-literature/ # Stage 1-6 共享
  │   ├── baseline/          # Stage 7-8 无社区
  │   ├── treatment/         # Stage 7-8 有社区
  │   └── pre_judge_report.md
  ├── no-lit/                # 无文献
  │   ├── baseline/          # Stage 7-8 无社区 (无文献卡片)
  │   ├── treatment/         # Stage 7-8 有社区 (无文献卡片)
  │   └── pre_judge_report.md
  └── combined_report.md     # 2×2 汇总对比
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline.runner import execute_pipeline
from researchclaw.pipeline.stages import Stage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("ablation")

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts" / "eeg-ablation"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _copy_stages(src: Path, dst: Path, stages: range) -> None:
    """Copy stage-NN directories from src to dst."""
    dst.mkdir(parents=True, exist_ok=True)
    for s in stages:
        stage_dir = src / f"stage-{s:02d}"
        if stage_dir.is_dir():
            target = dst / f"stage-{s:02d}"
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(stage_dir, target)
            logger.info("Copied %s → %s", stage_dir, target)


def _read_file(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _run_id(prefix: str) -> str:
    return f"rc-{prefix}-{_ts()}"


# ─────────────────────────────────────────────────────────────────
# Core: run one group (baseline or treatment) for Stage 7-8
# ─────────────────────────────────────────────────────────────────

def _run_hypothesis_stages(
    config: RCConfig,
    run_dir: Path,
    label: str,
) -> None:
    """Run Stage 7 (Synthesis) + Stage 8 (Hypothesis Gen) and stop.

    We patch hitl_required_stages to include Stage 9 so the pipeline
    blocks there, and use stop_on_gate=True to exit cleanly.
    """
    logger.info("=" * 60)
    logger.info("%s — Stage 7-8 (Synthesis + Hypothesis)", label)
    logger.info("=" * 60)

    run_dir.mkdir(parents=True, exist_ok=True)

    # Override security config to make Stage 9 a gate, so stop_on_gate works
    from dataclasses import replace as _replace  # noqa: PLC0415
    from researchclaw.config import SecurityConfig  # noqa: PLC0415
    patched_security = _replace(config.security, hitl_required_stages=(9,))
    patched_config = _replace(config, security=patched_security)

    results = execute_pipeline(
        run_dir=run_dir,
        run_id=_run_id(label.lower().replace(" ", "-")),
        config=patched_config,
        adapters=AdapterBundle(),
        from_stage=Stage.SYNTHESIS,
        auto_approve_gates=False,  # don't auto-approve Stage 9
        stop_on_gate=True,         # stop when hitting Stage 9 gate
        skip_noncritical=True,
    )
    logger.info("%s: %d stages completed", label, len(results))


# ─────────────────────────────────────────────────────────────────
# Phase: Prepare with literature (Stage 1-6 shared → 7-8)
# ─────────────────────────────────────────────────────────────────

def prepare_with_lit():
    """有文献: 共享 Stage 1-6 → baseline + treatment Stage 7-8."""
    t0 = time.monotonic()
    out_dir = ARTIFACTS / "with-lit"
    shared_dir = out_dir / "shared-literature"
    baseline_dir = out_dir / "baseline"
    treatment_dir = out_dir / "treatment"

    baseline_config = RCConfig.load(ROOT / "config_eeg_baseline.yaml")
    treatment_config = RCConfig.load(ROOT / "config_eeg_treatment.yaml")

    # 1. Shared literature: Stage 1-6
    logger.info("=" * 60)
    logger.info("WITH-LIT: Shared Literature (Stage 1-6)")
    logger.info("=" * 60)
    shared_dir.mkdir(parents=True, exist_ok=True)

    execute_pipeline(
        run_dir=shared_dir,
        run_id=_run_id("shared-lit"),
        config=baseline_config,
        adapters=AdapterBundle(),
        from_stage=Stage.TOPIC_INIT,
        auto_approve_gates=True,
        stop_on_gate=False,
        skip_noncritical=True,
    )

    # 2. Copy shared Stage 1-6 to both groups
    _copy_stages(shared_dir, baseline_dir, range(1, 7))
    _copy_stages(shared_dir, treatment_dir, range(1, 7))

    # 3. Baseline: Stage 7-8
    _run_hypothesis_stages(baseline_config, baseline_dir, "WITH-LIT Baseline")

    # 4. Treatment: Stage 7-8 (with community)
    _run_hypothesis_stages(treatment_config, treatment_dir, "WITH-LIT Treatment")

    # 5. Report
    _generate_pre_judge_report(out_dir, "with-lit", has_literature=True)

    elapsed = time.monotonic() - t0
    logger.info("WITH-LIT COMPLETE in %.0fs (%.1fmin)", elapsed, elapsed / 60)
    logger.info("Report: %s", out_dir / "pre_judge_report.md")


# ─────────────────────────────────────────────────────────────────
# Phase: Prepare without literature (Stage 7-8 only)
# ─────────────────────────────────────────────────────────────────

def prepare_no_lit():
    """无文献: 直接 Stage 7-8, LLM 靠自身知识生成综述和假设."""
    t0 = time.monotonic()
    out_dir = ARTIFACTS / "no-lit"
    baseline_dir = out_dir / "baseline"
    treatment_dir = out_dir / "treatment"

    baseline_config = RCConfig.load(ROOT / "config_eeg_baseline.yaml")
    treatment_config = RCConfig.load(ROOT / "config_eeg_treatment.yaml")

    # Create empty cards/ directories so Stage 7 (SYNTHESIS) passes
    # its input prerequisite check. With no cards, the LLM synthesizes
    # purely from the research topic string.
    for d in [baseline_dir, treatment_dir]:
        cards_dir = d / "stage-06" / "cards"
        cards_dir.mkdir(parents=True, exist_ok=True)

    # 1. Baseline: Stage 7-8 (no literature cards)
    _run_hypothesis_stages(baseline_config, baseline_dir, "NO-LIT Baseline")

    # 2. Treatment: Stage 7-8 (community only, no literature)
    _run_hypothesis_stages(treatment_config, treatment_dir, "NO-LIT Treatment")

    # 3. Report
    _generate_pre_judge_report(out_dir, "no-lit", has_literature=False)

    elapsed = time.monotonic() - t0
    logger.info("NO-LIT COMPLETE in %.0fs (%.1fmin)", elapsed, elapsed / 60)
    logger.info("Report: %s", out_dir / "pre_judge_report.md")


# ─────────────────────────────────────────────────────────────────
# Report generators
# ─────────────────────────────────────────────────────────────────

def _generate_pre_judge_report(out_dir: Path, variant: str, *, has_literature: bool):
    """Generate hypothesis comparison report for one variant."""
    baseline_dir = out_dir / "baseline"
    treatment_dir = out_dir / "treatment"

    baseline_hyp = _read_file(baseline_dir / "stage-08" / "hypotheses.md")
    treatment_hyp = _read_file(treatment_dir / "stage-08" / "hypotheses.md")
    baseline_syn = _read_file(baseline_dir / "stage-07" / "synthesis.md")
    treatment_syn = _read_file(treatment_dir / "stage-07" / "synthesis.md")

    # Community artifacts (treatment only)
    community_snapshot = _read_file(treatment_dir / "community" / "community_snapshot.json")
    problem_anchor = _read_file(treatment_dir / "community-refine" / "problem-anchor.md")
    score_history = _read_file(treatment_dir / "community-refine" / "score-history.md")
    refinement_log = _read_file(treatment_dir / "community-refine" / "REFINEMENT_LOG.md")
    final_proposal = _read_file(treatment_dir / "community-refine" / "FINAL_PROPOSAL.md")

    # Novelty
    b_nov = _parse_novelty(baseline_dir / "stage-08" / "novelty_report.json")
    t_nov = _parse_novelty(treatment_dir / "stage-08" / "novelty_report.json")

    # Community snapshot summary
    snapshot_summary = _summarize_snapshot(community_snapshot)

    lit_label = "有文献 (Stage 1-6 共享)" if has_literature else "无文献 (纯 LLM 知识)"

    report = f"""# Pre-Judge Report: {variant.upper()}

> 变体: **{lit_label}**
> Generated: {datetime.now(timezone.utc).isoformat(timespec="seconds")}

---

## 1. Synthesis 综述对比

### Baseline 综述 ({"文献卡片" if has_literature else "LLM自身知识"})

{_truncate(baseline_syn, 3000)}

### Treatment 综述 ({"文献卡片 + 社区知识" if has_literature else "社区知识 only"})

{_truncate(treatment_syn, 3000)}

---

## 2. Baseline 假设 (无社区)

{baseline_hyp or "(not yet generated)"}

### Novelty Score: {b_nov}

---

## 3. Treatment: 社区知识获取

### 3.1 社区快照

{snapshot_summary or "(未获取社区数据)"}

### 3.2 Problem Anchor

{problem_anchor or "(未生成)"}

### 3.3 提炼过程

{refinement_log or "(未生成)"}

### 3.4 分数演变

{score_history or "(未生成)"}

### 3.5 最终提炼提案

{_truncate(final_proposal, 3000) if final_proposal else "(未生成)"}

---

## 4. Treatment 假设 (有社区)

{treatment_hyp or "(not yet generated)"}

### Novelty Score: {t_nov}

---

## 5. 对比总结

| 维度 | Baseline | Treatment |
|------|----------|-----------|
| 文献 | {"有" if has_literature else "无"} | {"有" if has_literature else "无"} |
| 社区知识 | 无 | 有 |
| Problem Anchor | 无 (自由辩论) | 社区最高分回答冻结 |
| 提炼轮数 | 0 | ≤2 |
| Novelty | {b_nov} | {t_nov} |

---

## 6. 评分模板

| 假设 | 来源 | 创新性 | 可行性(6GB) | 清晰度 | 总分 | 备注 |
|------|------|--------|------------|--------|------|------|
| B-H1 | baseline | _ | _ | _ | _ | |
| B-H2 | baseline | _ | _ | _ | _ | |
| B-H3 | baseline | _ | _ | _ | _ | |
| T-H1 | treatment | _ | _ | _ | _ | |
| T-H2 | treatment | _ | _ | _ | _ | |
| T-H3 | treatment | _ | _ | _ | _ | |
"""

    out_path = out_dir / "pre_judge_report.md"
    out_path.write_text(report, encoding="utf-8")
    logger.info("Pre-judge report → %s", out_path)


def _generate_combined_report():
    """Generate 2×2 combined comparison across both variants."""
    with_lit = ARTIFACTS / "with-lit"
    no_lit = ARTIFACTS / "no-lit"

    cells = {}
    for variant, vdir in [("with-lit", with_lit), ("no-lit", no_lit)]:
        for group in ["baseline", "treatment"]:
            gdir = vdir / group
            hyp = _read_file(gdir / "stage-08" / "hypotheses.md")
            syn = _read_file(gdir / "stage-07" / "synthesis.md")
            nov = _parse_novelty(gdir / "stage-08" / "novelty_report.json")
            cells[f"{variant}/{group}"] = {
                "hypotheses": hyp,
                "synthesis": syn,
                "novelty": nov,
            }

    report = f"""# Combined 2×2 Ablation Report

> Generated: {datetime.now(timezone.utc).isoformat(timespec="seconds")}

## 实验设计

|  | Baseline (无社区) | Treatment (有社区) |
|--|---|---|
| **有文献** | A | B |
| **无文献** | C | D |

---

## A. 有文献 + 无社区 (Baseline)

### Synthesis
{_truncate(cells.get("with-lit/baseline", {}).get("synthesis", ""), 1500)}

### Hypotheses
{_truncate(cells.get("with-lit/baseline", {}).get("hypotheses", ""), 2000)}

Novelty: {cells.get("with-lit/baseline", {}).get("novelty", "N/A")}

---

## B. 有文献 + 有社区 (Treatment)

### Synthesis
{_truncate(cells.get("with-lit/treatment", {}).get("synthesis", ""), 1500)}

### Hypotheses
{_truncate(cells.get("with-lit/treatment", {}).get("hypotheses", ""), 2000)}

Novelty: {cells.get("with-lit/treatment", {}).get("novelty", "N/A")}

---

## C. 无文献 + 无社区 (Baseline)

### Synthesis
{_truncate(cells.get("no-lit/baseline", {}).get("synthesis", ""), 1500)}

### Hypotheses
{_truncate(cells.get("no-lit/baseline", {}).get("hypotheses", ""), 2000)}

Novelty: {cells.get("no-lit/baseline", {}).get("novelty", "N/A")}

---

## D. 无文献 + 有社区 (Treatment)

### Synthesis
{_truncate(cells.get("no-lit/treatment", {}).get("synthesis", ""), 1500)}

### Hypotheses
{_truncate(cells.get("no-lit/treatment", {}).get("hypotheses", ""), 2000)}

Novelty: {cells.get("no-lit/treatment", {}).get("novelty", "N/A")}

---

## 对比矩阵

| 维度 | A (文献+无社区) | B (文献+社区) | C (无文献+无社区) | D (无文献+社区) |
|------|----------------|--------------|------------------|----------------|
| Novelty | {cells.get("with-lit/baseline", {}).get("novelty", "?")} | {cells.get("with-lit/treatment", {}).get("novelty", "?")} | {cells.get("no-lit/baseline", {}).get("novelty", "?")} | {cells.get("no-lit/treatment", {}).get("novelty", "?")} |
| 创新性 | _ | _ | _ | _ |
| 可行性 | _ | _ | _ | _ |
| 清晰度 | _ | _ | _ | _ |

## 关键问题

1. **社区效果 (B vs A, D vs C)**: 接入社区后假设质量是否提升？
2. **文献效果 (A vs C, B vs D)**: 文献检索带来多少增益？
3. **交互效应 (B-A) vs (D-C)**: 社区知识在有/无文献时效果是否不同？
4. **社区是否替代文献 (D vs A)**: 无文献+社区 能否追上 有文献+无社区？
"""

    out = ARTIFACTS / "combined_report.md"
    out.write_text(report, encoding="utf-8")
    logger.info("Combined 2×2 report → %s", out)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _parse_novelty(path: Path) -> str:
    raw = _read_file(path)
    if not raw:
        return "N/A"
    try:
        return f"{json.loads(raw)['novelty_score']:.2f}"
    except (json.JSONDecodeError, KeyError):
        return "N/A"


def _summarize_snapshot(raw: str) -> str:
    if not raw:
        return ""
    try:
        snap = json.loads(raw)
    except json.JSONDecodeError:
        return "(解析失败)"
    lines = []
    for ch in snap.get("challenges", []):
        challenge = ch.get("challenge", {})
        lines.append(f"- **{challenge.get('title', '?')}** ({challenge.get('discipline_zh', '?')})")
        for ans in ch.get("model_answers", [])[:3]:
            s = f"{ans['score']:.1f}" if ans.get("score") is not None else "N/A"
            preview = ans.get("content", "")[:120]
            if len(ans.get("content", "")) > 120:
                preview += "..."
            lines.append(f"  - {ans.get('model_name', '?')} (score {s}): {preview}")
    return "\n".join(lines)


def _truncate(text: str, max_len: int) -> str:
    if not text:
        return "(empty)"
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n\n... (truncated)"


# ─────────────────────────────────────────────────────────────────
# Phase 2: Run experiments (Stage 9-22) — 暂不使用
# ─────────────────────────────────────────────────────────────────

def phase_run():
    """Continue from existing hypotheses through experiment + paper."""
    logger.info("Phase run not yet implemented for 2x2 design. Use --prepare first.")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EEG Emotion Recognition ablation: ARC-only vs ARC+Lab"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true",
                       help="生成假设 (Stage 7-8), 不跑实验")
    group.add_argument("--run", action="store_true",
                       help="从假设继续跑实验 (暂未实现)")
    group.add_argument("--all", action="store_true",
                       help="prepare + run 一键跑")
    parser.add_argument("--no-lit", action="store_true",
                        help="跳过文献检索, 纯 LLM 知识 + 社区")
    parser.add_argument("--with-lit", action="store_true",
                        help="包含文献检索 (默认)")
    parser.add_argument("--both", action="store_true",
                        help="同时跑 with-lit 和 no-lit 两种")
    args = parser.parse_args()

    if args.prepare:
        if args.both:
            # 跑两种
            prepare_with_lit()
            prepare_no_lit()
            _generate_combined_report()
        elif args.no_lit:
            prepare_no_lit()
        else:
            # 默认 with-lit
            prepare_with_lit()
    elif args.run:
        phase_run()
    elif args.all:
        prepare_with_lit()
        prepare_no_lit()
        _generate_combined_report()


if __name__ == "__main__":
    main()

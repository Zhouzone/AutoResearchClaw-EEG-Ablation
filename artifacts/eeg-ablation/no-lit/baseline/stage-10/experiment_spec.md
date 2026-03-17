# Experiment Specification

## Topic
Dynamic graph neural networks for EEG-based emotion recognition: comparing dynamic graph construction (adaptive adjacency, temporal sliding-window graphs) against static predefined graphs on DEAP and SEED datasets

## Project Structure
Multi-file experiment project with 4 file(s): `data_utils.py`, `main.py`, `metrics.py`, `models.py`

## Entry Point
`main.py` — executed directly via sandbox

## Outputs
- `main.py` emits metric lines in `name: value` format
- Primary metric key: `accuracy`

## Topic-Experiment Alignment
ALIGNED

## Constraints
- Time budget per run: 1800s
- Max iterations: 5
- Self-contained execution (no external data, no network)
- Validated: Code validation: 3 warning(s)

## Generated
2026-03-16T15:42:20+00:00

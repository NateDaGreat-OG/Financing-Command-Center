# AGENTS.md

## Purpose
This file gives coding agents a quick, repository-specific operating guide for `NateDaGreat-OG/Financing-Command-Center`.

## Repository layout
- `financial_command_center/`: Flask app with SQLAlchemy models, blueprints, templates, and static assets.
- `project/`: strategy, backtesting, services, and RL modules (`project/rl`).
- `migrations/`: Alembic migration environment and versions.
- `tests/`: regression coverage for the RL subsystem.

## Local setup
From repository root:

```bash
python -m pip install -r requirements.txt pytest torch numpy pandas
```

## Validation
Run the existing regression suite:

```bash
python -m pytest tests/test_rl_regression.py -v
```

## Change guidance for agents
- Keep edits scoped to the requested task; do not refactor unrelated modules.
- Do not commit API keys, secrets, or real credentials.
- Preserve current public API behavior unless the task explicitly requires changes.
- For symbol/file-path handling, prefer existing validation/safety helpers over new ad-hoc logic.
- If modifying RL code, maintain compatibility with the tested state/action assumptions used in `tests/test_rl_regression.py`.

## Notes
- `README.md` is minimal; rely on code structure and tests for implementation context.
- The root README explicitly warns to wire real Alpaca/Massive keys and replace placeholders before live use.

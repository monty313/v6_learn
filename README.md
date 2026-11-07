# v6_learn

## Project mission
Build and maintain the v6 learn project with reliable setup, execution, and delivery workflows.

## What this repository currently contains
Top-level structure:
- `.cursor` (dir)
- `.gitignore` (file)
- `HANDOFF_IRAC.md` (file)
- `INDICATOR_OBS_SETS_V6 (1).md` (file)
- `LESSONS_BOOK.md` (file)
- `pyproject.toml` (file)
- `README.md` (file)
- `runs` (dir)
- `spec` (dir)
- `tests` (dir)
- `v6rl` (dir)

Primary languages/tooling detected: Python (29), Markdown (14)

Likely key implementation files:
- `README.md`

## Quickstart (operator + LLM friendly)
Use this sequence first, then adapt to repo-specific details.

1. Clone and enter repo:
   ```bash
   git clone https://github.com/monty313/v6_learn.git
   cd v6_learn
   ```
2. Setup environment/dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```
3. Run locally:
   ```bash
   python main.py  # adjust to project entrypoint
   ```
4. Validate changes:
   ```bash
   pytest -q
   ```

## Operating intent (CEO-mode guide for future LLMs)
When continuing this project, optimize for business outcome first:

- Preserve or improve strategy quality, reliability, and reproducibility.
- Prefer small, testable, reversible changes over broad rewrites.
- Keep docs synchronized with code behavior.
- Add explicit assumptions, constraints, and success criteria in every major change.

## Delivery checklist
- [ ] Environment setup is deterministic.
- [ ] Primary run path is documented and works.
- [ ] Validation/test path is documented and works.
- [ ] Key risks and TODOs are tracked in the handoff file.

## Branch and release notes
- Default branch: `main`
- Keep PRs focused and attach proof-of-run or proof-of-test for behavior changes.

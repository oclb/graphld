---
name: graphld-setup
description: Use when installing, validating, repairing, or explaining the GraphLD development/runtime environment, including SuiteSparse, uv, data downloads, or O2-specific setup notes.
---

# GraphLD Setup

Use this skill for environment and dependency work, not for running analyses. Keep setup recipes out of `AGENTS.md`.

For LDGM or data-download setup, read `references/data-downloads.md` before recommending or running a download. Ask the user's workflow first and choose the smallest matching target.

Default local setup is:

```bash
uv venv
uv sync --extra dev
```

Validate the command surface with:

```bash
uv run graphld -h
uv run graphld reml -h
uv run estest --help
```

Full GraphLD needs SuiteSparse/CHOLMOD through `scikit-sparse`. Score-test-only work may only need:

```bash
uv run src/score_test/score_test.py --help
```

Read references as needed:

- `references/local-setup.md`: local development setup and dependency checks.
- `references/data-downloads.md`: workflow-driven data-download policy and Makefile targets.
- `references/o2-reference.md`: historical HMS O2 cluster setup notes.

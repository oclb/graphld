---
name: graphld-usage
description: Use when running, debugging, or documenting GraphLD analyses, CLI commands, APIs, file formats, or small workflow examples.
---

# GraphLD Usage

Use this skill for GraphLD analysis workflows. For environment setup, use `graphld-setup`.

Start with CLI help:

```bash
uv run graphld -h
uv run estest --help
```

Use `data/test/` for small fixtures only after checking that the files are present and non-empty.

Read the relevant reference:

- `references/graphreml.md`: graphREML, LDGM metadata, annotations, precision operations.
- `references/score-test.md`: `estest` workflows and HDF5 score-stat files.
- `references/other-cli.md`: BLUP, clump, simulate, and surrogates.
- `references/file-formats.md`: supported input formats and fixture conventions.

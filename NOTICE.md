# NOTICE

This repository contains a modified version of **jieba**
(<https://github.com/fxsjy/jieba>), Copyright (c) 2013 Sun Junyi,
released under the MIT License. The full MIT text is in `LICENSE`,
kept verbatim and unmodified.

## What was modified (2026-10-17)

Only two source files were touched. Both changes are marked inline with
`(optimization, see PATCH.md)`.

| File | Change |
|---|---|
| `jieba/__init__.py` | Added `Tokenizer._route()`, a single-pass replacement for the `get_DAG()` + `calc()` pair used by the accurate-mode cutters. `get_DAG()` and `calc()` themselves are kept unchanged for API compatibility. |
| `jieba/finalseg/__init__.py` | Rewrote `viterbi()` to use back-pointers instead of copying the full path at each step, and added an unrolled BMES fast path. |

Everything else — dictionaries, HMM/CRF models, `posseg`, `analyse`,
tests, `setup.py` — is upstream code as-is.

## License of the modifications

The modifications are released under the same MIT terms as the original.
See `LICENSE`.

## Upstream baseline

- Repository: <https://github.com/fxsjy/jieba>
- Version: `0.42.1`
- Commit: `67fa2e36e72f69d9134b8a1037b83fbb070b9775`
- SHA256 of the two files before modification: `original.sha256`

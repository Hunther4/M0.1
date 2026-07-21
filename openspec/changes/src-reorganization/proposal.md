# Proposal: Src Reorganization

## Intent

Unify overlapping directory responsibilities in `src/` and add missing `__init__.py` to `engine_v2/` so the package tree is consistent and easy to navigate.

## Changes

| Change | Type | Rationale |
|--------|------|-----------|
| `src/data/` — new package | Created | Unifies `src/dataset/prep.py` + `src/training/dataset.py` under one roof |
| `src/dataset/` — backward compat | Modified | Shim re-exporting from `src.data.prep` |
| `src/engine_v2/__init__.py` | Added | Missing package init with public API |
| `src/training/__init__.py` | Updated | Remove V1 legacy exports (loop, eval, setup, datasets) |

## Non-Goals

- No file deletions — all V1 modules remain for script backward compatibility
- No import changes in scripts (backward compat preserved)
- No rename of `engine_v2` → `engine` (would break 6 import sites)

## Rollback

Revert the four file changes. All existing imports remain intact because backward-compat shims were used throughout.

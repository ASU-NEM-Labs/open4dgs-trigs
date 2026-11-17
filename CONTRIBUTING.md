# Contributing to TriGS

Thank you for contributing. This repository is an open-source re-implementation
of TRiGS and follows a practical "small, reviewable changes" workflow.

## Development Setup

TriGS uses `uv` for dependency management.

```bash
git clone https://github.com/ASU-NEM-Labs/open4dgs-trigs.git
cd open4dgs-trigs
uv sync --preview-features extra-build-dependencies --group dev
```

The `extra-build-dependencies` preview feature is currently required for
source builds of `gsplat` and `pytorch3d`.

## Daily Workflow

1. Create a branch from `main`:
   ```bash
   git checkout -b feat/short-description
   ```
2. Make focused changes.
3. Run checks locally:
   ```bash
   uv run ruff check src/trigs/
   uv run mypy src/trigs/ --ignore-missing-imports
   uv run pytest
   ```
4. Update docs when behavior changes (`README.md`, CLI docstrings, comments).
5. Open a PR with a clear summary and test plan.

## Commit Message Standard (Required)

We use **Conventional Commits**.

### Format

```text
<type>(<optional-scope>): <imperative summary>

<body - explain why and impact, not just file list>

<optional footers>
```

### Allowed Types

- `feat` - new functionality
- `fix` - bug fix / correctness change
- `refactor` - code restructuring without behavior change
- `perf` - performance improvement
- `docs` - documentation-only changes
- `test` - tests only
- `chore` - maintenance/tooling changes
- `build` - dependency/build system changes
- `ci` - CI pipeline changes
- `revert` - revert a previous commit

### Rules

- Keep the subject line concise; prefer <= 72 characters.
- Use imperative mood ("add", "fix", "update"), not past tense.
- Capitalize only when needed; no trailing period in the subject.
- Explain **why** in the body, plus key behavioral impact.
- Wrap body lines at around 72 characters when possible.
- Reference issues in footers (`Refs: #123` / `Closes: #123`).
- For breaking API changes, add:
  - `BREAKING CHANGE: <what changed and migration guidance>`

### Good Examples

```text
feat(selfcap): add checkpoint evaluator and constant-view video export

Add a dedicated SelfCap evaluator CLI that reports PSNR/SSIM/LPIPS and
exports a constant-view mp4 from a held-out camera. This makes regression
tracking reproducible and avoids ad-hoc metric scripts.

Refs: #142
```

```text
fix(colmap): make end_frame semantics consistently inclusive

Pass end_frame + 1 to FreeTimeParser because it consumes an exclusive end
bound, while roma_combine uses inclusive bounds. This removes an off-by-one
frame mismatch between initialization and GT loading.
```

## Code Guidelines

- Use Python type hints for new code.
- Keep functions cohesive and avoid hidden side effects.
- Prefer clear names over short names unless mathematically standard.
- Add concise docstrings/comments for non-obvious logic.
- Preserve existing math/representation conventions (e.g., SE(3), quaternion
  ordering) unless the change explicitly migrates them.

## Testing Expectations

- Add tests for new behavior when practical.
- Update existing tests when behavior intentionally changes.
- For data-pipeline fixes, include at least one regression-style test.
- If a full test run is not feasible, document what was run and what was not.

## Documentation Expectations

When changing user-visible behavior, update at least one of:

- `README.md` (usage/flags/data format)
- relevant CLI module docstrings
- in-code comments where assumptions changed

For dataset-related changes, include canonical links when appropriate (for
example, SelfCap / LongVolcap).

## Pull Request Checklist

- [ ] Scope is focused and reviewable
- [ ] Commit messages follow Conventional Commits
- [ ] Lint/type/test checks were run (or limitations explained)
- [ ] Docs/comments were updated for behavior changes
- [ ] PR description includes a test plan

## License

By contributing, you agree that your contributions are licensed under
`AGPL-3.0` for this repository.

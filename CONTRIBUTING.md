# Contributing to MemJudge

Thanks for helping make memory benchmarks trustworthy.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,all]"
pytest && ruff check . && ruff format --check . && mypy
```

CI runs the same checks on Python 3.11–3.13. Pull requests must keep CI green.

## What we accept

- **Connectors** — see [docs/adding-a-connector.md](docs/adding-a-connector.md).
- **Datasets** — see [docs/adding-a-dataset.md](docs/adding-a-dataset.md); include the
  licence and a pinned source.
- **Errata** — corrections to gold answers with evidence from the source conversation.
- **Bug fixes and docs.** For larger changes, open an issue first.

## Standards

- Typed (mypy strict), documented, small modules. Avoid new dependencies unless needed;
  explain why in the PR.
- Tests for anything that affects scoring, signing, dataset adapters or connectors.
- Significant design decisions get a short ADR in `docs/decisions/`.
- Never commit secrets, test keys or personal data.
- Changes that alter scores (judge prompt, metrics, adapters) must say so in the PR and in
  `CHANGELOG.md`.

## Conflicts of interest

If you work for or with a memory-system vendor, say so in your PR. Vendors are welcome to
review and maintain connectors for their own systems (`label: vendor-maintained`).

## Releases

Semantic versioning; tagged releases published to PyPI by CI, which also builds and deploys
the hosted runner's container image for each release.

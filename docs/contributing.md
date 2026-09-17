# Contributing

Clone the repo and sync the environment with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/norlandrhagen/corduroy
cd corduroy
uv sync --all-groups
```

Run tests:

```bash
uv run pytest tests -n auto
```

Lint and format:

```bash
uv run prek run --all-files
```

Type check:

```bash
uv run ty check src/
```

Build and preview the docs:

```bash
uv sync --group docs
uv run mkdocs serve
```

CI runs the test suite on Python 3.12, 3.13 and 3.14; the lint job runs `prek`
and `ty`. Docs deploy to GitHub Pages from `main`.

## License

This code is licensed under the MIT License — see the
[LICENSE](https://github.com/norlandrhagen/corduroy/blob/main/LICENSE) file for
details.

# Repository Guidelines

## Project Structure & Module Organization

This repository covers Japanese medical-speech transcription, Whisper beam-search experiments, and pyannote diarization. Reusable code lives in `src/`, notably `beam_hook.py`, `run_whisper_bls.py`, and `dataset.py`. The primary analysis notebook is `263_full_paper.ipynb`; other root notebooks record experiments. Utilities such as `audio_explorer.py`, `mov_to_wav.py`, and `segment.py` prepare recordings. Generated results belong in `out_whisper/`, `pyannote_result/`, `beam_search_step/`, `_beamlog/`, or `_work/`. Place inputs under `data/`; the README also references `0604data/` and `0606data/`.

## Setup, Development, and Validation Commands

Create an isolated Python environment before installing the pinned stack:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Use `jupyter lab` for notebook workflows and `python audio_explorer.py --help` for that tool's CLI. Several scripts contain machine-specific paths; update their configuration before execution. There is no build step. Validate syntax with:

```bash
python -m compileall src *.py
```

Whisper and pyannote may download large models; record the model, device, and input when reporting results.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation. Use `snake_case` for functions, variables, and modules, `PascalCase` for classes/dataclasses, and uppercase constants. Add type hints and concise docstrings to reusable `src/` APIs. Restart notebook kernels and execute cells in order before committing. Avoid reformatting unrelated notebooks or generated files.

## Testing Guidelines

No automated framework or coverage threshold is configured; `test.py`, `test223.py`, and `test.ipynb` are exploratory. Run `compileall`, then the smallest relevant script or notebook cell on a short sample. Check expected SRT, JSONL, CSV, or plot output. New deterministic utilities should add pytest tests under `tests/` named `test_<module>.py`.

## Commit & Pull Request Guidelines

History contains only `first commit`, so no convention is established. Use imperative subjects such as `Add beam bias inspection logging`. Keep source changes separate from large artifacts. PRs should describe the change, list validation and model/data assumptions, link issues, and include representative plots or transcript excerpts for output changes.

## Security & Data Handling

Never commit Hugging Face tokens, credentials, patient identifiers, or raw sensitive recordings. Read secrets from environment variables such as `HF_TOKEN`. Review notebook outputs and metadata before committing, and keep large/private datasets outside Git.

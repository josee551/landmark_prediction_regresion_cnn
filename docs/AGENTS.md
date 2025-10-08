# Repository Guidelines

## Project Structure & Module Organization
The training entry point is `main.py`, supported by configuration files in `configs/`. Production code lives in `src/` with `src/data` handling dataset loaders and transforms, `src/models` defining the ResNet regression head, `src/training` orchestrating phases, and `src/evaluation` generating metrics and plots. Experiments and utilities sit at the repository root (`train_*`, `evaluate_*`, `visualize_*`), while outputs are saved to `checkpoints/`, `logs/`, and `evaluation_results/`. Keep raw inputs inside `data/dataset/` and annotations in `data/coordenadas/`; these folders are ignored by Git.

## Build, Test, and Development Commands
```
pip install -r requirements.txt        # project dependencies (PyTorch 2.4.1, etc.)
python main.py check                   # sanity-check data, paths, and GPU
python main.py train1                  # phase 1: head-only warm-up
python main.py train2                  # phase 2: full fine-tuning
python main.py train_geometric_complete# phase 4 complete-loss training
python main.py evaluate                # quick evaluation on latest checkpoint
python evaluate_complete.py            # full clinical metrics report
python main.py visualize_test_complete_loss  # export annotated predictions
python test_dataset.py                 # dataset smoke test + visualization
python test_gpu.py                     # ROCm/PyTorch device validation
```

## Coding Style & Naming Conventions
Write Python 3.12 code with 4-space indentation and follow PEP 8. Use type hints where feasible, snake_case for functions/modules, PascalCase for classes, and UPPER_SNAKE_CASE for constants. Configuration files are YAML (`configs/config.yaml`, `configs/config_geometric.yaml`)—mirror key names already in use. Keep plotting code deterministic by fixing random seeds when sampling examples and route new logs under `logs/phase_name/`.

## Testing Guidelines
Prefer fast feedback by running `python test_dataset.py` after data changes and `python main.py check` before long runs. End-to-end training stages should finish without exceptions, updating `checkpoints/` and `evaluation_results/`. When adding evaluation utilities, ensure they can operate headlessly (no GUI) and emit summaries to stdout plus CSV/PNG artifacts inside the relevant results folder. Attach sample outputs or metrics to PRs whenever new loss terms or transforms are introduced.

## Commit & Pull Request Guidelines
Use Conventional Commits (`feat:`, `fix:`, `docs:`, etc.) like the existing history. Keep subject lines ≤72 characters and elaborate in the body if behavior or data processing changes. Before opening a PR, run the smoke tests above and include: 1) purpose and scope, 2) commands used for validation, 3) key metrics or screenshots from `visualization_results/` when the visuals change, and 4) linked issues or roadmap references (e.g., `PHASE4_ROADMAP.md`). Ensure large datasets or proprietary assets remain untracked.

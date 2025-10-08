# 🔧 Scripts Directory

Standalone scripts for training, evaluation, visualization, and utilities.

## 📂 Directory Structure

```
scripts/
├── training/         # Training scripts for specific phases
├── evaluation/       # Evaluation and metrics scripts
├── visualization/    # Visualization and plotting scripts
└── utilities/        # Data exploration and testing utilities
```

## 🚀 Quick Access

### Training Scripts (`training/`)
- **[train_complete_simple.py](training/train_complete_simple.py)** - Phase 4: Complete Loss training (8.29px)
- **[train_symmetry_simple.py](training/train_symmetry_simple.py)** - Phase 3: Symmetry Loss training (8.91px)

### Evaluation Scripts (`evaluation/`)
- **[evaluate_complete.py](evaluation/evaluate_complete.py)** - Comprehensive evaluation with clinical metrics
- **[evaluate_symmetry.py](evaluation/evaluate_symmetry.py)** - Symmetry-specific evaluation

### Visualization Scripts (`visualization/`)
- **[visualize_complete_test.py](visualization/visualize_complete_test.py)** - Visualize test set with Complete Loss model
- **[visualize_symmetry.py](visualization/visualize_symmetry.py)** - Visualize symmetry constraints
- **[visualize_test_symmetry.py](visualization/visualize_test_symmetry.py)** - Test set visualization with symmetry

### Utility Scripts (`utilities/`)
- **[explore_data.py](utilities/explore_data.py)** - Dataset exploration and statistics
- **[test_dataset.py](utilities/test_dataset.py)** - Dataset loading and validation tests
- **[test_gpu.py](utilities/test_gpu.py)** - GPU/ROCm verification

## 💡 Usage Notes

**Recommended:** Use `main.py` CLI for most operations:
```bash
python main.py train_geometric_complete  # Uses scripts/training/train_complete_simple.py
python main.py evaluate                  # Uses scripts/evaluation/evaluate_complete.py
python main.py visualize_test            # Uses visualization scripts
```

**Direct execution:** Run scripts directly if needed:
```bash
python scripts/training/train_complete_simple.py
python scripts/evaluation/evaluate_complete.py
python scripts/utilities/explore_data.py
```

---

**Note:** These scripts are called by `main.py` CLI. Check `main.py` for the complete command list.

# 🏋️ Training Scripts

Phase-specific training scripts for the landmark prediction pipeline.

## Available Scripts

### Phase 3: Symmetry Loss
**[train_symmetry_simple.py](train_symmetry_simple.py)**
- Loss: Wing + 0.3×Symmetry
- Result: 8.91px error
- Base: Phase 2 checkpoint
- Training time: ~3 minutes

### Phase 4: Complete Loss (Current Best)
**[train_complete_simple.py](train_complete_simple.py)**
- Loss: Wing + 0.3×Symmetry + 0.2×Distance
- Result: 8.29px error ✅
- Base: Phase 3 checkpoint
- Training time: ~2.5 minutes

## Usage

**Via main.py (Recommended):**
```bash
python main.py train_geometric_symmetry   # Phase 3
python main.py train_geometric_complete   # Phase 4
```

**Direct execution:**
```bash
python scripts/training/train_symmetry_simple.py
python scripts/training/train_complete_simple.py
```

## Output

Checkpoints saved to `checkpoints/`:
- `geometric_symmetry.pt` - Phase 3 result (8.91px)
- `geometric_complete.pt` - Phase 4 result (8.29px)

---

**Note:** Phase 1 and Phase 2 training is handled by `src/training/` modules.

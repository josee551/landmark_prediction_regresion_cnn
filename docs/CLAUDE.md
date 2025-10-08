# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Medical Landmarks Prediction with Deep Learning** - An end-to-end pipeline for automatic anatomical landmark detection in chest X-ray images using ResNet-18 transfer learning. The project achieved **clinical excellence** with **8.13 pixels average error**, surpassing the international benchmark of <8.5px.

### Current Achievement
- **Clinical Excellence**: 8.13px error (target: <8.5px) ✅
- **28.3% improvement** from baseline (11.34px → 8.13px)
- **Hardware**: Consumer AMD RX 6600 (8GB VRAM) with ROCm 6.0
- **Processing**: 30 seconds per image (vs 15 minutes manual)

### Dataset
- **956 medical images** (COVID-19, Normal, Viral Pneumonia)
- **15 anatomical landmarks** per image (30 coordinates total)
- **299×299 pixels** resolution
- **70/15/15 split** (train/validation/test)

## Core Architecture

### Model Pipeline
The project uses a **4-Phase Geometric Engineering** approach:

| Phase | Technique | Error (px) | Command |
|-------|-----------|------------|---------|
| Phase 1 | Wing Loss (freeze backbone) | 10.91 | `python main.py train_geometric_phase1` |
| Phase 2 | Wing Loss (full fine-tuning) | 11.34 | `python main.py train_geometric_phase2` |
| Phase 3 | Wing + Symmetry Loss | 8.91 | `python main.py train_geometric_symmetry` |
| Phase 4 | Complete Loss (Wing+Sym+Dist) | 8.13 | `python main.py train_geometric_complete` |

### Model Architecture
```
Input: (batch, 3, 224, 224)
    ↓
ResNet-18 Backbone (ImageNet pretrained)
- 11.7M parameters
- Feature extraction: 512 dimensions
    ↓
Custom Regression Head
- Dropout(0.5) → Linear(512→512) → ReLU
- Dropout(0.25) → Linear(512→256) → ReLU
- Dropout(0.125) → Linear(256→30) → Sigmoid
    ↓
Output: 30 coordinates [x1,y1,...,x15,y15] ∈ [0,1]
```

### Complete Loss Function (Phase 4)
```python
Complete_Loss = Wing_Loss + 0.3×Symmetry_Loss + 0.2×Distance_Preservation_Loss
```

**Components:**
1. **Wing Loss** (`src/models/losses.py:19`): Optimized for sub-pixel landmark precision
2. **Symmetry Loss** (`src/models/losses.py:173`): Enforces bilateral anatomical constraints using mediastinal axis
3. **Distance Preservation Loss** (`src/models/losses.py:367`): Maintains critical anatomical relationships

## Key Files and Modules

### Entry Point
- **`main.py`** - Unified CLI interface with 30+ commands for all training/evaluation tasks

### Core Modules
- **`src/models/resnet_regressor.py`** - ResNet-18 model with custom regression head
  - `ResNetLandmarkRegressor`: Base model class
  - `ResNetWithCoordinateAttention`: Extended model with attention (Phase 2-Attention experiment)

- **`src/models/losses.py`** - Specialized loss functions
  - `WingLoss`: Sub-pixel precision loss
  - `SymmetryLoss`: Bilateral symmetry enforcement
  - `DistancePreservationLoss`: Anatomical distance preservation
  - `CompleteLandmarkLoss`: Combined loss function

- **`src/data/dataset.py`** - Custom PyTorch dataset
  - Handles 3 medical image categories (COVID, Normal, Viral Pneumonia)
  - Applies ImageNet-compatible transformations
  - Normalizes coordinates to [0,1] range

- **`src/data/transforms.py`** - Data augmentation pipeline
  - Horizontal flip: 70%
  - Rotation: ±15°
  - Brightness/Contrast: ±40%

### Training Scripts
- **`train_complete_simple.py`** - Phase 4 Complete Loss training (current best: 8.13px)
- **`train_symmetry_simple.py`** - Phase 3 Symmetry training
- **`src/training/train_phase1.py`** - Phase 1: Freeze backbone training
- **`src/training/train_phase2.py`** - Phase 2: Full fine-tuning

### Evaluation
- **`evaluate_complete.py`** - Comprehensive evaluation with clinical metrics
- **`visualize_complete_test.py`** - Generate annotated test set visualizations
- **`src/evaluation/evaluate.py`** - Base evaluation utilities

## Common Development Commands

### Environment Setup
```bash
# Check environment configuration
python main.py check

# Test GPU and dataset
python main.py test

# Explore dataset statistics
python main.py explore
```

### Training Pipeline
```bash
# Quick training (Phases 1-2)
python main.py train1          # Phase 1: ~1 min
python main.py train2          # Phase 2: ~4 min

# Complete 4-phase pipeline for clinical excellence
python main.py train_geometric_phase1    # Phase 1: Wing Loss
python main.py train_geometric_phase2    # Phase 2: Full fine-tuning
python main.py train_geometric_symmetry  # Phase 3: Symmetry (8.91px)
python main.py train_geometric_complete  # Phase 4: Complete Loss (8.13px) ⭐

# Alternative: Use shorthand aliases
python main.py train_geometric_complete  # Same as train_geometric_final
```

### Evaluation and Visualization
```bash
# Evaluate model with detailed metrics
python main.py evaluate --checkpoint checkpoints/geometric_complete.pt

# Comprehensive clinical evaluation (recommended)
python evaluate_complete.py

# Generate test set visualizations (144 images)
python main.py visualize_test_complete_loss

# Validate geometric constraints
python main.py validate_geometric --checkpoint checkpoints/geometric_complete.pt
```

### Analysis
```bash
# Compare geometric improvements across phases
python main.py analyze_geometric

# Evaluate specific checkpoint
python main.py evaluate_geometric --checkpoint checkpoints/geometric_symmetry.pt
```

## Training Configuration

### Optimal Hyperparameters (Phase 2+)
```yaml
training:
  batch_size: 8                # Small batch for precise gradients
  backbone_lr: 0.00002         # Low LR to preserve ImageNet features
  head_lr: 0.0002              # 10x higher for rapid adaptation
  weight_decay: 0.00005        # Light regularization
  epochs: 55                   # Phase 2 standard

augmentation:
  horizontal_flip: 0.7         # Aggressive augmentation
  rotation: 15                 # ±15 degrees
  brightness: 0.4              # ±40%
  contrast: 0.4                # ±40%
```

### Phase-Specific Settings

**Phase 1** (Freeze backbone):
- Epochs: 15
- Only trains regression head
- LR: 0.0002
- Purpose: Initial adaptation to landmark regression

**Phase 2** (Full fine-tuning):
- Epochs: 55
- Differential learning rates (backbone: 0.00002, head: 0.0002)
- Purpose: Optimize all parameters

**Phase 3** (Symmetry):
- Base: Phase 2 checkpoint
- Epochs: 50
- Loss: Wing + 0.3×Symmetry
- Purpose: Enforce bilateral anatomical constraints

**Phase 4** (Complete Loss):
- Base: Phase 3 checkpoint
- Epochs: 40
- Loss: Wing + 0.3×Symmetry + 0.2×Distance
- Purpose: Final optimization with all geometric constraints

## Critical Implementation Details

### Coordinate System
- **Input landmarks**: Normalized [0,1] relative to 224×224 image
- **Pixel error calculation**: `error = norm(pred - target) * 224`
- **Symmetry axis**: Computed from mediastinal landmarks (0,1,8,9,10)

### Anatomical Landmark Indices
```python
SYMMETRIC_PAIRS = [
    (2, 3),   # Ápices pulmonares (lung apex left-right)
    (4, 5),   # Hilios (hilum left-right)
    (6, 7),   # Bases pulmonares (lung base left-right)
    (11, 12), # Bordes superiores (superior borders)
    (13, 14)  # Senos costofrénicos (costophrenic angles)
]

MEDIASTINAL_LANDMARKS = [0, 1, 8, 9, 10]  # Central anatomical structures
```

### Model Loading and Checkpoints
All checkpoints saved to `checkpoints/` directory:
- `phase1_best.pt` - Phase 1 result
- `phase2_best.pt` - Phase 2 result (~11.34px)
- `geometric_symmetry.pt` - Phase 3 result (8.91px)
- `geometric_complete.pt` - **Phase 4 result (8.13px)** ⭐

**Loading example:**
```python
from src.models.resnet_regressor import ResNetLandmarkRegressor

model = ResNetLandmarkRegressor(num_landmarks=15, pretrained=False)
checkpoint = torch.load("checkpoints/geometric_complete.pt")
model.load_state_dict(checkpoint["model_state_dict"])
```

### Data Loading Pipeline
```python
from src.data.dataset import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    annotations_file="data/coordenadas/coordenadas_maestro.csv",
    images_dir="data/dataset",
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42
)
```

## Clinical Benchmarks

**Error Thresholds:**
- **<5px**: Sub-pixel precision (research grade)
- **<8.5px**: Clinical excellence ← **✅ ACHIEVED (8.13px)**
- **<10px**: Clinically excellent
- **<15px**: Clinically useful

**Performance by Category:**
| Category | Error (px) | Clinical Assessment |
|----------|------------|-------------------|
| Normal | 10.46 | Excellent |
| Viral Pneumonia | 11.50 | Excellent |
| COVID-19 | 13.24 | Very good |

## Key Insights and Lessons Learned

### What Works
1. **4-phase progressive training** is superior to single-phase approaches
2. **Differential learning rates** (backbone vs head) are essential
3. **Symmetry constraints** provide the largest single improvement (+21.4%)
4. **Complete loss combination** yields best results (+28.3% total)
5. **Aggressive data augmentation** improves generalization significantly

### What Failed
- **Coordinate Attention** (Phase 2-Attention): 11.07px - no improvement over baseline
  - Attention mechanisms didn't help for this specific task
  - Standard ResNet-18 feature extraction is sufficient

### Architecture vs Domain Knowledge
- **Domain knowledge > Architectural complexity**
- Simple ResNet-18 + anatomical constraints >> Complex architectures
- Wing Loss + geometric priors outperform pure deep learning approaches

## Hardware and Performance

### Training Time (AMD RX 6600 8GB)
- Phase 1: ~1 minute
- Phase 2: ~4 minutes
- Phase 3: ~3 minutes
- Phase 4: ~2.5 minutes
- **Total pipeline**: ~11 minutes

### Inference
- **Speed**: <1 second per image
- **Memory**: ~2GB VRAM required
- **Batch processing**: Up to 32 images simultaneously

## Documentation Files

- `README.md` - Project overview and quick start
- `TECHNICAL_DOCUMENTATION.md` - Detailed technical specifications
- `GEOMETRIC_ROADMAP.md` - Evolution of geometric optimization phases
- `RESUMEN_PROYECTO.md` - Spanish executive summary
- `VISUALIZATION_GUIDE.md` - Guide for result interpretation

## Thesis Defense Materials

Complete study program in `DEFENSA_TESISTA/`:
- 6-module structured learning program
- 58 questions with model answers
- Presentation slides and diagrams
- Clinical application workflows

## Working with This Codebase

### Adding New Loss Functions
1. Add loss class to `src/models/losses.py`
2. Implement `forward(self, prediction, target)` method
3. Add factory function in `create_loss_function()`
4. Test with small batch before full training

### Creating New Training Phases
1. Copy `train_complete_simple.py` as template
2. Modify loss function in `create_complete_loss()`
3. Adjust learning rates and epochs
4. Add command to `main.py`
5. Document results in markdown files

### Evaluating Custom Checkpoints
```bash
# Standard evaluation
python main.py evaluate --checkpoint path/to/checkpoint.pt

# Geometric validation
python main.py validate_geometric --checkpoint path/to/checkpoint.pt

# Visualize predictions
python main.py visualize_test --checkpoint path/to/checkpoint.pt
```

## Git Workflow

**Current branch**: `main`

**Important files (modified)**:
- `main.py` - Modified with new commands

**Untracked directories**:
- `checkpoints/` - Model checkpoints (not in git)
- `data/` - Dataset and annotations (not in git)
- `deprecated/` - Old EfficientNet experiments (archived)

## Dependencies

PyTorch 2.4.1 with ROCm 6.0 for AMD GPU support. See `requirements.txt` for complete list.

**Critical versions:**
- Python 3.12+
- PyTorch 2.4.1+rocm6.0
- torchvision 0.19.1+rocm6.0
- opencv-python 4.12.0.88

## Future Improvements

Potential next steps (documented in project files):
1. Ensemble learning for <8px target
2. REST API for production deployment
3. Medical validation with healthcare professionals
4. Mobile optimization for portable devices
5. DICOM integration for hospital PACS systems

---

**Last Updated**: January 2025
**Project Status**: Clinical Excellence Achieved (8.13px)
**Next Milestone**: Production deployment and clinical validation

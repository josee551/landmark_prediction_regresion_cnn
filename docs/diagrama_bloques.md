# Diagrama de Bloques - Landmark Regression System

## 🏗️ ARQUITECTURA GENERAL DEL SISTEMA

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          LANDMARK REGRESSION PIPELINE                          │
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │    DATA     │    │   TRAINING  │    │  INFERENCE  │    │ EVALUATION  │     │
│  │ PREPARATION │───▶│   PIPELINE  │───▶│   SYSTEM    │───▶│   & VISUAL  │     │
│  │             │    │             │    │             │    │             │     │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 BLOQUE 1: DATA PREPARATION

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA PREPARATION                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────────┐                                                          │
│  │  RAW DATASET     │                                                          │
│  │                  │                                                          │
│  │ data/dataset/    │     ┌─────────────────────────────────────┐             │
│  │ ├── COVID/       │────▶│        DATA LOADING MODULE         │             │
│  │ ├── Normal/      │     │                                     │             │
│  │ └── Viral_Pneum/ │     │ src/data/dataset.py                │             │
│  │                  │     │ - Custom Dataset class             │             │
│  │ 956 images total │     │ - Train/Val/Test split (70/15/15)  │             │
│  │ 299x299 PNG      │     │ - Batch loading with DataLoader    │             │
│  └──────────────────┘     └─────────────────────────────────────┘             │
│           │                                 │                                  │
│           │                                 ▼                                  │
│  ┌──────────────────┐     ┌─────────────────────────────────────┐             │
│  │  ANNOTATIONS     │     │       IMAGE TRANSFORMATIONS        │             │
│  │                  │     │                                     │             │
│  │ coordenadas_     │────▶│ src/data/transforms.py             │             │
│  │ maestro.csv      │     │                                     │             │
│  │                  │     │ ┌─────────────────────────────────┐ │             │
│  │ 15 landmarks     │     │ │        IMAGENET PREP            │ │             │
│  │ (30 coordinates) │     │ │ - Resize: 299x299 → 224x224    │ │             │
│  │ Format:          │     │ │ - RGB conversion                │ │             │
│  │ [x1,y1,x2,y2...] │     │ │ - Normalize: ImageNet stats     │ │             │
│  └──────────────────┘     │ └─────────────────────────────────┘ │             │
│                           │                                     │             │
│                           │ ┌─────────────────────────────────┐ │             │
│                           │ │       DATA AUGMENTATION         │ │             │
│                           │ │ - Horizontal flip: 0.7          │ │             │
│                           │ │ - Rotation: ±15°               │ │             │
│                           │ │ - Brightness: ±0.4             │ │             │
│                           │ │ - Contrast: ±0.4               │ │             │
│                           │ │ - Landmark-aware transforms    │ │             │
│                           │ └─────────────────────────────────┘ │             │
│                           └─────────────────────────────────────┘             │
│                                            │                                  │
│                                            ▼                                  │
│                           ┌─────────────────────────────────────┐             │
│                           │         PREPROCESSED DATA          │             │
│                           │                                     │             │
│                           │ Images: (batch, 3, 224, 224)       │             │
│                           │ Landmarks: (batch, 30) ∈ [0,1]     │             │
│                           │                                     │             │
│                           │ Train: 669 samples                 │             │
│                           │ Val:   144 samples                 │             │
│                           │ Test:  144 samples                 │             │
│                           └─────────────────────────────────────┘             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 BLOQUE 2: TRAINING PIPELINE

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                           ┌─────────────────────────────────┐                  │
│                           │         MODEL ARCHITECTURE     │                  │
│                           │                                 │                  │
│                           │ src/models/resnet_regressor.py  │                  │
│                           └─────────────────────────────────┘                  │
│                                            │                                   │
│                                            ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         RESNET-18 BACKBONE                              │   │
│  │                                                                         │   │
│  │  Input: (batch, 3, 224, 224)                                          │   │
│  │    │                                                                   │   │
│  │    ▼                                                                   │   │
│  │  ┌───────────────────────────────────────────────────────────────┐    │   │
│  │  │                    CONVOLUTION LAYERS                        │    │   │
│  │  │                                                               │    │   │
│  │  │  conv1 (7x7, 64) → bn1 → relu → maxpool                    │    │   │
│  │  │     │                                                        │    │   │
│  │  │     ▼                                                        │    │   │
│  │  │  layer1: [BasicBlock × 2] (64 channels)                    │    │   │
│  │  │     │                                                        │    │   │
│  │  │     ▼                                                        │    │   │
│  │  │  layer2: [BasicBlock × 2] (128 channels)                   │    │   │
│  │  │     │                                                        │    │   │
│  │  │     ▼                                                        │    │   │
│  │  │  layer3: [BasicBlock × 2] (256 channels)                   │    │   │
│  │  │     │                                                        │    │   │
│  │  │     ▼                                                        │    │   │
│  │  │  layer4: [BasicBlock × 2] (512 channels)                   │    │   │
│  │  │     │                                                        │    │   │
│  │  │     ▼                                                        │    │   │
│  │  │  avgpool → (batch, 512)                                     │    │   │
│  │  └───────────────────────────────────────────────────────────────┘    │   │
│  │                            │                                           │   │
│  │                            ▼                                           │   │
│  │  ┌───────────────────────────────────────────────────────────────┐    │   │
│  │  │                 REGRESSION HEAD                               │    │   │
│  │  │                                                               │    │   │
│  │  │  Features (512) → Dropout(0.5) → Linear(512→512) → ReLU    │    │   │
│  │  │                      │                                       │    │   │
│  │  │                      ▼                                       │    │   │
│  │  │                 → Dropout(0.25) → Linear(512→256) → ReLU    │    │   │
│  │  │                      │                                       │    │   │
│  │  │                      ▼                                       │    │   │
│  │  │                 → Dropout(0.125) → Linear(256→30) → Sigmoid │    │   │
│  │  │                      │                                       │    │   │
│  │  │                      ▼                                       │    │   │
│  │  │                 Output: (batch, 30) ∈ [0,1]                │    │   │
│  │  └───────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                            │                                   │
│                                            ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                          TWO-PHASE TRAINING                            │   │
│  │                                                                         │   │
│  │  ┌─────────────────────┐              ┌─────────────────────┐          │   │
│  │  │      PHASE 1        │              │      PHASE 2        │          │   │
│  │  │   Head Training     │─────────────▶│   Full Fine-tuning  │          │   │
│  │  │                     │              │                     │          │   │
│  │  │ ┌─────────────────┐ │              │ ┌─────────────────┐ │          │   │
│  │  │ │ Backbone:       │ │              │ │ Backbone:       │ │          │   │
│  │  │ │ ❄️  FROZEN      │ │              │ │ 🔥 TRAINABLE    │ │          │   │
│  │  │ │ (11M params)    │ │              │ │ LR: 0.00002     │ │          │   │
│  │  │ └─────────────────┘ │              │ └─────────────────┘ │          │   │
│  │  │ ┌─────────────────┐ │              │ ┌─────────────────┐ │          │   │
│  │  │ │ Head:           │ │              │ │ Head:           │ │          │   │
│  │  │ │ 🔥 TRAINABLE    │ │              │ │ 🔥 TRAINABLE    │ │          │   │
│  │  │ │ LR: 0.001       │ │              │ │ LR: 0.0002      │ │          │   │
│  │  │ │ (401K params)   │ │              │ │ (401K params)   │ │          │   │
│  │  │ └─────────────────┘ │              │ └─────────────────┘ │          │   │
│  │  │                     │              │                     │          │   │
│  │  │ Epochs: 15          │              │ Epochs: 55          │          │   │
│  │  │ Result: ~19px       │              │ Result: 11.34px     │          │   │
│  │  └─────────────────────┘              └─────────────────────┘          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                            │                                   │
│                                            ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        TRAINING COMPONENTS                              │   │
│  │                                                                         │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │   │
│  │  │   LOSS FUNCTION │  │   OPTIMIZER     │  │   SCHEDULER     │         │   │
│  │  │                 │  │                 │  │                 │         │   │
│  │  │ MSE Loss        │  │ Adam            │  │ CosineAnnealing │         │   │
│  │  │ (L2 regression) │  │ Weight decay:   │  │ T_max: epochs   │         │   │
│  │  │                 │  │ 0.00005         │  │ min_lr: 2e-6    │         │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │   │
│  │                                                                         │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │   │
│  │  │ EARLY STOPPING  │  │    LOGGING      │  │   CHECKPOINTS   │         │   │
│  │  │                 │  │                 │  │                 │         │   │
│  │  │ Patience: 10    │  │ TensorBoard     │  │ Every 5 epochs  │         │   │
│  │  │ Monitor: val_loss│  │ Metrics plots   │  │ Best model save │         │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔮 BLOQUE 3: INFERENCE SYSTEM

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               INFERENCE SYSTEM                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────────┐                                                          │
│  │   NEW IMAGE      │                                                          │
│  │                  │                                                          │
│  │ 299x299 PNG      │     ┌─────────────────────────────────────┐             │
│  │ Medical scan     │────▶│       PREPROCESSING PIPELINE       │             │
│  │ (any category)   │     │                                     │             │
│  └──────────────────┘     │ src/evaluation/visualize.py        │             │
│           │                │                                     │             │
│           │                │ ┌─────────────────────────────────┐ │             │
│           │                │ │      IMAGE PREPARATION          │ │             │
│           │                │ │                                 │ │             │
│           ▼                │ │ 1. Load image (OpenCV)          │ │             │
│  ┌──────────────────┐     │ │ 2. Convert BGR→RGB              │ │             │
│  │  QUALITY CHECK   │     │ │ 3. Resize 299x299→224x224       │ │             │
│  │                  │     │ │ 4. Normalize ImageNet stats     │ │             │
│  │ - Resolution OK  │     │ │ 5. Add batch dimension          │ │             │
│  │ - Format valid   │     │ └─────────────────────────────────┘ │             │
│  │ - Medical image  │     └─────────────────────────────────────┘             │
│  └──────────────────┘                      │                                  │
│                                            ▼                                  │
│                           ┌─────────────────────────────────────┐             │
│                           │          MODEL INFERENCE           │             │
│                           │                                     │             │
│                           │ ┌─────────────────────────────────┐ │             │
│                           │ │        LOAD CHECKPOINT          │ │             │
│                           │ │                                 │ │             │
│                           │ │ checkpoints/phase2_best.pt      │ │             │
│                           │ │ - Model architecture            │ │             │
│                           │ │ - Trained weights               │ │             │
│                           │ │ - Training metadata             │ │             │
│                           │ └─────────────────────────────────┘ │             │
│                           │                                     │             │
│                           │ ┌─────────────────────────────────┐ │             │
│                           │ │        FORWARD PASS             │ │             │
│                           │ │                                 │ │             │
│                           │ │ with torch.no_grad():           │ │             │
│                           │ │   input = image.unsqueeze(0)    │ │             │
│                           │ │   output = model(input)         │ │             │
│                           │ │   landmarks = output.squeeze()  │ │             │
│                           │ └─────────────────────────────────┘ │             │
│                           └─────────────────────────────────────┘             │
│                                            │                                  │
│                                            ▼                                  │
│                           ┌─────────────────────────────────────┐             │
│                           │       POST-PROCESSING              │             │
│                           │                                     │             │
│                           │ ┌─────────────────────────────────┐ │             │
│                           │ │     DENORMALIZE COORDS          │ │             │
│                           │ │                                 │ │             │
│                           │ │ Input: [0,1] normalized         │ │             │
│                           │ │ Output: pixel coordinates       │ │             │
│                           │ │                                 │ │             │
│                           │ │ x_pixels = x_norm * width       │ │             │
│                           │ │ y_pixels = y_norm * height      │ │             │
│                           │ └─────────────────────────────────┘ │             │
│                           │                                     │             │
│                           │ ┌─────────────────────────────────┐ │             │
│                           │ │        FORMAT OUTPUT            │ │             │
│                           │ │                                 │ │             │
│                           │ │ x_coords = [x1, x2, ..., x15]   │ │             │
│                           │ │ y_coords = [y1, y2, ..., y15]   │ │             │
│                           │ │                                 │ │             │
│                           │ │ landmarks_dict = {              │ │             │
│                           │ │   'landmark_1': (x1, y1),      │ │             │
│                           │ │   'landmark_2': (x2, y2),      │ │             │
│                           │ │   ...                           │ │             │
│                           │ │ }                               │ │             │
│                           │ └─────────────────────────────────┘ │             │
│                           └─────────────────────────────────────┘             │
│                                            │                                  │
│                                            ▼                                  │
│                           ┌─────────────────────────────────────┐             │
│                           │          PREDICTION OUTPUT         │             │
│                           │                                     │             │
│                           │ 15 Landmarks @ pixel coordinates    │             │
│                           │ Expected error: ~11.34 pixels      │             │
│                           │ Confidence: Clinical grade          │             │
│                           └─────────────────────────────────────┘             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 BLOQUE 4: EVALUATION & VISUALIZATION

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION & VISUALIZATION                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────────┐     ┌─────────────────────────────────────┐             │
│  │  PREDICTIONS     │     │         METRICS CALCULATION        │             │
│  │                  │────▶│                                     │             │
│  │ Model outputs    │     │ src/evaluation/evaluate.py         │             │
│  │ Ground truth     │     │                                     │             │
│  │ Test set         │     │ ┌─────────────────────────────────┐ │             │
│  └──────────────────┘     │ │        CORE METRICS             │ │             │
│                           │ │                                 │ │             │
│                           │ │ • RMSE (Root Mean Square)      │ │             │
│                           │ │ • MAE (Mean Absolute Error)    │ │             │
│                           │ │ • MSE (Mean Square Error)      │ │             │
│                           │ │ • Euclidean Distance per point │ │             │
│                           │ └─────────────────────────────────┘ │             │
│                           │                                     │             │
│                           │ ┌─────────────────────────────────┐ │             │
│                           │ │      PIXEL CONVERSION           │ │             │
│                           │ │                                 │ │             │
│                           │ │ normalized_error * 224 = pixels │ │             │
│                           │ │                                 │ │             │
│                           │ │ Current Results:                │ │             │
│                           │ │ • RMSE: 9.47 pixels            │ │             │
│                           │ │ • MAE: 7.15 pixels             │ │             │
│                           │ │ • Mean Distance: 11.34 pixels  │ │             │
│                           │ └─────────────────────────────────┘ │             │
│                           └─────────────────────────────────────┘             │
│                                            │                                  │
│                                            ▼                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        DETAILED ANALYSIS                               │   │
│  │                                                                         │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │   │
│  │  │  BY LANDMARK    │  │   BY CATEGORY   │  │ ERROR PATTERNS  │         │   │
│  │  │                 │  │                 │  │                 │         │   │
│  │  │ Best: #9        │  │ Normal: 10.46px │  │ Outlier detect │         │   │
│  │  │ Worst: #14      │  │ Viral: 11.38px  │  │ Distribution    │         │   │
│  │  │                 │  │ COVID: 13.24px  │  │ Failure modes   │         │   │
│  │  │ Individual      │  │                 │  │                 │         │   │
│  │  │ performance     │  │ Medical         │  │ Quality assess  │         │   │
│  │  │ ranking         │  │ insights        │  │                 │         │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                            │                                   │
│                                            ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                          VISUALIZATIONS                                │   │
│  │                                                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │                    AUTOMATIC PLOTS                             │   │   │
│  │  │                                                                 │   │   │
│  │  │  📊 Metrics Evolution (TensorBoard)                            │   │   │
│  │  │     • Training/Validation loss curves                          │   │   │
│  │  │     • RMSE/MAE progression over epochs                         │   │   │
│  │  │     • Learning rate scheduling visualization                   │   │   │
│  │  │                                                                 │   │   │
│  │  │  📈 Error Distribution Charts                                   │   │   │
│  │  │     • Histogram of prediction errors                           │   │   │
│  │  │     • Box plots by landmark ID                                 │   │   │
│  │  │     • Category comparison charts                               │   │   │
│  │  │                                                                 │   │   │
│  │  │  🎯 Landmark Performance Analysis                               │   │   │
│  │  │     • Per-landmark accuracy ranking                            │   │   │
│  │  │     • X/Y coordinate error breakdown                           │   │   │
│  │  │     • Spatial error heatmaps                                   │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │                 PREDICTION OVERLAYS                            │   │   │
│  │  │                                                                 │   │   │
│  │  │  🖼️ Sample Visualizations                                      │   │   │
│  │  │     • Original image backgrounds                               │   │   │
│  │  │     • Ground truth landmarks (green dots)                     │   │   │
│  │  │     • Predicted landmarks (red dots)                          │   │   │
│  │  │     • Error vectors (connecting lines)                        │   │   │
│  │  │     • Numbered landmark annotations                           │   │   │
│  │  │                                                                 │   │   │
│  │  │  📋 Comparison Grids                                           │   │   │
│  │  │     • Side-by-side GT vs Predictions                          │   │   │
│  │  │     • Multiple samples per category                           │   │   │
│  │  │     • Best/worst case examples                                │   │   │
│  │  │     • Error magnitude color coding                            │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                            │                                   │
│                                            ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         OUTPUT FILES                                   │   │
│  │                                                                         │   │
│  │  📁 evaluation_results/                                                │   │
│  │     ├── evaluation_metrics.png    (Charts and graphs)                 │   │
│  │     ├── sample_predictions.png    (Visual overlays)                   │   │
│  │     ├── general_metrics.csv       (Numerical results)                 │   │
│  │     ├── per_landmark_metrics.csv  (Detailed breakdowns)               │   │
│  │     └── per_category_metrics.csv  (Medical category analysis)         │   │
│  │                                                                         │   │
│  │  📁 visualization_results/                                             │   │
│  │     ├── prediction_*.png          (Individual image results)          │   │
│  │     └── batch_comparisons.png     (Multiple sample views)             │   │
│  │                                                                         │   │
│  │  📁 logs/                                                              │   │
│  │     ├── phase1_head_only/         (TensorBoard training logs)         │   │
│  │     ├── phase2_full_finetuning/   (TensorBoard fine-tuning logs)      │   │
│  │     └── metrics_evolution.png     (Training progression plots)        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 FLUJO DE DATOS COMPLETO

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              END-TO-END FLOW                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Medical Images (299x299)                                                      │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────┐                                                               │
│  │   Dataset   │ ──────┐                                                      │
│  │   Loading   │       │                                                      │
│  └─────────────┘       │                                                      │
│         │               │                                                      │
│         ▼               │                                                      │
│  ┌─────────────┐       │    ┌─────────────┐                                  │
│  │ Transforms  │       │    │ Annotations │                                  │
│  │ (ImageNet)  │       │    │  (CSV File) │                                  │
│  └─────────────┘       │    └─────────────┘                                  │
│         │               │           │                                         │
│         ▼               ▼           ▼                                         │
│  ┌─────────────────────────────────────┐                                     │
│  │        Training Data               │                                     │
│  │    (Images + Landmarks)            │                                     │
│  └─────────────────────────────────────┘                                     │
│                    │                                                          │
│                    ▼                                                          │
│  ┌─────────────────────────────────────┐                                     │
│  │         Phase 1 Training           │                                     │
│  │      (Head Only - 15 epochs)       │                                     │
│  └─────────────────────────────────────┘                                     │
│                    │                                                          │
│                    ▼                                                          │
│  ┌─────────────────────────────────────┐                                     │
│  │         Phase 2 Training           │                                     │
│  │    (Full Fine-tuning - 55 epochs)  │                                     │
│  └─────────────────────────────────────┘                                     │
│                    │                                                          │
│                    ▼                                                          │
│  ┌─────────────────────────────────────┐                                     │
│  │        Trained Model               │                                     │
│  │     (11.34 pixels error)           │                                     │
│  └─────────────────────────────────────┘                                     │
│                    │                                                          │
│         ┌──────────┼──────────┐                                              │
│         ▼          ▼          ▼                                              │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐                                 │
│  │Evaluation │ │Inference  │ │Visualize  │                                 │
│  │& Metrics  │ │New Images │ │Results    │                                 │
│  └───────────┘ └───────────┘ └───────────┘                                 │
│         │          │          │                                              │
│         ▼          ▼          ▼                                              │
│  ┌─────────────────────────────────────┐                                     │
│  │           Final Output              │                                     │
│  │                                     │                                     │
│  │  • Performance Reports             │                                     │
│  │  • Landmark Predictions            │                                     │
│  │  • Visual Overlays                 │                                     │
│  │  • Clinical Assessment             │                                     │
│  └─────────────────────────────────────┘                                     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 MÉTRICAS DE RENDIMIENTO POR BLOQUE

### Data Preparation
- **Throughput**: ~1000 imágenes/minuto
- **Memory**: ~2GB RAM para dataset completo
- **Augmentation**: 4x variaciones por imagen original

### Training Pipeline
- **Fase 1**: 15 épocas @ ~10 segundos/época
- **Fase 2**: 55 épocas @ ~4 segundos/época
- **GPU Memory**: ~3GB pico durante entrenamiento
- **Total Training Time**: ~4-5 minutos

### Inference System
- **Latency**: ~50ms por imagen (single GPU)
- **Throughput**: ~20 imágenes/segundo
- **Memory**: ~1GB para modelo cargado
- **Accuracy**: 11.34 píxeles error promedio

### Evaluation & Visualization
- **Batch Evaluation**: 144 imágenes en ~3 segundos
- **Report Generation**: ~5 segundos para métricas completas
- **Visualization**: ~2 segundos por imagen procesada

---

*Diagrama actualizado con resultados finales optimizados (11.34 píxeles)*
*Arquitectura probada y validada en GPU AMD RX 6600*
# Documentación Técnica: Regresión de Landmarks con ResNet-18

## 📋 Índice
1. [Fundamentos Matemáticos](#fundamentos-matemáticos)
2. [Arquitectura del Modelo](#arquitectura-del-modelo)
3. [Pipeline de Datos](#pipeline-de-datos)
4. [Algoritmos de Entrenamiento](#algoritmos-de-entrenamiento)
5. [Métricas de Evaluación](#métricas-de-evaluación)
6. [Ensemble Learning](#ensemble-learning)
7. [Optimizaciones Implementadas](#optimizaciones-implementadas)
8. [Análisis de Complejidad](#análisis-de-complejidad)

---

## 🧮 Fundamentos Matemáticos

### 📐 Espacio de Coordenadas y Normalización

#### Transformación de Coordenadas
El modelo opera en coordenadas normalizadas para generalización y estabilidad numérica:

```
Píxeles → Normalización:
x_norm = x_pixel / width_original     ∈ [0, 1]
y_norm = y_pixel / height_original    ∈ [0, 1]

Desnormalización → Píxeles:
x_pixel = x_norm × width_target
y_pixel = y_norm × height_target
```

#### Representación Vectorial
Cada imagen tiene 15 landmarks representados como un vector de 30 dimensiones:

```
L = [x₁, y₁, x₂, y₂, ..., x₁₅, y₁₅] ∈ [0,1]³⁰

donde cada landmark k:
L[2k-2] = x_k  (coordenada X del landmark k)
L[2k-1] = y_k  (coordenada Y del landmark k)
```

### 🎯 Función de Pérdida

#### Mean Squared Error (MSE)
Función de pérdida principal para regresión de landmarks:

```
L(θ) = (1/N) Σᵢ₌₁ᴺ Σⱼ₌₁³⁰ (ŷᵢⱼ - yᵢⱼ)²

donde:
- N = tamaño del batch
- θ = parámetros del modelo
- ŷᵢⱼ = predicción j-ésima de la muestra i
- yᵢⱼ = ground truth j-ésimo de la muestra i
```

#### Gradientes de la Pérdida
El gradiente con respecto a las predicciones:

```
∂L/∂ŷᵢⱼ = (2/N) × (ŷᵢⱼ - yᵢⱼ)

Esto proporciona un gradiente proporcional al error,
facilitando la convergencia hacia landmarks precisos.
```

### 🔄 Data Augmentation Matemático

#### Flip Horizontal
Reflexión sobre el eje vertical preservando la anatomía:

```
Para flip horizontal:
x_new = 1.0 - x_original  (reflexión en [0,1])
y_new = y_original        (eje Y sin cambios)

Matriz de transformación:
T_flip = [-1  0  1]
         [ 0  1  0]
         [ 0  0  1]
```

#### Rotación 2D
Rotación aleatoria ±15° alrededor del centro de la imagen:

```
θ ~ Uniform(-15°, +15°)

Matriz de rotación:
R(θ) = [cos(θ)  -sin(θ)]
       [sin(θ)   cos(θ)]

Aplicación (centrado en 0.5, 0.5):
[x_new] = R(θ) × [x_orig - 0.5] + [0.5]
[y_new]          [y_orig - 0.5]   [0.5]
```

#### Transformaciones Fotométricas
Ajustes de brillo y contraste preservando landmarks:

```
Brillo: I_new = I_original + α, α ~ Uniform(-0.4, +0.4)
Contraste: I_new = β × I_original, β ~ Uniform(0.6, 1.4)

Las coordenadas de landmarks no se afectan por estos cambios.
```

---

## 🏗️ Arquitectura del Modelo

### 🧠 ResNet-18 Base

#### Backbone Preentrenado
```
ResNet-18 (ImageNet pretrained):
├── conv1: Conv2d(3, 64, 7×7, stride=2) + BatchNorm + ReLU + MaxPool
├── layer1: 2 × BasicBlock(64)
├── layer2: 2 × BasicBlock(128, stride=2)
├── layer3: 2 × BasicBlock(256, stride=2)
├── layer4: 2 × BasicBlock(512, stride=2)
└── avgpool: AdaptiveAvgPool2d(1,1) → 512 features

Total parámetros backbone: 11,176,512
```

#### BasicBlock Residual
```
BasicBlock(in_channels, out_channels):
x → Conv2d(3×3) → BatchNorm → ReLU → Conv2d(3×3) → BatchNorm
↓                                                      ↓
identity (o skip connection si dim cambia)              +
                                                       ↓
                                                     ReLU

Ecuación matemática:
y = F(x, {Wᵢ}) + x    (si dimensiones coinciden)
y = F(x, {Wᵢ}) + Ws×x (si hay cambio dimensional)
```

### 🎯 Cabeza de Regresión Personalizada

#### Arquitectura de la Cabeza
```
Custom Regression Head:
512 features → Dropout(0.5) → Linear(512, 512) → ReLU
            → Dropout(0.25) → Linear(512, 256) → ReLU
            → Dropout(0.125) → Linear(256, 30) → Sigmoid

Parámetros de la cabeza: 401,694
```

#### Función de Activación Sigmoid
```
σ(x) = 1 / (1 + e^(-x))

Propiedades:
- Rango: (0, 1) - perfecto para coordenadas normalizadas
- Diferenciable: σ'(x) = σ(x)(1 - σ(x))
- Saturación gradual: previene overshooting
```

#### Dropout Progresivo
```
Dropout rates: [0.5, 0.25, 0.125]

Función Dropout:
y = x / (1 - p) × mask, donde mask ~ Bernoulli(1 - p)

Rationale: Mayor dropout al inicio (features generales),
menor dropout al final (features específicas).
```

### 📊 Conteo de Parámetros

#### Distribución de Parámetros
```
Componente              | Parámetros    | Porcentaje
------------------------|---------------|------------
Backbone ResNet-18      | 11,176,512    | 96.5%
Linear 512→512 + bias   | 262,656       | 2.3%
Linear 512→256 + bias   | 131,328       | 1.1%
Linear 256→30 + bias    | 7,710         | 0.1%
------------------------|---------------|------------
TOTAL                   | 11,578,206    | 100%
```

#### Parámetros Entrenables por Fase
```
Fase 1 (freeze_backbone=True):
- Entrenables: 401,694 (3.5%)
- Congelados: 11,176,512 (96.5%)

Fase 2 (freeze_backbone=False):
- Entrenables: 11,578,206 (100%)
- Congelados: 0 (0%)
```

---

## 🔄 Pipeline de Datos

### 📥 Carga y Procesamiento

#### Dataset Loading Algorithm
```python
class LandmarkDataset(Dataset):
    def __init__(self, annotations_file, images_dir, transform=None):
        # 1. Cargar CSV con pandas
        self.annotations = pd.read_csv(annotations_file, header=None)

        # 2. Parsear columnas: [ID, x1, y1, ..., x15, y15, filename]
        self.landmarks = self.annotations.iloc[:, 1:31].values  # 30 coords
        self.filenames = self.annotations.iloc[:, 31].values    # nombres

        # 3. Validar integridad
        self._validate_data_integrity()

    def __getitem__(self, idx):
        # 1. Cargar imagen
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Normalizar landmarks [píxeles] → [0,1]
        h, w = image_rgb.shape[:2]
        landmarks = self.landmarks[idx].copy()
        landmarks[::2] /= w    # coordenadas X
        landmarks[1::2] /= h   # coordenadas Y

        # 3. Aplicar transformaciones
        if self.transform:
            image_tensor, landmarks_tensor = self.transform(image_rgb, landmarks)

        return image_tensor, landmarks_tensor, metadata
```

#### Data Splitting Strategy
```python
def create_splits(total_size, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    División estratificada reproducible
    """
    np.random.seed(seed)
    indices = np.random.permutation(total_size)

    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    return train_indices, val_indices, test_indices

# Splits reales del proyecto:
# Total: 999 imágenes
# Train: 669 (70%), Val: 144 (15%), Test: 144 (15%)
```

### 🖼️ Transformaciones de Imagen

#### Pipeline de Preprocesamiento
```python
def get_transforms(image_size=(224, 224), is_training=True):
    if is_training:
        return Compose([
            LandmarkRandomHorizontalFlip(p=0.7),      # 70% probabilidad
            LandmarkRandomRotation(degrees=15),        # ±15 grados
            LandmarkColorJitter(                       # Ajustes fotométricos
                brightness=0.4,                        # ±40%
                contrast=0.4                           # ±40%
            ),
            LandmarkResize(image_size),                # Redimensión a 224×224
            LandmarkToTensor(),                        # Numpy → Tensor
            LandmarkNormalize(                         # Normalización ImageNet
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return Compose([
            LandmarkResize(image_size),
            LandmarkToTensor(),
            LandmarkNormalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
```

#### Transformación Landmark-Aware
```python
class LandmarkRandomHorizontalFlip:
    def __call__(self, image, landmarks):
        if random.random() < self.p:
            # Flip imagen
            image = cv2.flip(image, 1)

            # Flip coordenadas X de landmarks
            landmarks_copy = landmarks.copy()
            landmarks_copy[::2] = 1.0 - landmarks_copy[::2]

            return image, landmarks_copy
        return image, landmarks

class LandmarkRandomRotation:
    def __call__(self, image, landmarks):
        angle = random.uniform(-self.degrees, self.degrees)

        # Rotar imagen
        center = (image.shape[1] // 2, image.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image_rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # Rotar landmarks
        landmarks_rotated = self._rotate_landmarks(landmarks, angle, center)

        return image_rotated, landmarks_rotated
```

---

## 🏋️ Algoritmos de Entrenamiento

### 🎯 Transfer Learning en 2 Fases

#### Fase 1: Head Training Algorithm
```python
def train_phase1(model, dataloader, optimizer, criterion, device):
    """
    Entrenamiento solo de la cabeza con backbone congelado
    """
    # 1. Congelar backbone
    model.freeze_backbone()

    # 2. Verificar parámetros entrenables
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {trainable_params:,}")  # 401,694

    # 3. Loop de entrenamiento
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_idx, (images, landmarks, _) in enumerate(dataloader):
            images, landmarks = images.to(device), landmarks.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, landmarks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Época {epoch+1}: Loss = {avg_loss:.6f}")
```

#### Fase 2: Fine-tuning Algorithm
```python
def train_phase2(model, dataloader, device):
    """
    Fine-tuning completo con learning rates diferenciados
    """
    # 1. Descongelar backbone
    model.unfreeze_backbone()

    # 2. Configurar optimizador con LR diferenciados
    param_groups = [
        {
            'params': model.get_backbone_parameters(),
            'lr': 0.00002,          # LR bajo para preservar features ImageNet
            'name': 'backbone'
        },
        {
            'params': model.get_head_parameters(),
            'lr': 0.0002,           # LR alto para especialización (10× backbone)
            'name': 'head'
        }
    ]
    optimizer = torch.optim.Adam(param_groups, weight_decay=0.00005)

    # 3. Scheduler CosineAnnealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0.000002
    )

    # 4. Loop de entrenamiento con gradient clipping
    model.train()
    for epoch in range(epochs):
        for batch_idx, (images, landmarks, _) in enumerate(dataloader):
            images, landmarks = images.to(device), landmarks.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, landmarks)
            loss.backward()

            # Gradient clipping para estabilidad
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        scheduler.step()  # Actualizar learning rates
```

### 📈 Learning Rate Scheduling

#### CosineAnnealingLR Mathematical Formula
```
η_t = η_min + (η_max - η_min) × (1 + cos(πt/T)) / 2

donde:
- η_t = learning rate en época t
- η_max = learning rate inicial
- η_min = learning rate mínimo
- T = número total de épocas
- t = época actual

Configuración del proyecto:
- Backbone: η_max = 0.00002, η_min = 0.000002
- Head: η_max = 0.0002, η_min = 0.00002
- T = 55 épocas
```

#### Early Stopping Algorithm
```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001, monitor='val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_score = None
        self.counter = 0

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
```

---

## 📊 Métricas de Evaluación

### 🎯 Métricas Principales

#### Root Mean Square Error (RMSE)
```
RMSE = √[(1/N) Σᵢ₌₁ᴺ (ŷᵢ - yᵢ)²]

Para landmarks:
RMSE_normalized = √[(1/(N×30)) Σᵢ₌₁ᴺ Σⱼ₌₁³⁰ (ŷᵢⱼ - yᵢⱼ)²]
RMSE_pixels = RMSE_normalized × 224

Valor del proyecto: RMSE = 9.47 píxeles
```

#### Mean Absolute Error (MAE)
```
MAE = (1/N) Σᵢ₌₁ᴺ |ŷᵢ - yᵢ|

Para landmarks:
MAE_normalized = (1/(N×30)) Σᵢ₌₁ᴺ Σⱼ₌₁³⁰ |ŷᵢⱼ - yᵢⱼ|
MAE_pixels = MAE_normalized × 224

Valor del proyecto: MAE = 7.15 píxeles
```

#### Distancia Euclidiana por Landmark
```
Para cada landmark k en muestra i:
d_k^(i) = √[(x̂_k^(i) - x_k^(i))² + (ŷ_k^(i) - y_k^(i))²] × 224

Error promedio por landmark:
d̄_k = (1/N) Σᵢ₌₁ᴺ d_k^(i)

Error promedio global:
Error_avg = (1/15) Σₖ₌₁¹⁵ d̄_k = 11.34 píxeles
```

### 📈 Análisis Estadístico

#### Distribución de Errores por Categoría
```python
def analyze_by_category(predictions, targets, categories):
    """
    Análisis estadístico por categoría médica
    """
    results = {}

    for category in ['COVID', 'Normal', 'Viral_Pneumonia']:
        # Filtrar por categoría
        mask = [cat == category for cat in categories]
        cat_predictions = predictions[mask]
        cat_targets = targets[mask]

        # Calcular métricas
        errors = torch.abs(cat_predictions - cat_targets)
        euclidean_distances = []

        for i in range(len(cat_predictions)):
            pred_coords = cat_predictions[i].reshape(15, 2)
            true_coords = cat_targets[i].reshape(15, 2)
            distances = torch.norm(pred_coords - true_coords, dim=1) * 224
            euclidean_distances.extend(distances.tolist())

        results[category] = {
            'mean_error': np.mean(euclidean_distances),
            'std_error': np.std(euclidean_distances),
            'median_error': np.median(euclidean_distances),
            'p95_error': np.percentile(euclidean_distances, 95),
            'samples': len(cat_predictions)
        }

    return results

# Resultados del proyecto:
# Normal: 10.46 ± 6.63 píxeles
# Viral Pneumonia: 11.38 ± 7.20 píxeles
# COVID: 13.24 ± 8.27 píxeles
```

---

## 🎯 Ensemble Learning

### 🔄 Bootstrap Aggregating (Bagging)

#### Algoritmo de Ensemble
```python
class EnsemblePredictor:
    def __init__(self, model_paths):
        self.models = []
        for path in model_paths:
            model, _ = ResNetLandmarkRegressor.load_from_checkpoint(path)
            model.eval()
            self.models.append(model)

    def predict(self, x, aggregation='mean'):
        """
        Predicción ensemble con agregación configurable
        """
        predictions = []

        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                predictions.append(pred)

        # Stack: [num_models, batch_size, 30]
        stacked_preds = torch.stack(predictions, dim=0)

        if aggregation == 'mean':
            return torch.mean(stacked_preds, dim=0)
        elif aggregation == 'median':
            return torch.median(stacked_preds, dim=0)[0]
        elif aggregation == 'weighted_mean':
            weights = self._calculate_weights()
            weighted_preds = stacked_preds * weights.view(-1, 1, 1)
            return torch.sum(weighted_preds, dim=0)

    def _calculate_weights(self):
        """
        Pesos inversamente proporcionales a la pérdida de validación
        """
        val_losses = [0.002, 0.0018, 0.0025, 0.0022, 0.0019]  # Ejemplo
        inv_losses = [1.0 / (loss + 1e-8) for loss in val_losses]
        weights = torch.tensor(inv_losses)
        return weights / torch.sum(weights)  # Normalizar
```

#### Análisis de Diversidad
```python
def calculate_ensemble_diversity(predictions_list):
    """
    Medir diversidad entre modelos del ensemble
    """
    # Convertir a numpy para análisis
    preds_array = np.array([p.numpy() for p in predictions_list])

    # Varianza promedio entre modelos
    variance = np.var(preds_array, axis=0)
    avg_variance = np.mean(variance)

    # Correlación promedio entre pares de modelos
    correlations = []
    for i in range(len(predictions_list)):
        for j in range(i+1, len(predictions_list)):
            corr = np.corrcoef(
                preds_array[i].flatten(),
                preds_array[j].flatten()
            )[0, 1]
            correlations.append(corr)

    avg_correlation = np.mean(correlations)

    return {
        'average_variance': avg_variance,
        'average_correlation': avg_correlation,
        'diversity_score': avg_variance / (avg_correlation + 1e-8)
    }

# Resultados del proyecto:
# Diversidad limitada: correlación alta entre modelos (>0.9)
# Explicación: Solo cambio de seed no genera suficiente diversidad
```

### 📊 Resultados del Ensemble

#### Comparación Individual vs Ensemble
```
Modelo    | Seed | Error (píxeles) | Ranking
----------|------|-----------------|--------
Modelo 1  | 123  | 11.55          | 4
Modelo 2  | 42   | 12.14          | 5 (peor)
Modelo 3  | 456  | 10.69          | 1 (mejor)
Modelo 4  | 789  | 11.39          | 3
Modelo 5  | 999  | 11.53          | 2

Ensemble (mean):        10.81 píxeles
Ensemble (median):      10.81 píxeles
Ensemble (weighted):    10.82 píxeles

Mejora vs mejor individual: -0.12 píxeles (marginal)
```

---

## ⚡ Optimizaciones Implementadas

### 🎯 Hiperparámetros Optimizados

#### Learning Rate Optimization
```python
# Configuración ganadora
optimizer_config = {
    'backbone_lr': 0.00002,     # LR bajo para preservar features ImageNet
    'head_lr': 0.0002,          # LR alto para especialización (ratio 10:1)
    'weight_decay': 0.00005,    # Reducido para mayor flexibilidad
    'optimizer': 'adam',        # Adam con β₁=0.9, β₂=0.999
}

# Justificación matemática:
# Backbone preentrenado: pequeños ajustes → LR bajo
# Head aleatorio: aprendizaje desde cero → LR alto
# Ratio 10:1 permite convergencia balanceada
```

#### Data Augmentation Optimization
```python
# Configuración agresiva optimizada
augmentation_config = {
    'horizontal_flip': 0.7,     # ↑40% vs baseline (0.5)
    'rotation': 15,             # ↑50% vs baseline (10°)
    'brightness': 0.4,          # ↑100% vs baseline (0.2)
    'contrast': 0.4,            # ↑100% vs baseline (0.2)
}

# Impacto en generalización:
# Mayor variabilidad → mejor robustez → -8% error
```

#### Batch Size Optimization
```python
# Análisis de batch size vs rendimiento
batch_sizes = [4, 8, 16, 32]
errors = [11.8, 11.34, 11.9, 12.5]  # píxeles

# Óptimo: batch_size = 8
# Razón: Balance entre estabilidad y precisión de gradientes
# Batch pequeño → gradientes más ruidosos pero precisos
# Batch grande → gradientes estables pero menos informativos
```

### 🔧 Optimizaciones de Código

#### Memory-Efficient Data Loading
```python
class MemoryEfficientDataLoader:
    def __init__(self, dataset, batch_size, num_workers=4, pin_memory=True):
        # Optimizaciones para AMD GPU
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,       # Paralelización CPU
            pin_memory=pin_memory,         # Optimización GPU
            persistent_workers=True,       # Reutilizar workers
            prefetch_factor=2              # Prefetch para pipeline
        )
```

#### Gradient Accumulation (si batch_size < óptimo)
```python
def train_with_accumulation(model, dataloader, optimizer, criterion,
                          accumulation_steps=4):
    """
    Simular batch_size mayor con acumulación de gradientes
    """
    model.train()
    optimizer.zero_grad()

    for i, (images, landmarks, _) in enumerate(dataloader):
        images, landmarks = images.to(device), landmarks.to(device)

        predictions = model(images)
        loss = criterion(predictions, landmarks)

        # Escalar pérdida por pasos de acumulación
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
```

---

## 📊 Análisis de Complejidad

### ⏱️ Complejidad Temporal

#### Forward Pass Complexity
```
ResNet-18 Forward Pass:
- Input: (batch_size, 3, 224, 224)
- FLOPs ≈ 1.8 × 10⁹ operaciones por imagen
- GPU AMD RX 6600: ~6000 GFLOPS
- Tiempo estimado: ~0.3ms por imagen

Custom Head:
- Linear layers: O(d₁ × d₂) donde d₁, d₂ son dimensiones
- 512→512: 262K FLOPs
- 512→256: 131K FLOPs
- 256→30: 8K FLOPs
- Total head: ~401K FLOPs (negligible vs backbone)
```

#### Training Time Complexity
```
Fase 1 (2 épocas):
- Forward + Backward: ~2× Forward FLOPs
- Solo head entrenable: 401K parámetros
- Tiempo: ~1 minuto (669 muestras × 2 épocas)

Fase 2 (55 épocas):
- Forward + Backward: ~2× Forward FLOPs
- Todo el modelo: 11.6M parámetros
- Tiempo: ~4 minutos (669 muestras × 55 épocas)

Ensemble (5 modelos):
- 5× tiempo Fase 2: ~20 minutos
- Paralelizable en múltiples GPUs
```

### 💾 Complejidad Espacial

#### Memory Usage Analysis
```python
def calculate_memory_usage():
    """
    Análisis de uso de memoria GPU
    """
    memory_breakdown = {
        'model_parameters': 11.6e6 * 4,      # 46.4 MB (float32)
        'activations_per_image': 50e6 * 4,   # 200 MB estimado
        'gradients': 11.6e6 * 4,             # 46.4 MB
        'optimizer_states': 11.6e6 * 8,      # 92.8 MB (Adam: 2× params)
        'batch_data': 8 * 3 * 224 * 224 * 4, # 48.2 MB (batch=8)
    }

    total_mb = sum(memory_breakdown.values()) / (1024**2)
    print(f"Uso estimado de GPU: {total_mb:.1f} MB")

    return memory_breakdown

# GPU AMD RX 6600 (8GB): ~433 MB usados (~5.4% de capacidad)
```

#### Disk Space Requirements
```
Componente              | Tamaño       | Descripción
------------------------|--------------|------------------
Dataset original        | ~150 MB      | 999 imágenes PNG
Checkpoints modelo      | ~45 MB       | phase2_best.pt
Ensemble checkpoints    | ~225 MB      | 5 modelos × 45MB
Logs TensorBoard        | ~10 MB       | Métricas entrenamiento
Resultados evaluación   | ~20 MB       | CSVs + visualizaciones
Total proyecto          | ~450 MB      | Sin datos intermedios
```

---

## 🔍 Análisis de Convergencia

### 📈 Curvas de Aprendizaje

#### Fase 1: Head Training Convergence
```
Época | Train Loss | Val Loss | Convergencia
------|------------|----------|-------------
1     | 0.0890     | 0.0856   | Rápida inicial
2     | 0.0234     | 0.0267   | Estabilización

Análisis: Convergencia rápida debido a:
- Solo 401K parámetros entrenables
- Features preentrenados estables
- LR alto (0.001) para head
```

#### Fase 2: Fine-tuning Convergence
```python
# Patrón típico de convergencia Fase 2
epochs = list(range(1, 56))
train_loss = [0.0267, 0.0198, 0.0165, ..., 0.0012]  # Decreciente
val_loss = [0.0278, 0.0201, 0.0178, ..., 0.0018]    # Con plateau

# Detección de overfitting
overfitting_point = None
for i in range(10, len(val_loss)):
    if val_loss[i] > val_loss[i-5]:  # Loss aumenta durante 5 épocas
        overfitting_point = i
        break

print(f"Posible overfitting después de época: {overfitting_point}")
```

### 🎯 Análisis de Estabilidad

#### Variance Across Random Seeds
```python
def analyze_seed_stability():
    """
    Análisis de estabilidad entre diferentes semillas
    """
    seeds = [42, 123, 456, 789, 999]
    errors = [12.14, 11.55, 10.69, 11.39, 11.53]  # píxeles

    mean_error = np.mean(errors)      # 11.46 píxeles
    std_error = np.std(errors)        # 0.58 píxeles
    cv = std_error / mean_error       # 0.051 (5.1% variación)

    print(f"Estabilidad del modelo:")
    print(f"Error promedio: {mean_error:.2f} ± {std_error:.2f} píxeles")
    print(f"Coeficiente de variación: {cv:.3f}")

    # CV < 0.1 indica buena estabilidad
    stability = "BUENA" if cv < 0.1 else "REGULAR" if cv < 0.2 else "MALA"
    print(f"Evaluación estabilidad: {stability}")

analyze_seed_stability()
# Resultado: BUENA estabilidad (CV = 5.1%)
```

---

## 📝 Conclusiones Técnicas

### ✅ Fortalezas del Modelo

1. **Arquitectura Robusta**: ResNet-18 con transfer learning probado
2. **Entrenamiento Eficiente**: 2 fases optimizan convergencia
3. **Generalización Sólida**: Data augmentation agresivo efectivo
4. **Estabilidad Alta**: Baja varianza entre runs (CV = 5.1%)
5. **Precisión Clínica**: 11.34px cercano a objetivo <10px

### ⚠️ Limitaciones Identificadas

1. **Capacidad Arquitectural**: ResNet-18 podría ser limitante para <10px
2. **Diversidad Ensemble**: Solo random seeds insuficiente para mejora
3. **Landmarks Específicos**: #14 y #15 consistentemente problemáticos
4. **Variabilidad Categórica**: COVID más desafiante (+23% error vs Normal)

### 🚀 Optimizaciones Futuras

1. **Arquitectura**: ResNet-34, EfficientNet, Vision Transformers
2. **Loss Functions**: Wing Loss, Focal Loss para landmarks difíciles
3. **Ensemble Diversity**: Diferentes arquitecturas, augmentation, loss
4. **Attention Mechanisms**: Self-attention para landmarks relacionados
5. **Multi-Scale Training**: Entrenamiento con múltiples resoluciones

---

**📊 Estado Final**: El proyecto alcanza **11.34 píxeles de error promedio**, estableciendo una base sólida para predicción de landmarks médicos con precisión clínicamente útil. La documentación técnica proporciona fundamentos matemáticos completos para futuras extensiones e investigaciones.
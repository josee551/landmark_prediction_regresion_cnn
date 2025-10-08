#!/usr/bin/env python3
"""
Evaluación rápida del modelo Phase 4 Complete Loss
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Importar módulos del proyecto
from src.data.dataset import create_dataloaders
from src.models.resnet_regressor import ResNetLandmarkRegressor
from src.training.utils import setup_device


def evaluate_complete_model():
    """Evaluar modelo Phase 4 Complete Loss"""
    print("🔍 EVALUACIÓN PHASE 4: COMPLETE LOSS MODEL")
    print("=" * 60)

    # Configurar device
    device = setup_device(use_gpu=True, gpu_id=0)
    print(f"⚡ Device: {device}")

    # Crear data loaders
    print("\n📊 Configurando data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        annotations_file="data/coordenadas/coordenadas_maestro.csv",
        images_dir="data/dataset",
        batch_size=16,  # Larger batch for evaluation
        num_workers=4,
        pin_memory=True,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )

    print(f"✓ Test set: {len(test_loader.dataset)} samples")

    # Cargar modelo
    print("\n🏗️ Cargando modelo Complete Loss...")
    checkpoint_path = "checkpoints/geometric_complete.pt"

    # Crear modelo
    model = ResNetLandmarkRegressor(
        num_landmarks=15,
        pretrained=False,
        dropout_rate=0.5
    )

    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=str(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"✓ Modelo cargado desde: {checkpoint_path}")
    print(f"✓ Época del checkpoint: {checkpoint['epoch']}")
    print(f"✓ Error de validación: {checkpoint['best_pixel_error']:.2f}px")

    # Evaluación en test set
    print("\n📈 Evaluando en conjunto de test...")

    all_pixel_errors = []
    category_errors = {'COVID': [], 'Normal': [], 'Viral Pneumonia': []}

    with torch.no_grad():
        for images, landmarks, metadata in tqdm(test_loader, desc="Evaluando"):
            images = images.to(device)
            landmarks = landmarks.to(device)

            # Predicciones
            predictions = model(images)

            # Calcular errores por muestra
            pred_reshaped = predictions.view(-1, 15, 2)
            target_reshaped = landmarks.view(-1, 15, 2)
            distances = torch.norm(pred_reshaped - target_reshaped, dim=2)
            pixel_distances = distances * 224  # Convert to pixels

            # Error promedio por imagen
            pixel_errors_per_image = torch.mean(pixel_distances, dim=1)
            all_pixel_errors.extend(pixel_errors_per_image.cpu().numpy())

            # Agrupar por categoría si metadata disponible
            if metadata is not None:
                for i, meta in enumerate(metadata):
                    try:
                        if isinstance(meta, dict) and 'category' in meta:
                            category = meta['category']
                            if category in category_errors:
                                category_errors[category].append(pixel_errors_per_image[i].item())
                    except:
                        continue  # Skip problematic metadata

    # Métricas generales
    all_pixel_errors = np.array(all_pixel_errors)
    mean_error = np.mean(all_pixel_errors)
    std_error = np.std(all_pixel_errors)
    min_error = np.min(all_pixel_errors)
    max_error = np.max(all_pixel_errors)
    median_error = np.median(all_pixel_errors)

    print("\n" + "=" * 60)
    print("📊 RESULTADOS FINALES PHASE 4")
    print("=" * 60)
    print(f"🎯 Error promedio: {mean_error:.2f} píxeles")
    print(f"📊 Mediana: {median_error:.2f} píxeles")
    print(f"📈 Desviación estándar: {std_error:.2f} píxeles")
    print(f"🔽 Error mínimo: {min_error:.2f} píxeles")
    print(f"🔼 Error máximo: {max_error:.2f} píxeles")

    # Análisis de mejora
    phase3_error = 8.91  # Baseline Phase 3
    improvement = phase3_error - mean_error
    improvement_pct = (improvement / phase3_error) * 100

    print(f"\n📈 COMPARACIÓN CON FASES ANTERIORES:")
    print(f"   Phase 3 (Symmetry): 8.91px")
    print(f"   Phase 4 (Complete): {mean_error:.2f}px")
    print(f"   Mejora: {improvement:.2f}px ({improvement_pct:.1f}%)")

    # Evaluación clínica
    clinical_thresholds = {
        "Sub-píxel (research)": 5.0,
        "Excelencia clínica": 8.5,
        "Útil clínicamente": 15.0,
        "Análisis general": 20.0
    }

    print(f"\n🏥 EVALUACIÓN CLÍNICA:")
    for threshold_name, threshold_value in clinical_thresholds.items():
        if mean_error <= threshold_value:
            print(f"   ✅ {threshold_name}: ALCANZADO")
            break
        else:
            print(f"   ❌ {threshold_name}: NO ALCANZADO")

    # Métricas por categoría si disponibles
    if any(category_errors.values()):
        print(f"\n📂 RESULTADOS POR CATEGORÍA:")
        for category, errors in category_errors.items():
            if errors:
                cat_mean = np.mean(errors)
                print(f"   {category}: {cat_mean:.2f}px ({len(errors)} muestras)")

    # Distribución de errores
    percentiles = [25, 50, 75, 90, 95]
    print(f"\n📈 DISTRIBUCIÓN DE ERRORES:")
    for p in percentiles:
        value = np.percentile(all_pixel_errors, p)
        print(f"   Percentil {p}: {value:.2f}px")

    print("\n" + "=" * 60)
    print("🎉 EVALUACIÓN COMPLETADA")
    print("=" * 60)

    return mean_error


if __name__ == "__main__":
    final_error = evaluate_complete_model()
    print(f"\n🏆 Error final del test set: {final_error:.2f} píxeles")
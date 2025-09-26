#!/usr/bin/env python3
"""
Evaluación simple del modelo de symmetry training
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Importar módulos del proyecto
from src.data.dataset import create_dataloaders
from src.models.resnet_regressor import ResNetLandmarkRegressor
from src.training.utils import setup_device

def compute_pixel_error(predictions, targets, image_size=224):
    """Calcular error en píxeles"""
    pred_reshaped = predictions.view(-1, 15, 2)
    target_reshaped = targets.view(-1, 15, 2)
    distances = torch.norm(pred_reshaped - target_reshaped, dim=2)
    pixel_distances = distances * image_size
    return torch.mean(pixel_distances)

def evaluate_model():
    print("🔍 EVALUACIÓN DEL MODELO SYMMETRY")
    print("=" * 50)

    # Configurar device
    device = setup_device(use_gpu=True, gpu_id=0)
    print(f"Device: {device}")

    # Cargar datos
    print("\n📊 Cargando datos...")
    _, val_loader, test_loader = create_dataloaders(
        annotations_file="data/coordenadas/coordenadas_maestro.csv",
        images_dir="data/dataset",
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )

    print(f"✓ Validation: {len(val_loader.dataset)} samples")
    print(f"✓ Test: {len(test_loader.dataset)} samples")

    # Cargar modelo
    print("\n🏗️ Cargando modelo...")
    checkpoint_path = "checkpoints/geometric_symmetry.pt"

    # Crear modelo base
    model = ResNetLandmarkRegressor(num_landmarks=15, pretrained=False, freeze_backbone=False)

    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=str(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Modelo cargado desde: {checkpoint_path}")
    print(f"✓ Época entrenada: {checkpoint.get('epoch', 'N/A')}")
    print(f"✓ Best error durante entrenamiento: {checkpoint.get('best_pixel_error', 'N/A'):.2f}px")

    # Evaluación en validation set
    print("\n📈 Evaluando en validation set...")
    val_errors = []
    val_total_error = 0.0

    with torch.no_grad():
        for images, landmarks, _ in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            landmarks = landmarks.to(device)

            predictions = model(images)
            pixel_error = compute_pixel_error(predictions, landmarks)

            val_total_error += pixel_error.item()

            # Errores por muestra
            pred_reshaped = predictions.view(-1, 15, 2)
            target_reshaped = landmarks.view(-1, 15, 2)
            distances = torch.norm(pred_reshaped - target_reshaped, dim=2)
            pixel_distances = distances * 224
            sample_errors = torch.mean(pixel_distances, dim=1)
            val_errors.extend(sample_errors.cpu().numpy())

    val_mean_error = val_total_error / len(val_loader)

    # Evaluación en test set
    print("\n🎯 Evaluando en test set...")
    test_errors = []
    test_total_error = 0.0

    with torch.no_grad():
        for images, landmarks, _ in tqdm(test_loader, desc="Test"):
            images = images.to(device)
            landmarks = landmarks.to(device)

            predictions = model(images)
            pixel_error = compute_pixel_error(predictions, landmarks)

            test_total_error += pixel_error.item()

            # Errores por muestra
            pred_reshaped = predictions.view(-1, 15, 2)
            target_reshaped = landmarks.view(-1, 15, 2)
            distances = torch.norm(pred_reshaped - target_reshaped, dim=2)
            pixel_distances = distances * 224
            sample_errors = torch.mean(pixel_distances, dim=1)
            test_errors.extend(sample_errors.cpu().numpy())

    test_mean_error = test_total_error / len(test_loader)

    # Resultados finales
    print("\n" + "=" * 60)
    print("🎉 RESULTADOS FINALES - PHASE 3: SYMMETRY LOSS")
    print("=" * 60)

    print(f"\n📊 VALIDATION SET:")
    print(f"  Error promedio: {val_mean_error:.2f} píxeles")
    print(f"  Error std: {np.std(val_errors):.2f} píxeles")
    print(f"  Error mínimo: {np.min(val_errors):.2f} píxeles")
    print(f"  Error máximo: {np.max(val_errors):.2f} píxeles")

    print(f"\n🎯 TEST SET:")
    print(f"  Error promedio: {test_mean_error:.2f} píxeles")
    print(f"  Error std: {np.std(test_errors):.2f} píxeles")
    print(f"  Error mínimo: {np.min(test_errors):.2f} píxeles")
    print(f"  Error máximo: {np.max(test_errors):.2f} píxeles")

    # Comparación con objetivos
    print(f"\n🏆 COMPARACIÓN CON OBJETIVOS:")
    print(f"  Baseline original: 11.34px")
    print(f"  Phase 1 (Wing Loss): 10.91px")
    print(f"  Phase 3 (Symmetry): {test_mean_error:.2f}px")
    print(f"  Objetivo Phase 3: ≤9.3px")
    print(f"  Estado: {'✅ ALCANZADO' if test_mean_error <= 9.3 else '❌ NO ALCANZADO'}")

    if test_mean_error <= 9.3:
        improvement = 11.34 - test_mean_error
        percentage = (improvement / 11.34) * 100
        print(f"  Mejora total: {improvement:.2f}px ({percentage:.1f}% reducción)")

    return test_mean_error

if __name__ == "__main__":
    final_error = evaluate_model()
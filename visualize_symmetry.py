#!/usr/bin/env python3
"""
Visualización de predicciones del modelo symmetry
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import DataLoader

# Importar módulos del proyecto
from src.data.dataset import create_dataloaders
from src.models.resnet_regressor import ResNetLandmarkRegressor
from src.training.utils import setup_device

def visualize_predictions():
    print("🎨 VISUALIZACIÓN DEL MODELO SYMMETRY")
    print("=" * 50)

    # Configurar device
    device = setup_device(use_gpu=True, gpu_id=0)
    print(f"Device: {device}")

    # Cargar datos
    print("\n📊 Cargando datos...")
    _, val_loader, test_loader = create_dataloaders(
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

    # Cargar modelo
    print("\n🏗️ Cargando modelo...")
    model = ResNetLandmarkRegressor(num_landmarks=15, pretrained=False, freeze_backbone=False)
    checkpoint = torch.load("checkpoints/geometric_symmetry.pt", map_location=str(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Modelo Phase 3 Symmetry cargado")
    print(f"✓ Best error: {checkpoint['best_pixel_error']:.2f}px")

    # Crear directorio de visualizaciones
    viz_dir = Path("evaluation_results/symmetry_visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🎨 Generando visualizaciones en: {viz_dir}")

    # Obtener algunas muestras del test set
    test_iter = iter(test_loader)
    images, landmarks, categories = next(test_iter)

    images = images.to(device)
    landmarks = landmarks.to(device)

    with torch.no_grad():
        predictions = model(images)

    # Convertir a numpy y píxeles
    images_np = images.cpu().numpy()
    landmarks_np = landmarks.cpu().numpy() * 224  # Convert to pixels
    predictions_np = predictions.cpu().numpy() * 224  # Convert to pixels
    categories_list = categories

    # Calcular errores por muestra
    errors = []
    for i in range(len(predictions_np)):
        pred_reshaped = predictions_np[i].reshape(15, 2)
        target_reshaped = landmarks_np[i].reshape(15, 2)
        distances = np.linalg.norm(pred_reshaped - target_reshaped, axis=1)
        mean_error = np.mean(distances)
        errors.append(mean_error)

    # Visualizar las primeras 8 muestras
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    colors_gt = ['lime', 'cyan', 'yellow', 'magenta', 'orange']
    colors_pred = ['red', 'blue', 'green', 'purple', 'brown']

    for i in range(min(8, len(images_np))):
        ax = axes[i]

        # Mostrar imagen (convertir de tensor a imagen)
        img = images_np[i].transpose(1, 2, 0)  # CHW -> HWC
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))  # Desnormalizar
        img = np.clip(img, 0, 1)

        ax.imshow(img)

        # Landmarks ground truth (verde/lime)
        gt_points = landmarks_np[i].reshape(15, 2)
        pred_points = predictions_np[i].reshape(15, 2)

        # Plot ground truth
        ax.scatter(gt_points[:, 0], gt_points[:, 1],
                  c='lime', s=50, marker='o', alpha=0.8, label='Ground Truth')

        # Plot predictions
        ax.scatter(pred_points[:, 0], pred_points[:, 1],
                  c='red', s=30, marker='x', alpha=0.8, label='Prediction')

        # Conectar pares con líneas para mostrar diferencias
        for j in range(15):
            ax.plot([gt_points[j, 0], pred_points[j, 0]],
                   [gt_points[j, 1], pred_points[j, 1]],
                   'yellow', alpha=0.5, linewidth=1)

        try:
            category = categories_list[i] if hasattr(categories_list, '__getitem__') else f"Sample_{i}"
        except:
            category = f"Sample_{i}"
        ax.set_title(f'{category}\nError: {errors[i]:.1f}px', fontsize=10)
        ax.set_xlim(0, 224)
        ax.set_ylim(224, 0)  # Invertir Y para mostrar correctamente
        ax.axis('off')

        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle(f'Phase 3: Symmetry Loss Model\nBest Validation: {checkpoint["best_pixel_error"]:.2f}px',
                 fontsize=16, y=0.95)
    plt.tight_layout()

    # Guardar visualización
    output_path = viz_dir / "symmetry_predictions_overview.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Guardado: {output_path}")
    plt.close()

    # Crear visualización comparativa de errores
    plt.figure(figsize=(12, 6))

    # Subplot 1: Histograma de errores
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(errors), color='red', linestyle='--',
                label=f'Mean: {np.mean(errors):.1f}px')
    plt.axvline(9.3, color='green', linestyle='--',
                label='Target: 9.3px')
    plt.xlabel('Error (píxeles)')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Errores (Batch)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Comparación con fases anteriores
    plt.subplot(1, 2, 2)
    phases = ['Baseline\n(MSE)', 'Phase 1\n(Wing Loss)', 'Phase 3\n(Symmetry)']
    errors_comparison = [11.34, 10.91, checkpoint['best_pixel_error']]
    colors = ['lightcoral', 'orange', 'lightgreen']

    bars = plt.bar(phases, errors_comparison, color=colors, alpha=0.8, edgecolor='black')
    plt.axhline(9.3, color='red', linestyle='--', alpha=0.7, label='Target: 9.3px')

    # Añadir valores en las barras
    for bar, value in zip(bars, errors_comparison):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}px', ha='center', va='bottom', fontweight='bold')

    plt.ylabel('Error Promedio (píxeles)')
    plt.title('Progreso del Proyecto')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 12)

    plt.suptitle('Phase 3: Symmetry Loss - Análisis de Resultados', fontsize=14, y=0.98)
    plt.tight_layout()

    # Guardar análisis
    analysis_path = viz_dir / "symmetry_analysis.png"
    plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
    print(f"✓ Guardado: {analysis_path}")
    plt.close()

    print(f"\n🎉 VISUALIZACIONES COMPLETADAS")
    print(f"📁 Ubicación: {viz_dir}")
    print(f"📊 Archivos generados:")
    print(f"  - symmetry_predictions_overview.png")
    print(f"  - symmetry_analysis.png")

if __name__ == "__main__":
    visualize_predictions()
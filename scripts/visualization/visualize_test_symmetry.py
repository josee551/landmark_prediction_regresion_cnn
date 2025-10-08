#!/usr/bin/env python3
"""
Visualización completa del dataset de test para el modelo Symmetry
Genera imágenes individuales para todas las 144 muestras del test set
Similar al comando visualize_test pero para el modelo de symmetry
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

# Importar módulos del proyecto
from src.data.dataset import create_dataloaders
from src.models.resnet_regressor import ResNetLandmarkRegressor
from src.training.utils import setup_device

def create_individual_prediction_image(image, gt_landmarks, pred_landmarks,
                                     error, category, filename, save_path):
    """
    Crear imagen individual con predicciones vs ground truth
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Convertir imagen de tensor a numpy
    if isinstance(image, torch.Tensor):
        img = image.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
        # Desnormalizar (ImageNet stats)
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img = np.clip(img, 0, 1)
    else:
        img = image

    ax.imshow(img)

    # Convertir landmarks a píxeles si están normalizados
    if gt_landmarks.max() <= 1.0:
        gt_points = gt_landmarks.reshape(15, 2) * 224
        pred_points = pred_landmarks.reshape(15, 2) * 224
    else:
        gt_points = gt_landmarks.reshape(15, 2)
        pred_points = pred_landmarks.reshape(15, 2)

    # Plot ground truth (verde)
    ax.scatter(gt_points[:, 0], gt_points[:, 1],
              c='lime', s=80, marker='o', alpha=0.9,
              label='Ground Truth', edgecolors='darkgreen', linewidth=2)

    # Plot predictions (rojo)
    ax.scatter(pred_points[:, 0], pred_points[:, 1],
              c='red', s=60, marker='x', alpha=0.9,
              label='Prediction', linewidth=3)

    # Líneas conectando GT con predicciones
    for i in range(15):
        ax.plot([gt_points[i, 0], pred_points[i, 0]],
               [gt_points[i, 1], pred_points[i, 1]],
               'yellow', alpha=0.6, linewidth=2)

    # Numerar landmarks para análisis detallado
    for i in range(15):
        ax.annotate(f'{i}', (gt_points[i, 0], gt_points[i, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, color='white', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    # Configurar título y estilo
    color = '🟢' if error <= 9.3 else '🟡' if error <= 12 else '🔴'
    ax.set_title(f'{color} {category} - {filename}\n'
                f'Error: {error:.2f}px | Target: ≤9.3px | '
                f'Phase 3: Symmetry Loss',
                fontsize=12, pad=20)

    ax.set_xlim(0, 224)
    ax.set_ylim(224, 0)  # Invertir Y
    ax.legend(loc='upper right', fontsize=10)
    ax.axis('off')

    # Añadir información adicional
    textstr = f'Model: Phase 3 Symmetry\nWing Loss + Bilateral Constraints\nBest Training: 8.48px'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def compute_pixel_error(predictions, targets, image_size=224):
    """Calcular error en píxeles por muestra"""
    pred_reshaped = predictions.view(-1, 15, 2)
    target_reshaped = targets.view(-1, 15, 2)
    distances = torch.norm(pred_reshaped - target_reshaped, dim=2)
    pixel_distances = distances * image_size
    return torch.mean(pixel_distances, dim=1)  # Error por muestra

def visualize_complete_test_set():
    print("🎨 VISUALIZACIÓN COMPLETA DEL TEST SET - SYMMETRY MODEL")
    print("=" * 70)

    # Configurar device
    device = setup_device(use_gpu=True, gpu_id=0)
    print(f"Device: {device}")

    # Cargar datos - usar batch_size=1 para procesar individualmente
    print("\n📊 Cargando datos...")
    _, _, test_loader = create_dataloaders(
        annotations_file="data/coordenadas/coordenadas_maestro.csv",
        images_dir="data/dataset",
        batch_size=1,  # Procesar de a una imagen
        num_workers=2,
        pin_memory=True,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )

    print(f"✓ Test samples: {len(test_loader.dataset)}")

    # Cargar modelo
    print("\n🏗️ Cargando modelo Symmetry...")
    model = ResNetLandmarkRegressor(num_landmarks=15, pretrained=False, freeze_backbone=False)
    checkpoint = torch.load("checkpoints/geometric_symmetry.pt", map_location=str(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Modelo Phase 3 Symmetry cargado")
    print(f"✓ Best training error: {checkpoint['best_pixel_error']:.2f}px")

    # Crear directorio de salida
    output_dir = Path("evaluation_results/test_predictions_symmetry")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Directorio de salida: {output_dir}")

    # Procesar todas las muestras del test set
    print(f"\n🔄 Procesando {len(test_loader)} muestras del test set...")

    all_errors = []
    categories_count = {}
    results_data = []

    with torch.no_grad():
        for idx, (images, landmarks, categories) in enumerate(tqdm(test_loader, desc="Generando visualizaciones")):
            # Mover a device
            images = images.to(device)
            landmarks = landmarks.to(device)

            # Hacer predicción
            predictions = model(images)

            # Calcular error para esta muestra
            error = compute_pixel_error(predictions, landmarks)[0].item()
            all_errors.append(error)

            # Obtener información de la muestra
            category = categories[0] if len(categories) > 0 else "Unknown"
            categories_count[category] = categories_count.get(category, 0) + 1

            # Generar nombre de archivo
            filename = f"{category}_{categories_count[category]:03d}_error_{error:.2f}px"
            save_path = output_dir / f"{filename}.png"

            # Crear visualización individual
            create_individual_prediction_image(
                image=images[0],
                gt_landmarks=landmarks[0].cpu().numpy(),
                pred_landmarks=predictions[0].cpu().numpy(),
                error=error,
                category=category,
                filename=f"{category}_{categories_count[category]:03d}",
                save_path=save_path
            )

            # Guardar datos para análisis
            results_data.append({
                'filename': filename,
                'category': category,
                'error_px': error,
                'achieved_target': error <= 9.3
            })

    # Crear resumen estadístico
    print(f"\n📊 RESUMEN DE RESULTADOS:")
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    min_error = np.min(all_errors)
    max_error = np.max(all_errors)

    print(f"  📈 Error promedio: {mean_error:.2f}px")
    print(f"  📊 Desviación estándar: {std_error:.2f}px")
    print(f"  🎯 Error mínimo: {min_error:.2f}px")
    print(f"  ⚠️  Error máximo: {max_error:.2f}px")

    # Análisis por categorías
    print(f"\n📋 ANÁLISIS POR CATEGORÍA:")
    for category in categories_count:
        cat_errors = [r['error_px'] for r in results_data if r['category'] == category]
        cat_achieved = [r['achieved_target'] for r in results_data if r['category'] == category]

        print(f"  {category}:")
        print(f"    Muestras: {len(cat_errors)}")
        print(f"    Error promedio: {np.mean(cat_errors):.2f}px")
        print(f"    Target alcanzado: {sum(cat_achieved)}/{len(cat_achieved)} ({100*sum(cat_achieved)/len(cat_achieved):.1f}%)")

    # Comparación con objetivos
    achieved_target = sum(1 for e in all_errors if e <= 9.3)
    print(f"\n🎯 CUMPLIMIENTO DE OBJETIVOS:")
    print(f"  Target (≤9.3px): {achieved_target}/{len(all_errors)} ({100*achieved_target/len(all_errors):.1f}%)")
    print(f"  Baseline original: 11.34px")
    print(f"  Phase 1 (Wing): 10.91px")
    print(f"  Phase 3 (Symmetry): {mean_error:.2f}px")
    print(f"  Mejora total: {11.34 - mean_error:.2f}px ({100*(11.34-mean_error)/11.34:.1f}% reducción)")

    # Guardar resultados en CSV
    df_results = pd.DataFrame(results_data)
    csv_path = output_dir / "symmetry_test_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n💾 Resultados guardados en: {csv_path}")

    # Crear gráfico de resumen
    plt.figure(figsize=(15, 10))

    # Subplot 1: Histograma de errores
    plt.subplot(2, 2, 1)
    plt.hist(all_errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(mean_error, color='red', linestyle='--', label=f'Mean: {mean_error:.2f}px')
    plt.axvline(9.3, color='green', linestyle='--', label='Target: 9.3px')
    plt.xlabel('Error (píxeles)')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Errores - Test Set Completo')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Errores por categoría
    plt.subplot(2, 2, 2)
    categories = list(categories_count.keys())
    cat_means = [np.mean([r['error_px'] for r in results_data if r['category'] == cat]) for cat in categories]
    colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow'][:len(categories)]

    bars = plt.bar(categories, cat_means, color=colors, alpha=0.8, edgecolor='black')
    plt.axhline(9.3, color='red', linestyle='--', alpha=0.7, label='Target: 9.3px')

    for bar, value in zip(bars, cat_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}px', ha='center', va='bottom', fontweight='bold')

    plt.ylabel('Error Promedio (píxeles)')
    plt.title('Error por Categoría Médica')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)

    # Subplot 3: Progreso del proyecto
    plt.subplot(2, 2, 3)
    phases = ['Baseline\n(MSE)', 'Phase 1\n(Wing)', 'Phase 3\n(Symmetry)']
    errors_progress = [11.34, 10.91, mean_error]
    colors_progress = ['lightcoral', 'orange', 'lightgreen']

    bars = plt.bar(phases, errors_progress, color=colors_progress, alpha=0.8, edgecolor='black')
    plt.axhline(9.3, color='red', linestyle='--', alpha=0.7, label='Target: 9.3px')

    for bar, value in zip(bars, errors_progress):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}px', ha='center', va='bottom', fontweight='bold')

    plt.ylabel('Error Promedio (píxeles)')
    plt.title('Progreso del Proyecto')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 12)

    # Subplot 4: Cumplimiento de targets
    plt.subplot(2, 2, 4)
    target_data = [achieved_target, len(all_errors) - achieved_target]
    labels = [f'≤9.3px\n({achieved_target})', f'>9.3px\n({len(all_errors) - achieved_target})']
    colors_pie = ['lightgreen', 'lightcoral']

    plt.pie(target_data, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    plt.title('Cumplimiento del Target')

    plt.suptitle(f'Phase 3: Symmetry Loss - Análisis Completo del Test Set\n'
                f'Error Promedio: {mean_error:.2f}px | Target: ≤9.3px | '
                f'Muestras: {len(all_errors)}',
                fontsize=16, y=0.98)
    plt.tight_layout()

    # Guardar gráfico de resumen
    summary_path = output_dir / "symmetry_complete_analysis.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n🎉 VISUALIZACIÓN COMPLETA TERMINADA")
    print(f"📁 Ubicación: {output_dir}")
    print(f"📊 Archivos generados:")
    print(f"  - {len(all_errors)} imágenes individuales")
    print(f"  - symmetry_test_results.csv (datos completos)")
    print(f"  - symmetry_complete_analysis.png (resumen)")

    return mean_error, output_dir

if __name__ == "__main__":
    final_error, output_path = visualize_complete_test_set()
#!/usr/bin/env python3
"""
Generación de visualizaciones completas para el modelo Phase 4: Complete Loss
Script robusto para generar todas las predicciones del test set
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Importar módulos del proyecto
from src.data.dataset import create_dataloaders
from src.models.resnet_regressor import ResNetLandmarkRegressor
from src.training.utils import setup_device


def create_prediction_image(image, gt_landmarks, pred_landmarks, error, filename, save_path):
    """Crear imagen con predicciones vs ground truth"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Procesar imagen
    if isinstance(image, torch.Tensor):
        img = image.cpu().numpy().transpose(1, 2, 0)
        # Desnormalizar ImageNet
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img = np.clip(img, 0, 1)
    else:
        img = image

    ax.imshow(img)

    # Convertir landmarks a píxeles
    if gt_landmarks.max() <= 1.0:
        gt_points = gt_landmarks.reshape(15, 2) * 224
        pred_points = pred_landmarks.reshape(15, 2) * 224
    else:
        gt_points = gt_landmarks.reshape(15, 2)
        pred_points = pred_landmarks.reshape(15, 2)

    # Plotear landmarks
    ax.scatter(gt_points[:, 0], gt_points[:, 1], c='lime', s=100, marker='o',
              alpha=0.9, label='Ground Truth', edgecolors='darkgreen', linewidth=2)
    ax.scatter(pred_points[:, 0], pred_points[:, 1], c='red', s=80, marker='x',
              alpha=0.9, label='Prediction', linewidth=3)

    # Líneas de conexión para mostrar errores
    for i in range(15):
        ax.plot([gt_points[i, 0], pred_points[i, 0]],
               [gt_points[i, 1], pred_points[i, 1]],
               'yellow', alpha=0.6, linewidth=1)

    # Configurar plot
    ax.set_xlim(0, 224)
    ax.set_ylim(224, 0)  # Invertir Y para que coincida con imagen

    # Crear título más informativo
    title_parts = filename.split('_')
    if len(title_parts) >= 2:
        category = title_parts[0]
        image_name = '_'.join(title_parts[1:])
        title = f'{category}: {image_name}\nError: {error:.2f}px (Complete Loss Model)'
    else:
        title = f'{filename}\nError: {error:.2f}px (Complete Loss Model)'

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Guardar imagen
    plt.tight_layout()
    full_path = os.path.join(save_path, f"{filename}_error_{error:.2f}px.png")
    plt.savefig(full_path, dpi=150, bbox_inches='tight')
    plt.close()

    return full_path


def compute_pixel_error(pred_landmarks, gt_landmarks):
    """Calcular error en píxeles"""
    pred_reshaped = pred_landmarks.reshape(15, 2)
    gt_reshaped = gt_landmarks.reshape(15, 2)

    if pred_reshaped.max() <= 1.0:
        pred_reshaped *= 224
        gt_reshaped *= 224

    distances = np.sqrt(np.sum((pred_reshaped - gt_reshaped) ** 2, axis=1))
    return np.mean(distances)


def visualize_complete_loss_test_set():
    """Generar visualizaciones para todo el test set del modelo Complete Loss"""
    print("🖼️ GENERACIÓN DE VISUALIZACIONES - PHASE 4 COMPLETE LOSS")
    print("=" * 70)

    # Configurar device
    device = setup_device(use_gpu=True, gpu_id=0)
    print(f"⚡ Device: {device}")

    # Función de collate personalizada para manejar metadata correctamente
    def custom_collate_fn(batch):
        """Collate function personalizada para preservar metadata"""
        images = torch.stack([item[0] for item in batch])
        landmarks = torch.stack([item[1] for item in batch])
        metadata = [item[2] for item in batch]  # Mantener como lista
        return images, landmarks, metadata

    # Crear data loaders
    print("\n📊 Configurando data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        annotations_file="data/coordenadas/coordenadas_maestro.csv",
        images_dir="data/dataset",
        batch_size=1,  # Batch size 1 para visualizaciones individuales
        num_workers=2,
        pin_memory=True,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )

    # Reemplazar el collate_fn del test_loader
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_loader.dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    print(f"✓ Test set: {len(test_loader.dataset)} samples")

    # Cargar modelo Complete Loss
    print("\n🏗️ Cargando modelo Complete Loss...")
    checkpoint_path = "checkpoints/geometric_complete.pt"

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint no encontrado: {checkpoint_path}")
        return False

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

    # Crear directorio de salida
    output_dir = "evaluation_results/test_predictions_complete_loss"
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Directorio de salida: {output_dir}")

    # Generar visualizaciones
    print("\n🎨 Generando visualizaciones...")

    all_errors = []
    visualizations_created = 0

    with torch.no_grad():
        for idx, (images, landmarks, metadata) in enumerate(tqdm(test_loader, desc="Generando visualizaciones")):
            images = images.to(device)
            landmarks = landmarks.to(device)

            # Predicciones
            predictions = model(images)

            # Procesar cada imagen en el batch (batch_size=1, así que solo una)
            for i in range(images.size(0)):
                image = images[i]
                pred_landmarks = predictions[i].cpu().numpy()
                gt_landmarks = landmarks[i].cpu().numpy()

                # Calcular error
                error = compute_pixel_error(pred_landmarks, gt_landmarks)
                all_errors.append(error)

                # Extraer información de metadata
                category = "Unknown"
                original_filename = f"imagen_{idx+1:03d}"

                if metadata and len(metadata) > i:
                    try:
                        meta = metadata[i]
                        if isinstance(meta, dict):
                            # Usar categoría del metadata si está disponible
                            if 'category' in meta and meta['category']:
                                category = meta['category']

                            # Usar filename original si está disponible
                            if 'filename' in meta and meta['filename']:
                                original_filename = meta['filename']
                                # Si no hay categoría, extraerla del filename
                                if category == "Unknown":
                                    filename_str = str(original_filename)
                                    if 'COVID' in filename_str:
                                        category = 'COVID'
                                    elif 'Normal' in filename_str:
                                        category = 'Normal'
                                    elif 'Viral' in filename_str:
                                        category = 'Viral_Pneumonia'
                    except Exception as e:
                        print(f"Error procesando metadata para imagen {idx}: {e}")

                # Crear nombre de archivo descriptivo
                # Remover extensión si existe
                base_filename = str(original_filename).replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
                filename = f"{category}_{base_filename}"

                # Generar visualización
                saved_path = create_prediction_image(
                    image, gt_landmarks, pred_landmarks, error, filename, output_dir
                )

                visualizations_created += 1

    # Estadísticas finales
    all_errors = np.array(all_errors)
    mean_error = np.mean(all_errors)
    median_error = np.median(all_errors)
    std_error = np.std(all_errors)
    min_error = np.min(all_errors)
    max_error = np.max(all_errors)

    print("\n" + "=" * 70)
    print("📊 RESUMEN DE VISUALIZACIONES GENERADAS")
    print("=" * 70)
    print(f"🖼️ Visualizaciones creadas: {visualizations_created}")
    print(f"📁 Directorio de salida: {output_dir}")
    print(f"🎯 Error promedio: {mean_error:.2f}px")
    print(f"📊 Error mediano: {median_error:.2f}px")
    print(f"📈 Desviación estándar: {std_error:.2f}px")
    print(f"🔽 Error mínimo: {min_error:.2f}px")
    print(f"🔼 Error máximo: {max_error:.2f}px")

    # Distribución de errores
    excellent = np.sum(all_errors < 5.0)
    very_good = np.sum((all_errors >= 5.0) & (all_errors < 8.5))
    good = np.sum((all_errors >= 8.5) & (all_errors < 15.0))
    acceptable = np.sum(all_errors >= 15.0)

    print(f"\n📈 DISTRIBUCIÓN DE CALIDAD:")
    print(f"   Excelente (<5px): {excellent} ({excellent/len(all_errors)*100:.1f}%)")
    print(f"   Muy bueno (5-8.5px): {very_good} ({very_good/len(all_errors)*100:.1f}%)")
    print(f"   Bueno (8.5-15px): {good} ({good/len(all_errors)*100:.1f}%)")
    print(f"   Aceptable (≥15px): {acceptable} ({acceptable/len(all_errors)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("✅ VISUALIZACIONES COMPLETADAS EXITOSAMENTE")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = visualize_complete_loss_test_set()
    if success:
        print("\n🎉 Todas las visualizaciones han sido generadas correctamente!")
    else:
        print("\n❌ Error al generar las visualizaciones")
        sys.exit(1)
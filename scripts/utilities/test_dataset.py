#!/usr/bin/env python3
"""
Script para probar el dataset y visualizar algunas muestras
"""

import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

# Agregar src al path
sys.path.append("src")

from data.dataset import create_dataloaders
from data.transforms import get_transforms

def test_dataset():
    """Probar carga de dataset"""
    print("🧪 Probando dataset...")

    # Configuración
    config = {
        'data': {
            'annotations_file': 'data/coordenadas/coordenadas_maestro.csv',
            'images_dir': 'data/dataset'
        },
        'split': {
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'random_seed': 42
        }
    }

    try:
        # Crear dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            annotations_file=config['data']['annotations_file'],
            images_dir=config['data']['images_dir'],
            batch_size=4,
            num_workers=0,  # 0 para debugging
            pin_memory=False,
            **config['split']
        )

        print(f"✓ Train batches: {len(train_loader)}")
        print(f"✓ Val batches: {len(val_loader)}")
        print(f"✓ Test batches: {len(test_loader)}")

        # Probar un batch
        print("\n🔍 Probando un batch...")
        batch = next(iter(train_loader))
        images, landmarks, metadata = batch

        print(f"✓ Images shape: {images.shape}")
        print(f"✓ Landmarks shape: {landmarks.shape}")
        print(f"✓ Batch size: {len(metadata['filename'])}")

        # Verificar rangos
        print(f"✓ Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"✓ Landmarks range: [{landmarks.min():.3f}, {landmarks.max():.3f}]")

        # Visualizar una muestra
        visualize_sample(images[0], landmarks[0], metadata['filename'][0])

        return True

    except Exception as e:
        print(f"❌ Error probando dataset: {e}")
        return False

def visualize_sample(image_tensor, landmarks_tensor, filename):
    """
    Visualizar una muestra del dataset

    Args:
        image_tensor: Tensor de imagen (3, 224, 224)
        landmarks_tensor: Tensor de landmarks (30,)
        filename: Nombre del archivo
    """
    print(f"\n📊 Visualizando muestra: {filename}")

    # Desnormalizar imagen (ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image_denorm = image_tensor * std + mean
    image_denorm = torch.clamp(image_denorm, 0, 1)

    # Convertir a numpy para matplotlib
    image_np = image_denorm.permute(1, 2, 0).numpy()

    # Desnormalizar landmarks
    landmarks_denorm = landmarks_tensor * 224  # Desnormalizar de [0,1] a [0,224]

    # Separar coordenadas x e y
    x_coords = landmarks_denorm[::2].numpy()   # Índices pares
    y_coords = landmarks_denorm[1::2].numpy()  # Índices impares

    # Crear visualización
    plt.figure(figsize=(10, 8))

    # Mostrar imagen
    plt.imshow(image_np)
    plt.scatter(x_coords, y_coords, c='red', s=30, alpha=0.8)

    # Numerar landmarks
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        plt.annotate(str(i+1), (x, y), xytext=(3, 3),
                    textcoords='offset points', fontsize=8, color='yellow',
                    weight='bold')

    plt.title(f'Muestra del Dataset: {filename}')
    plt.axis('off')

    # Guardar visualización
    save_path = f"test_sample_visualization.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualización guardada en: {save_path}")
    print(f"✓ Landmarks encontrados: {len(x_coords)}")
    print(f"✓ Rango X: [{x_coords.min():.1f}, {x_coords.max():.1f}]")
    print(f"✓ Rango Y: [{y_coords.min():.1f}, {y_coords.max():.1f}]")

if __name__ == "__main__":
    success = test_dataset()
    if success:
        print("\n🎉 Dataset funciona correctamente!")
    else:
        print("\n❌ Hay problemas con el dataset")
        sys.exit(1)
#!/usr/bin/env python3
"""
Script de visualización interactiva para el modelo de regresión de landmarks
Permite cargar el modelo y hacer predicciones en imágenes individuales
"""

import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, Dict
import argparse

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent))

from models.resnet_regressor import ResNetLandmarkRegressor
from data.transforms import get_transforms
from training.utils import setup_device, load_config


class LandmarkVisualizer:
    """
    Visualizador interactivo para predicciones de landmarks
    """

    def __init__(self, checkpoint_path: str, config_path: str):
        """
        Args:
            checkpoint_path: Ruta al checkpoint del modelo
            config_path: Ruta al archivo de configuración
        """
        self.checkpoint_path = checkpoint_path
        self.config = load_config(config_path)
        self.device = setup_device(
            use_gpu=self.config['device']['use_gpu'],
            gpu_id=self.config['device']['gpu_id']
        )

        self.model = None
        self.transform = None

        print("="*60)
        print("VISUALIZADOR DE LANDMARKS - MODELO REGRESSOR")
        print("="*60)

    def load_model(self):
        """
        Cargar modelo desde checkpoint
        """
        print(f"\n🔧 Cargando modelo desde: {self.checkpoint_path}")

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint no encontrado: {self.checkpoint_path}")

        # Cargar modelo
        self.model, checkpoint = ResNetLandmarkRegressor.load_from_checkpoint(
            self.checkpoint_path,
            map_location=self.device
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Configurar transformaciones
        self.transform = get_transforms(image_size=(224, 224), is_training=False)

        print("✓ Modelo cargado exitosamente")
        print(f"✓ Época del checkpoint: {checkpoint['epoch']}")
        print(f"✓ Landmarks por imagen: {self.model.num_landmarks}")

        return checkpoint

    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
        """
        Preprocesar imagen para inferencia

        Args:
            image_path: Ruta de la imagen

        Returns:
            Tupla de (tensor_procesado, imagen_original, dimensiones_originales)
        """
        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")

        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = original_image.shape[:2]

        # Crear landmarks dummy para el transform (no se usan en inferencia)
        dummy_landmarks = np.zeros(30)  # 15 landmarks * 2 coordenadas

        # Aplicar transformaciones
        image_tensor, _ = self.transform(original_image, dummy_landmarks)

        return image_tensor, original_image, (original_width, original_height)

    def predict_landmarks(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Predecir landmarks en una imagen

        Args:
            image_tensor: Tensor de imagen procesada

        Returns:
            Array de coordenadas predichas normalizadas [0,1]
        """
        with torch.no_grad():
            # Agregar dimensión de batch
            image_batch = image_tensor.unsqueeze(0).to(self.device)

            # Predicción
            predictions = self.model(image_batch)

            # Remover dimensión de batch y convertir a numpy
            landmarks = predictions.squeeze(0).cpu().numpy()

        return landmarks

    def denormalize_landmarks(self, normalized_landmarks: np.ndarray,
                            width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Desnormalizar landmarks de [0,1] a coordenadas de píxeles

        Args:
            normalized_landmarks: Landmarks normalizados
            width: Ancho de la imagen
            height: Alto de la imagen

        Returns:
            Tupla de (coordenadas_x, coordenadas_y)
        """
        # Desnormalizar
        landmarks = normalized_landmarks.copy()
        landmarks[::2] *= width   # Coordenadas X
        landmarks[1::2] *= height # Coordenadas Y

        # Separar coordenadas
        x_coords = landmarks[::2]
        y_coords = landmarks[1::2]

        return x_coords, y_coords

    def visualize_prediction(self, image_path: str, save_path: Optional[str] = None) -> Dict:
        """
        Visualizar predicción en una imagen

        Args:
            image_path: Ruta de la imagen
            save_path: Ruta donde guardar la visualización (opcional)

        Returns:
            Diccionario con resultados de la predicción
        """
        print(f"\n🔍 Procesando imagen: {Path(image_path).name}")

        # Preprocesar imagen
        image_tensor, original_image, (width, height) = self.preprocess_image(image_path)

        print(f"✓ Dimensiones originales: {width}x{height}")

        # Predecir landmarks
        normalized_landmarks = self.predict_landmarks(image_tensor)

        # Desnormalizar coordenadas
        x_coords, y_coords = self.denormalize_landmarks(normalized_landmarks, width, height)

        print(f"✓ {len(x_coords)} landmarks predichos")

        # Crear visualización
        plt.figure(figsize=(12, 8))

        # Mostrar imagen original
        plt.imshow(original_image)

        # Mostrar landmarks predichos
        plt.scatter(x_coords, y_coords, c='red', s=50, alpha=0.8, label='Landmarks Predichos')

        # Numerar landmarks
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            plt.annotate(str(i+1), (x, y), xytext=(3, 3),
                        textcoords='offset points', fontsize=10, color='yellow',
                        weight='bold', bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

        plt.title(f'Predicción de Landmarks - {Path(image_path).name}')
        plt.legend()
        plt.axis('off')

        # Guardar si se especifica ruta
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualización guardada en: {save_path}")

        plt.show()

        # Estadísticas de la predicción
        prediction_stats = {
            'image_path': image_path,
            'original_size': (width, height),
            'num_landmarks': len(x_coords),
            'x_coords': x_coords,
            'y_coords': y_coords,
            'normalized_landmarks': normalized_landmarks,
            'x_range': [x_coords.min(), x_coords.max()],
            'y_range': [y_coords.min(), y_coords.max()]
        }

        print(f"✓ Rango X: [{x_coords.min():.1f}, {x_coords.max():.1f}]")
        print(f"✓ Rango Y: [{y_coords.min():.1f}, {y_coords.max():.1f}]")

        return prediction_stats

    def batch_visualize(self, image_dir: str, output_dir: str, max_images: int = 10):
        """
        Visualizar predicciones en múltiples imágenes

        Args:
            image_dir: Directorio con imágenes
            output_dir: Directorio de salida
            max_images: Número máximo de imágenes a procesar
        """
        print(f"\n📁 Procesando imágenes desde: {image_dir}")

        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Buscar imágenes
        image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        image_files = []

        for ext in image_extensions:
            image_files.extend(list(image_dir.glob(f"**/*{ext}")))

        image_files = image_files[:max_images]

        print(f"✓ {len(image_files)} imágenes encontradas")

        all_predictions = []

        for i, image_path in enumerate(image_files):
            print(f"\n--- Procesando {i+1}/{len(image_files)} ---")

            try:
                save_path = output_dir / f"prediction_{image_path.stem}.png"
                prediction_stats = self.visualize_prediction(str(image_path), str(save_path))
                all_predictions.append(prediction_stats)

            except Exception as e:
                print(f"❌ Error procesando {image_path.name}: {e}")
                continue

        print(f"\n✅ Procesamiento completo: {len(all_predictions)} imágenes exitosas")
        return all_predictions

    def compare_with_ground_truth(self, image_path: str, ground_truth_coords: np.ndarray,
                                save_path: Optional[str] = None):
        """
        Comparar predicción con ground truth

        Args:
            image_path: Ruta de la imagen
            ground_truth_coords: Coordenadas ground truth normalizadas
            save_path: Ruta donde guardar la comparación
        """
        print(f"\n⚖️ Comparando con ground truth: {Path(image_path).name}")

        # Predecir
        image_tensor, original_image, (width, height) = self.preprocess_image(image_path)
        predicted_coords = self.predict_landmarks(image_tensor)

        # Desnormalizar ambos
        pred_x, pred_y = self.denormalize_landmarks(predicted_coords, width, height)
        true_x, true_y = self.denormalize_landmarks(ground_truth_coords, width, height)

        # Calcular errores
        x_errors = np.abs(pred_x - true_x)
        y_errors = np.abs(pred_y - true_y)
        euclidean_errors = np.sqrt(x_errors**2 + y_errors**2)

        # Crear visualización comparativa
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Imagen con ground truth
        axes[0].imshow(original_image)
        axes[0].scatter(true_x, true_y, c='green', s=50, alpha=0.8, label='Ground Truth')
        axes[0].set_title('Ground Truth')
        axes[0].legend()
        axes[0].axis('off')

        # Imagen con predicción
        axes[1].imshow(original_image)
        axes[1].scatter(pred_x, pred_y, c='red', s=50, alpha=0.8, label='Predicción')
        axes[1].set_title('Predicción')
        axes[1].legend()
        axes[1].axis('off')

        # Imagen con comparación
        axes[2].imshow(original_image)
        axes[2].scatter(true_x, true_y, c='green', s=50, alpha=0.8, label='Ground Truth')
        axes[2].scatter(pred_x, pred_y, c='red', s=50, alpha=0.8, label='Predicción')

        # Líneas conectando landmarks correspondientes
        for i in range(len(pred_x)):
            axes[2].plot([true_x[i], pred_x[i]], [true_y[i], pred_y[i]],
                        'b--', alpha=0.6, linewidth=1)

        axes[2].set_title('Comparación')
        axes[2].legend()
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Comparación guardada en: {save_path}")

        plt.show()

        # Estadísticas de error
        print(f"\n📊 Estadísticas de error:")
        print(f"  Error euclidiano promedio: {np.mean(euclidean_errors):.2f} píxeles")
        print(f"  Error euclidiano máximo: {np.max(euclidean_errors):.2f} píxeles")
        print(f"  Error euclidiano mínimo: {np.min(euclidean_errors):.2f} píxeles")
        print(f"  Error MAE X: {np.mean(x_errors):.2f} píxeles")
        print(f"  Error MAE Y: {np.mean(y_errors):.2f} píxeles")

        return {
            'euclidean_errors': euclidean_errors,
            'x_errors': x_errors,
            'y_errors': y_errors,
            'mean_euclidean_error': np.mean(euclidean_errors)
        }


def main():
    """
    Función principal del visualizador
    """
    parser = argparse.ArgumentParser(description='Visualizar predicciones de landmarks')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Ruta al checkpoint del modelo')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Ruta al archivo de configuración')
    parser.add_argument('--image', type=str,
                       help='Ruta a imagen individual para predecir')
    parser.add_argument('--image_dir', type=str,
                       help='Directorio con múltiples imágenes')
    parser.add_argument('--output_dir', type=str, default='visualization_results',
                       help='Directorio de salida para visualizaciones')
    parser.add_argument('--max_images', type=int, default=10,
                       help='Número máximo de imágenes a procesar (para batch)')

    args = parser.parse_args()

    # Verificar argumentos
    if not args.image and not args.image_dir:
        print("❌ Especifica --image o --image_dir")
        return

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint no encontrado: {args.checkpoint}")
        return

    # Crear visualizador
    visualizer = LandmarkVisualizer(args.checkpoint, args.config)
    visualizer.load_model()

    try:
        if args.image:
            # Imagen individual
            if not os.path.exists(args.image):
                print(f"❌ Imagen no encontrada: {args.image}")
                return

            output_path = Path(args.output_dir) / f"prediction_{Path(args.image).stem}.png"
            output_path.parent.mkdir(exist_ok=True)

            visualizer.visualize_prediction(args.image, str(output_path))

        elif args.image_dir:
            # Múltiples imágenes
            if not os.path.exists(args.image_dir):
                print(f"❌ Directorio no encontrado: {args.image_dir}")
                return

            visualizer.batch_visualize(args.image_dir, args.output_dir, args.max_images)

        print("\n🎉 Visualización completada exitosamente!")

    except Exception as e:
        print(f"\n❌ Error durante la visualización: {e}")
        raise


if __name__ == "__main__":
    main()
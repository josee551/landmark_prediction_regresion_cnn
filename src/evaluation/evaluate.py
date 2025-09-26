#!/usr/bin/env python3
"""
Script de evaluación completa del modelo entrenado
Evalúa el modelo en el conjunto de test y genera métricas detalladas
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent))

from models.resnet_regressor import ResNetLandmarkRegressor
from data.dataset import create_dataloaders
from training.utils import LandmarkMetrics, setup_device, load_config


class ModelEvaluator:
    """
    Evaluador completo del modelo de regresión de landmarks
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

        # Crear directorio de resultados
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)

        self.model = None
        self.test_loader = None

        print("="*60)
        print("EVALUACIÓN DEL MODELO LANDMARK REGRESSOR")
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

        print("✓ Modelo cargado exitosamente")
        print(f"✓ Época del checkpoint: {checkpoint['epoch']}")
        print(f"✓ Pérdida del checkpoint: {checkpoint.get('loss', 'N/A')}")

        return checkpoint

    def setup_data(self):
        """
        Configurar dataset de test
        """
        print("\n📊 Configurando datos de test...")

        # Crear dataloaders
        _, _, self.test_loader = create_dataloaders(
            annotations_file=self.config['data']['annotations_file'],
            images_dir=self.config['data']['images_dir'],
            batch_size=self.config['training_phase1']['batch_size'],
            num_workers=self.config['device']['num_workers'],
            pin_memory=self.config['device']['pin_memory'],
            train_ratio=self.config['split']['train_ratio'],
            val_ratio=self.config['split']['val_ratio'],
            test_ratio=self.config['split']['test_ratio'],
            random_seed=self.config['split']['random_seed']
        )

        print(f"✓ Test batches: {len(self.test_loader)}")

    def evaluate_model(self) -> Dict:
        """
        Evaluar modelo en conjunto de test

        Returns:
            Diccionario con resultados de evaluación
        """
        print("\n🔍 Evaluando modelo en conjunto de test...")

        all_predictions = []
        all_targets = []
        all_metadata = []
        total_loss = 0.0

        criterion = torch.nn.MSELoss()

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Evaluando")

            for batch_idx, (images, landmarks, metadata) in enumerate(pbar):
                # Mover datos al dispositivo
                images = images.to(self.device)
                landmarks = landmarks.to(self.device)

                # Forward pass
                predictions = self.model(images)

                # Calcular pérdida
                loss = criterion(predictions, landmarks)
                total_loss += loss.item()

                # Guardar resultados
                all_predictions.append(predictions.cpu())
                all_targets.append(landmarks.cpu())
                all_metadata.extend([
                    {
                        'filename': metadata['filename'][i],
                        'category': metadata['category'][i],
                        'image_path': metadata['image_path'][i]
                    } for i in range(len(metadata['filename']))
                ])

                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        # Concatenar todos los resultados
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calcular métricas
        avg_loss = total_loss / len(self.test_loader)
        metrics = LandmarkMetrics.calculate_all_metrics(all_predictions, all_targets)

        # Convertir métricas a píxeles (imágenes son redimensionadas a 224x224)
        IMAGE_SIZE = 224
        rmse_pixels = metrics['rmse'] * IMAGE_SIZE
        mae_pixels = metrics['mae'] * IMAGE_SIZE
        euclidean_pixels = metrics['mean_euclidean_distance'] * IMAGE_SIZE

        print(f"\n📈 Resultados de evaluación:")
        print(f"  Loss: {avg_loss:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f} (normalizado) | {rmse_pixels:.2f} píxeles")
        print(f"  MAE: {metrics['mae']:.6f} (normalizado) | {mae_pixels:.2f} píxeles")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  Distancia Euclidiana Promedio: {metrics['mean_euclidean_distance']:.6f} (normalizado) | {euclidean_pixels:.2f} píxeles")

        # Evaluación clínica
        print(f"\n🏥 Evaluación clínica:")
        if euclidean_pixels < 5:
            clinical_assessment = "EXCELENTE - Precisión sub-píxel"
        elif euclidean_pixels < 10:
            clinical_assessment = "MUY BUENA - Clínicamente aceptable"
        elif euclidean_pixels < 15:
            clinical_assessment = "BUENA - Útil para análisis general"
        else:
            clinical_assessment = "REGULAR - Necesita mejoras"
        print(f"  Precisión clínica: {clinical_assessment}")
        print(f"  Error promedio por landmark: {euclidean_pixels:.2f} píxeles")

        return {
            'predictions': all_predictions,
            'targets': all_targets,
            'metadata': all_metadata,
            'loss': avg_loss,
            'metrics': metrics
        }

    def analyze_per_landmark_performance(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict:
        """
        Analizar rendimiento por landmark individual

        Args:
            predictions: Predicciones del modelo
            targets: Valores verdaderos

        Returns:
            Diccionario con análisis por landmark
        """
        print("\n🎯 Analizando rendimiento por landmark...")

        # Calcular métricas por landmark
        euclidean_distances = LandmarkMetrics.euclidean_distance_per_landmark(predictions, targets)

        # Crear DataFrame para análisis
        landmark_analysis = []
        for i, distance in enumerate(euclidean_distances):
            # Coordenadas X e Y para este landmark
            x_pred = predictions[:, i*2]
            y_pred = predictions[:, i*2 + 1]
            x_true = targets[:, i*2]
            y_true = targets[:, i*2 + 1]

            # Errores por coordenada
            x_error = torch.abs(x_pred - x_true)
            y_error = torch.abs(y_pred - y_true)

            landmark_analysis.append({
                'landmark_id': i + 1,
                'euclidean_distance': distance,
                'x_mae': torch.mean(x_error).item(),
                'y_mae': torch.mean(y_error).item(),
                'x_std': torch.std(x_error).item(),
                'y_std': torch.std(y_error).item()
            })

        df_landmarks = pd.DataFrame(landmark_analysis)

        # Mostrar estadísticas
        print(f"Landmark con mejor rendimiento: {df_landmarks.loc[df_landmarks['euclidean_distance'].idxmin(), 'landmark_id']}")
        print(f"Landmark con peor rendimiento: {df_landmarks.loc[df_landmarks['euclidean_distance'].idxmax(), 'landmark_id']}")

        return {
            'per_landmark_metrics': df_landmarks,
            'euclidean_distances': euclidean_distances
        }

    def analyze_per_category_performance(self, predictions: torch.Tensor, targets: torch.Tensor,
                                       metadata: List[Dict]) -> Dict:
        """
        Analizar rendimiento por categoría de imagen

        Args:
            predictions: Predicciones del modelo
            targets: Valores verdaderos
            metadata: Metadatos de las muestras

        Returns:
            Diccionario con análisis por categoría
        """
        print("\n🏷️ Analizando rendimiento por categoría...")

        # Agrupar por categoría
        categories = [meta['category'] for meta in metadata]
        unique_categories = list(set(categories))

        category_analysis = []
        for category in unique_categories:
            # Índices de esta categoría
            category_indices = [i for i, cat in enumerate(categories) if cat == category]

            if len(category_indices) > 0:
                # Predicciones y targets de esta categoría
                cat_predictions = predictions[category_indices]
                cat_targets = targets[category_indices]

                # Calcular métricas
                cat_metrics = LandmarkMetrics.calculate_all_metrics(cat_predictions, cat_targets)

                category_analysis.append({
                    'category': category,
                    'samples': len(category_indices),
                    'rmse': cat_metrics['rmse'],
                    'mae': cat_metrics['mae'],
                    'mse': cat_metrics['mse'],
                    'euclidean_distance': cat_metrics['mean_euclidean_distance']
                })

        df_categories = pd.DataFrame(category_analysis)

        # Mostrar estadísticas
        print("Rendimiento por categoría:")
        print(df_categories.to_string(index=False))

        # Mostrar análisis en píxeles por categoría
        IMAGE_SIZE = 224
        print(f"\n🏥 Análisis clínico por categoría (en píxeles):")
        for _, row in df_categories.iterrows():
            euclidean_pixels = row['euclidean_distance'] * IMAGE_SIZE
            mae_pixels = row['mae'] * IMAGE_SIZE

            if euclidean_pixels < 15:
                assessment = "Excelente precisión"
            elif euclidean_pixels < 20:
                assessment = "Buena precisión"
            else:
                assessment = "Precisión moderada"

            print(f"  {row['category']:15}: {euclidean_pixels:5.2f} píxeles | {mae_pixels:5.2f} MAE | {assessment}")

        return {
            'per_category_metrics': df_categories
        }

    def create_visualizations(self, evaluation_results: Dict, landmark_analysis: Dict,
                            category_analysis: Dict):
        """
        Crear visualizaciones de los resultados

        Args:
            evaluation_results: Resultados de evaluación general
            landmark_analysis: Análisis por landmark
            category_analysis: Análisis por categoría
        """
        print("\n📊 Creando visualizaciones...")

        # Configurar estilo
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Gráfico de rendimiento por landmark
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Distancia euclidiana por landmark
        df_landmarks = landmark_analysis['per_landmark_metrics']
        axes[0, 0].bar(df_landmarks['landmark_id'], df_landmarks['euclidean_distance'])
        axes[0, 0].set_title('Distancia Euclidiana por Landmark')
        axes[0, 0].set_xlabel('Landmark ID')
        axes[0, 0].set_ylabel('Distancia Euclidiana')
        axes[0, 0].grid(True, alpha=0.3)

        # Error MAE en X e Y por landmark
        x_pos = np.arange(len(df_landmarks))
        width = 0.35
        axes[0, 1].bar(x_pos - width/2, df_landmarks['x_mae'], width, label='MAE X', alpha=0.8)
        axes[0, 1].bar(x_pos + width/2, df_landmarks['y_mae'], width, label='MAE Y', alpha=0.8)
        axes[0, 1].set_title('Error MAE por Coordenada y Landmark')
        axes[0, 1].set_xlabel('Landmark ID')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(df_landmarks['landmark_id'])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Rendimiento por categoría
        df_categories = category_analysis['per_category_metrics']
        axes[1, 0].bar(df_categories['category'], df_categories['rmse'])
        axes[1, 0].set_title('RMSE por Categoría')
        axes[1, 0].set_xlabel('Categoría')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # Distribución de errores
        predictions = evaluation_results['predictions']
        targets = evaluation_results['targets']
        errors = torch.abs(predictions - targets).numpy().flatten()

        axes[1, 1].hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].set_title('Distribución de Errores Absolutos')
        axes[1, 1].set_xlabel('Error Absoluto')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].axvline(np.mean(errors), color='red', linestyle='--',
                          label=f'Media: {np.mean(errors):.4f}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'evaluation_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Visualizaciones de muestras con predicciones
        self.visualize_sample_predictions(evaluation_results)

        print(f"✓ Visualizaciones guardadas en: {self.results_dir}")

    def visualize_sample_predictions(self, evaluation_results: Dict, num_samples: int = 8):
        """
        Visualizar muestras con predicciones vs ground truth

        Args:
            evaluation_results: Resultados de evaluación
            num_samples: Número de muestras a visualizar
        """
        print("\n🖼️ Creando visualizaciones de muestras...")

        predictions = evaluation_results['predictions']
        targets = evaluation_results['targets']
        metadata = evaluation_results['metadata']

        # Seleccionar muestras aleatorias
        indices = np.random.choice(len(predictions), size=min(num_samples, len(predictions)), replace=False)

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for i, idx in enumerate(indices):
            if i >= len(axes):
                break

            # Cargar imagen original
            import cv2
            img_path = metadata[idx]['image_path']

            try:
                image = cv2.imread(img_path)
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Desnormalizar coordenadas (de [0,1] a píxeles)
                    h, w = image_rgb.shape[:2]

                    pred_coords = predictions[idx] * torch.tensor([w, h] * 15)
                    true_coords = targets[idx] * torch.tensor([w, h] * 15)

                    # Separar coordenadas X e Y
                    pred_x = pred_coords[::2].numpy()
                    pred_y = pred_coords[1::2].numpy()
                    true_x = true_coords[::2].numpy()
                    true_y = true_coords[1::2].numpy()

                    # Mostrar imagen
                    axes[i].imshow(image_rgb)

                    # Mostrar landmarks verdaderos (verde) y predichos (rojo)
                    axes[i].scatter(true_x, true_y, c='green', s=30, alpha=0.8, label='Ground Truth')
                    axes[i].scatter(pred_x, pred_y, c='red', s=30, alpha=0.8, label='Predicción')

                    # Calcular error promedio para esta muestra
                    sample_error = torch.mean(torch.abs(predictions[idx] - targets[idx])).item()

                    axes[i].set_title(f'{metadata[idx]["category"]}\nError: {sample_error:.4f}')
                    axes[i].axis('off')

                    if i == 0:
                        axes[i].legend()

                else:
                    axes[i].text(0.5, 0.5, 'Imagen no\ndisponible', ha='center', va='center')
                    axes[i].axis('off')

            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error cargando\nimagen: {str(e)[:20]}...',
                           ha='center', va='center')
                axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'sample_predictions.png', dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_all_test_predictions(self, evaluation_results: Dict):
        """
        Visualizar predicciones vs ground truth para TODAS las imágenes del conjunto de test

        Args:
            evaluation_results: Resultados de evaluación con predicciones y targets
        """
        print("\n🖼️ Generando visualizaciones para todo el conjunto de test...")

        predictions = evaluation_results['predictions']
        targets = evaluation_results['targets']
        metadata = evaluation_results['metadata']

        # Crear directorio específico para visualizaciones del test
        test_viz_dir = self.results_dir / "test_predictions"
        test_viz_dir.mkdir(exist_ok=True)

        print(f"📁 Directorio de salida: {test_viz_dir}")
        print(f"🎯 Procesando {len(predictions)} imágenes del conjunto de test...")

        successful_visualizations = 0
        failed_visualizations = 0

        # Procesar cada imagen del conjunto de test
        for idx in tqdm(range(len(predictions)), desc="Generando visualizaciones"):
            try:
                # Cargar imagen original
                import cv2
                img_path = metadata[idx]['image_path']
                filename = metadata[idx]['filename']
                category = metadata[idx]['category']

                image = cv2.imread(img_path)
                if image is None:
                    print(f"⚠ No se pudo cargar imagen: {img_path}")
                    failed_visualizations += 1
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Desnormalizar coordenadas (de [0,1] a píxeles)
                h, w = image_rgb.shape[:2]

                pred_coords = predictions[idx] * torch.tensor([w, h] * 15)
                true_coords = targets[idx] * torch.tensor([w, h] * 15)

                # Separar coordenadas X e Y
                pred_x = pred_coords[::2].numpy()
                pred_y = pred_coords[1::2].numpy()
                true_x = true_coords[::2].numpy()
                true_y = true_coords[1::2].numpy()

                # Calcular error euclidiano promedio para esta imagen
                euclidean_errors = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
                mean_error = np.mean(euclidean_errors)

                # Crear visualización
                plt.figure(figsize=(12, 8))

                # Mostrar imagen
                plt.imshow(image_rgb)

                # Mostrar landmarks verdaderos (verde) y predichos (rojo)
                plt.scatter(true_x, true_y, c='green', s=40, alpha=0.8,
                           label='Ground Truth', marker='o', edgecolors='darkgreen', linewidths=1)
                plt.scatter(pred_x, pred_y, c='red', s=40, alpha=0.8,
                           label='Predicción', marker='s', edgecolors='darkred', linewidths=1)

                # Líneas conectoras (opcional - solo si el error es significativo)
                if mean_error > 5:  # Solo mostrar líneas si hay error > 5 píxeles
                    for i in range(len(pred_x)):
                        plt.plot([true_x[i], pred_x[i]], [true_y[i], pred_y[i]],
                                'b--', alpha=0.4, linewidth=0.8)

                # Numerar landmarks (pequeños para no saturar)
                for i, (x, y) in enumerate(zip(true_x, true_y)):
                    plt.annotate(str(i+1), (x, y), xytext=(2, 2),
                               textcoords='offset points', fontsize=8, color='white',
                               weight='bold', bbox=dict(boxstyle="round,pad=0.1",
                               facecolor="green", alpha=0.7))

                # Configurar título con información
                plt.title(f'{category} - {filename}\n'
                         f'Error promedio: {mean_error:.2f} píxeles | '
                         f'Error máximo: {np.max(euclidean_errors):.2f}px',
                         fontsize=12, fontweight='bold')

                plt.legend(loc='upper right', fontsize=10)
                plt.axis('off')

                # Nombre del archivo con información del error
                safe_filename = filename.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
                output_filename = f"{category}_{safe_filename}_error_{mean_error:.2f}px.png"
                output_path = test_viz_dir / output_filename

                # Guardar visualización
                plt.savefig(output_path, dpi=150, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                plt.close()

                successful_visualizations += 1

            except Exception as e:
                print(f"❌ Error procesando imagen {idx}: {e}")
                failed_visualizations += 1
                continue

        # Estadísticas finales
        print(f"\n✅ Visualizaciones completadas:")
        print(f"  • Exitosas: {successful_visualizations}")
        print(f"  • Fallidas: {failed_visualizations}")
        print(f"  • Total procesadas: {len(predictions)}")
        print(f"  • Directorio: {test_viz_dir}")

        # Crear archivo de resumen
        summary = {
            'total_images': len(predictions),
            'successful_visualizations': successful_visualizations,
            'failed_visualizations': failed_visualizations,
            'output_directory': str(test_viz_dir),
            'timestamp': pd.Timestamp.now().isoformat()
        }

        summary_path = test_viz_dir / "visualization_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"📋 Resumen guardado en: {summary_path}")

    def save_detailed_results(self, evaluation_results: Dict, landmark_analysis: Dict,
                            category_analysis: Dict, checkpoint_info: Dict):
        """
        Guardar resultados detallados en archivos

        Args:
            evaluation_results: Resultados de evaluación
            landmark_analysis: Análisis por landmark
            category_analysis: Análisis por categoría
            checkpoint_info: Información del checkpoint
        """
        print("\n💾 Guardando resultados detallados...")

        # Guardar métricas generales
        general_metrics = {
            'checkpoint_path': self.checkpoint_path,
            'checkpoint_epoch': checkpoint_info['epoch'],
            'test_loss': evaluation_results['loss'],
            **evaluation_results['metrics']
        }

        # Guardar en CSV
        pd.DataFrame([general_metrics]).to_csv(
            self.results_dir / 'general_metrics.csv', index=False
        )

        # Guardar análisis por landmark
        landmark_analysis['per_landmark_metrics'].to_csv(
            self.results_dir / 'per_landmark_metrics.csv', index=False
        )

        # Guardar análisis por categoría
        category_analysis['per_category_metrics'].to_csv(
            self.results_dir / 'per_category_metrics.csv', index=False
        )

        print(f"✓ Resultados guardados en: {self.results_dir}")

    def run_evaluation(self):
        """
        Ejecutar evaluación completa
        """
        print("\n🚀 Iniciando evaluación completa...")

        # Cargar modelo y datos
        checkpoint_info = self.load_model()
        self.setup_data()

        # Evaluación principal
        evaluation_results = self.evaluate_model()

        # Análisis detallados
        landmark_analysis = self.analyze_per_landmark_performance(
            evaluation_results['predictions'],
            evaluation_results['targets']
        )

        category_analysis = self.analyze_per_category_performance(
            evaluation_results['predictions'],
            evaluation_results['targets'],
            evaluation_results['metadata']
        )

        # Visualizaciones
        self.create_visualizations(evaluation_results, landmark_analysis, category_analysis)

        # Guardar resultados
        self.save_detailed_results(evaluation_results, landmark_analysis,
                                 category_analysis, checkpoint_info)

        print("\n✅ Evaluación completa finalizada!")
        return evaluation_results

    def run_test_visualization(self):
        """
        Ejecutar solo las visualizaciones completas del conjunto de test
        """
        print("\n🚀 Iniciando visualización completa del conjunto de test...")

        # Cargar modelo y datos
        checkpoint_info = self.load_model()
        self.setup_data()

        # Evaluación principal para obtener predicciones
        evaluation_results = self.evaluate_model()

        # Generar visualizaciones completas del test
        self.visualize_all_test_predictions(evaluation_results)

        print("\n✅ Visualización completa del test finalizada!")
        return evaluation_results


def main():
    """
    Función principal de evaluación
    """
    import argparse

    parser = argparse.ArgumentParser(description='Evaluar modelo de regresión de landmarks')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Ruta al checkpoint del modelo')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Ruta al archivo de configuración')

    args = parser.parse_args()

    # Verificar archivos
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint no encontrado: {args.checkpoint}")
        return

    if not os.path.exists(args.config):
        print(f"❌ Configuración no encontrada: {args.config}")
        return

    # Ejecutar evaluación
    evaluator = ModelEvaluator(args.checkpoint, args.config)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Evaluación de Ensemble Learning para Landmark Prediction
Evalúa múltiples modelos y combina sus predicciones para mejor rendimiento
"""

import os
import sys
import json
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
from evaluation.evaluate import ModelEvaluator


class EnsembleEvaluator:
    """
    Evaluador de Ensemble Learning para regresión de landmarks

    Estrategias de agregación:
    - mean: Promedio simple de predicciones
    - weighted_mean: Promedio ponderado por rendimiento individual
    - median: Mediana de predicciones (robusto a outliers)
    """

    def __init__(self, ensemble_dir: str, config_path: str):
        """
        Args:
            ensemble_dir: Directorio con modelos del ensemble
            config_path: Ruta al archivo de configuración
        """
        self.ensemble_dir = Path(ensemble_dir)
        self.config = load_config(config_path)
        self.device = setup_device(
            use_gpu=self.config['device']['use_gpu'],
            gpu_id=self.config['device']['gpu_id']
        )

        # Crear directorio de resultados ensemble
        self.results_dir = Path("evaluation_results") / "ensemble"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Variables del ensemble
        self.models = []
        self.model_checkpoints = []
        self.ensemble_metadata = {}
        self.test_loader = None

        print("="*70)
        print("EVALUACIÓN ENSEMBLE: BOOTSTRAP AGGREGATING")
        print("="*70)

    def load_ensemble_metadata(self):
        """
        Cargar metadatos del ensemble
        """
        metadata_path = self.ensemble_dir / "ensemble_metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata del ensemble no encontrada: {metadata_path}")

        with open(metadata_path, 'r') as f:
            self.ensemble_metadata = json.load(f)

        print(f"📋 Metadata del ensemble cargada:")
        print(f"  • Modelos solicitados: {self.ensemble_metadata['num_models_requested']}")
        print(f"  • Modelos completados: {self.ensemble_metadata['num_models_completed']}")
        print(f"  • Seeds utilizadas: {self.ensemble_metadata['random_seeds']}")

        if self.ensemble_metadata['num_models_completed'] < 3:
            print("⚠ Advertencia: Menos de 3 modelos disponibles - resultados pueden ser subóptimos")

    def load_ensemble_models(self):
        """
        Cargar todos los modelos del ensemble
        """
        print(f"\n🔧 Cargando modelos del ensemble desde: {self.ensemble_dir}")

        # Buscar checkpoints de modelos
        model_pattern = "model_*.pt"
        checkpoint_files = list(self.ensemble_dir.glob(model_pattern))

        if not checkpoint_files:
            raise FileNotFoundError(f"No se encontraron checkpoints en: {self.ensemble_dir}")

        print(f"📁 Encontrados {len(checkpoint_files)} checkpoints de modelo")

        # Cargar cada modelo
        successful_loads = 0
        for checkpoint_path in sorted(checkpoint_files):
            try:
                print(f"  Cargando: {checkpoint_path.name}")

                # Cargar modelo
                model, checkpoint = ResNetLandmarkRegressor.load_from_checkpoint(
                    str(checkpoint_path),
                    map_location=self.device
                )

                model = model.to(self.device)
                model.eval()

                self.models.append(model)
                self.model_checkpoints.append({
                    'path': str(checkpoint_path),
                    'checkpoint': checkpoint,
                    'seed': checkpoint.get('ensemble_info', {}).get('random_seed', 'unknown')
                })

                successful_loads += 1

            except Exception as e:
                print(f"  ❌ Error cargando {checkpoint_path.name}: {e}")

        if successful_loads == 0:
            raise RuntimeError("No se pudo cargar ningún modelo del ensemble")

        print(f"✓ {successful_loads} modelos cargados exitosamente")

        # Mostrar información de los modelos
        print(f"\n📊 Modelos en el ensemble:")
        for i, checkpoint_info in enumerate(self.model_checkpoints):
            seed = checkpoint_info['seed']
            epoch = checkpoint_info['checkpoint']['epoch']
            loss = checkpoint_info['checkpoint'].get('loss', 'N/A')
            print(f"  Modelo {i+1}: Seed {seed} | Época {epoch} | Loss {loss}")

    def setup_data(self):
        """
        Configurar dataset de test (mismo para todos los modelos)
        """
        print("\n📊 Configurando datos de test...")

        # Usar la misma configuración de datos para todos los modelos
        _, _, self.test_loader = create_dataloaders(
            annotations_file=self.config['data']['annotations_file'],
            images_dir=self.config['data']['images_dir'],
            batch_size=self.config['training_phase1']['batch_size'],
            num_workers=self.config['device']['num_workers'],
            pin_memory=self.config['device']['pin_memory'],
            train_ratio=self.config['split']['train_ratio'],
            val_ratio=self.config['split']['val_ratio'],
            test_ratio=self.config['split']['test_ratio'],
            random_seed=self.config['split']['random_seed']  # Usar seed consistente para test
        )

        print(f"✓ Test batches: {len(self.test_loader)}")

    def get_individual_predictions(self) -> List[torch.Tensor]:
        """
        Obtener predicciones de cada modelo individual

        Returns:
            Lista de tensores con predicciones de cada modelo
        """
        print("\n🔍 Obteniendo predicciones individuales...")

        all_model_predictions = []
        all_targets = []
        all_metadata = []

        for model_idx, model in enumerate(self.models):
            print(f"  Evaluando modelo {model_idx+1}/{len(self.models)}")

            model_predictions = []

            with torch.no_grad():
                for batch_idx, (images, landmarks, metadata) in enumerate(tqdm(self.test_loader, desc=f"Modelo {model_idx+1}", leave=False)):
                    # Mover datos al dispositivo
                    images = images.to(self.device)

                    # Forward pass
                    predictions = model(images)
                    model_predictions.append(predictions.cpu())

                    # Guardar targets y metadata solo una vez
                    if model_idx == 0:
                        all_targets.append(landmarks)
                        all_metadata.extend([
                            {
                                'filename': metadata['filename'][i],
                                'category': metadata['category'][i],
                                'image_path': metadata['image_path'][i]
                            } for i in range(len(metadata['filename']))
                        ])

            # Concatenar predicciones de este modelo
            model_predictions = torch.cat(model_predictions, dim=0)
            all_model_predictions.append(model_predictions)

        # Concatenar targets
        all_targets = torch.cat(all_targets, dim=0)

        print("✓ Predicciones individuales obtenidas")

        return all_model_predictions, all_targets, all_metadata

    def aggregate_predictions(self, individual_predictions: List[torch.Tensor],
                            aggregation_method: str = "mean") -> torch.Tensor:
        """
        Agregar predicciones de múltiples modelos

        Args:
            individual_predictions: Lista de predicciones de cada modelo
            aggregation_method: Método de agregación ('mean', 'weighted_mean', 'median')

        Returns:
            Predicciones agregadas del ensemble
        """
        print(f"\n🔄 Agregando predicciones usando método: {aggregation_method}")

        # Stack predicciones: [num_models, num_samples, num_landmarks*2]
        stacked_predictions = torch.stack(individual_predictions, dim=0)

        if aggregation_method == "mean":
            # Promedio simple
            ensemble_predictions = torch.mean(stacked_predictions, dim=0)

        elif aggregation_method == "median":
            # Mediana (robusto a outliers)
            ensemble_predictions = torch.median(stacked_predictions, dim=0)[0]

        elif aggregation_method == "weighted_mean":
            # Promedio ponderado por rendimiento individual
            # Calcular pesos basados en pérdida de validación de cada modelo
            weights = []
            for checkpoint_info in self.model_checkpoints:
                val_loss = checkpoint_info['checkpoint'].get('loss', 1.0)
                # Peso inversamente proporcional a la pérdida
                weight = 1.0 / (val_loss + 1e-8)
                weights.append(weight)

            # Normalizar pesos
            weights = torch.tensor(weights, dtype=torch.float32)
            weights = weights / torch.sum(weights)

            print(f"  Pesos calculados: {weights.tolist()}")

            # Aplicar pesos
            weights = weights.view(-1, 1, 1)  # [num_models, 1, 1]
            ensemble_predictions = torch.sum(stacked_predictions * weights, dim=0)

        else:
            raise ValueError(f"Método de agregación no soportado: {aggregation_method}")

        print("✓ Predicciones agregadas")
        return ensemble_predictions

    def evaluate_ensemble(self, aggregation_methods: List[str] = ["mean", "weighted_mean", "median"]) -> Dict:
        """
        Evaluar ensemble con diferentes métodos de agregación

        Args:
            aggregation_methods: Lista de métodos de agregación a evaluar

        Returns:
            Diccionario con resultados de evaluación
        """
        print("\n🎯 Evaluando ensemble...")

        # Obtener predicciones individuales
        individual_predictions, targets, metadata = self.get_individual_predictions()

        # Evaluar cada modelo individual
        individual_results = self.evaluate_individual_models(individual_predictions, targets)

        # Evaluar diferentes métodos de agregación
        ensemble_results = {}

        for method in aggregation_methods:
            print(f"\n--- Evaluando método: {method} ---")

            # Agregar predicciones
            ensemble_predictions = self.aggregate_predictions(individual_predictions, method)

            # Calcular métricas
            metrics = LandmarkMetrics.calculate_all_metrics(ensemble_predictions, targets)

            # Convertir a píxeles
            IMAGE_SIZE = 224
            rmse_pixels = metrics['rmse'] * IMAGE_SIZE
            mae_pixels = metrics['mae'] * IMAGE_SIZE
            euclidean_pixels = metrics['mean_euclidean_distance'] * IMAGE_SIZE

            # Evaluación clínica
            if euclidean_pixels < 5:
                clinical_assessment = "EXCELENTE - Precisión sub-píxel"
            elif euclidean_pixels < 10:
                clinical_assessment = "MUY BUENA - Clínicamente aceptable"
            elif euclidean_pixels < 15:
                clinical_assessment = "BUENA - Útil para análisis general"
            else:
                clinical_assessment = "REGULAR - Necesita mejoras"

            ensemble_results[method] = {
                'predictions': ensemble_predictions,
                'metrics': metrics,
                'rmse_pixels': rmse_pixels,
                'mae_pixels': mae_pixels,
                'euclidean_pixels': euclidean_pixels,
                'clinical_assessment': clinical_assessment
            }

            print(f"📈 Resultados {method}:")
            print(f"  RMSE: {rmse_pixels:.2f} píxeles")
            print(f"  MAE: {mae_pixels:.2f} píxeles")
            print(f"  Error promedio: {euclidean_pixels:.2f} píxeles")
            print(f"  Evaluación clínica: {clinical_assessment}")

        return {
            'individual_results': individual_results,
            'ensemble_results': ensemble_results,
            'targets': targets,
            'metadata': metadata
        }

    def evaluate_individual_models(self, individual_predictions: List[torch.Tensor],
                                 targets: torch.Tensor) -> List[Dict]:
        """
        Evaluar cada modelo individual del ensemble

        Args:
            individual_predictions: Lista de predicciones de cada modelo
            targets: Valores verdaderos

        Returns:
            Lista de diccionarios con métricas de cada modelo
        """
        print("\n📊 Evaluando modelos individuales...")

        individual_results = []

        for i, predictions in enumerate(individual_predictions):
            metrics = LandmarkMetrics.calculate_all_metrics(predictions, targets)

            # Convertir a píxeles
            IMAGE_SIZE = 224
            euclidean_pixels = metrics['mean_euclidean_distance'] * IMAGE_SIZE

            individual_results.append({
                'model_index': i + 1,
                'seed': self.model_checkpoints[i]['seed'],
                'metrics': metrics,
                'euclidean_pixels': euclidean_pixels
            })

            print(f"  Modelo {i+1} (seed {self.model_checkpoints[i]['seed']}): {euclidean_pixels:.2f} píxeles")

        return individual_results

    def calculate_uncertainty(self, individual_predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Calcular incertidumbre del ensemble basada en varianza entre modelos

        Args:
            individual_predictions: Lista de predicciones de cada modelo

        Returns:
            Tensor con incertidumbre por landmark
        """
        # Stack predicciones: [num_models, num_samples, num_landmarks*2]
        stacked_predictions = torch.stack(individual_predictions, dim=0)

        # Calcular desviación estándar entre modelos
        uncertainty = torch.std(stacked_predictions, dim=0)

        return uncertainty

    def create_ensemble_visualizations(self, evaluation_results: Dict):
        """
        Crear visualizaciones específicas del ensemble

        Args:
            evaluation_results: Resultados de evaluación del ensemble
        """
        print("\n📊 Creando visualizaciones del ensemble...")

        # Configurar estilo
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Comparación de métodos de agregación
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Métricas por método de agregación
        methods = list(evaluation_results['ensemble_results'].keys())
        rmse_values = [evaluation_results['ensemble_results'][m]['rmse_pixels'] for m in methods]
        mae_values = [evaluation_results['ensemble_results'][m]['mae_pixels'] for m in methods]
        euclidean_values = [evaluation_results['ensemble_results'][m]['euclidean_pixels'] for m in methods]

        # RMSE por método
        axes[0, 0].bar(methods, rmse_values, alpha=0.8)
        axes[0, 0].set_title('RMSE por Método de Agregación')
        axes[0, 0].set_ylabel('RMSE (píxeles)')
        axes[0, 0].grid(True, alpha=0.3)

        # Error euclidiano por método
        axes[0, 1].bar(methods, euclidean_values, alpha=0.8, color='orange')
        axes[0, 1].set_title('Error Promedio por Método de Agregación')
        axes[0, 1].set_ylabel('Error Euclidiano (píxeles)')
        axes[0, 1].grid(True, alpha=0.3)

        # Comparación individual vs ensemble
        individual_errors = [r['euclidean_pixels'] for r in evaluation_results['individual_results']]
        best_ensemble_error = min(euclidean_values)

        x_pos = range(len(individual_errors) + 1)
        all_errors = individual_errors + [best_ensemble_error]
        colors = ['lightblue'] * len(individual_errors) + ['red']
        labels = [f'Modelo {i+1}' for i in range(len(individual_errors))] + ['Ensemble']

        bars = axes[1, 0].bar(x_pos, all_errors, color=colors, alpha=0.8)
        axes[1, 0].set_title('Comparación Individual vs Ensemble')
        axes[1, 0].set_ylabel('Error Promedio (píxeles)')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(labels, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # Añadir línea de referencia para objetivo <10px
        axes[1, 0].axhline(y=10, color='green', linestyle='--', alpha=0.7, label='Objetivo <10px')
        axes[1, 0].legend()

        # Distribución de errores del mejor ensemble
        best_method = min(methods, key=lambda m: evaluation_results['ensemble_results'][m]['euclidean_pixels'])
        best_predictions = evaluation_results['ensemble_results'][best_method]['predictions']
        targets = evaluation_results['targets']

        errors = torch.abs(best_predictions - targets).numpy().flatten()

        axes[1, 1].hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].set_title(f'Distribución de Errores - {best_method}')
        axes[1, 1].set_xlabel('Error Absoluto')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].axvline(np.mean(errors), color='red', linestyle='--',
                          label=f'Media: {np.mean(errors):.4f}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'ensemble_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Visualización de incertidumbre
        self.visualize_uncertainty(evaluation_results)

        print(f"✓ Visualizaciones guardadas en: {self.results_dir}")

    def visualize_uncertainty(self, evaluation_results: Dict):
        """
        Visualizar incertidumbre del ensemble

        Args:
            evaluation_results: Resultados de evaluación
        """
        # Calcular incertidumbre
        individual_preds = [evaluation_results['ensemble_results']['mean']['predictions']]  # Placeholder
        # En implementación real, necesitarías guardar las predicciones individuales

        # Por ahora, crear visualización básica de variabilidad
        individual_results = evaluation_results['individual_results']
        errors = [r['euclidean_pixels'] for r in individual_results]

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Mostrar variabilidad entre modelos
        models = [f"Modelo {r['model_index']}" for r in individual_results]
        ax.bar(models, errors, alpha=0.7, color='lightcoral')
        ax.set_title('Variabilidad entre Modelos del Ensemble')
        ax.set_ylabel('Error Promedio (píxeles)')
        ax.grid(True, alpha=0.3)

        # Línea de promedio
        avg_error = np.mean(errors)
        ax.axhline(y=avg_error, color='blue', linestyle='--',
                  label=f'Promedio: {avg_error:.2f}px')
        ax.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'ensemble_uncertainty.png', dpi=150, bbox_inches='tight')
        plt.close()

    def save_ensemble_results(self, evaluation_results: Dict):
        """
        Guardar resultados detallados del ensemble

        Args:
            evaluation_results: Resultados de evaluación
        """
        print("\n💾 Guardando resultados del ensemble...")

        # Resumen general
        summary = {
            'num_models': len(self.models),
            'seeds_used': [info['seed'] for info in self.model_checkpoints],
            'ensemble_metadata': self.ensemble_metadata
        }

        # Resultados individuales
        individual_df = pd.DataFrame([
            {
                'model_index': r['model_index'],
                'seed': r['seed'],
                'euclidean_pixels': r['euclidean_pixels'],
                'rmse': r['metrics']['rmse'],
                'mae': r['metrics']['mae'],
                'mse': r['metrics']['mse']
            } for r in evaluation_results['individual_results']
        ])

        # Resultados del ensemble
        ensemble_df = pd.DataFrame([
            {
                'aggregation_method': method,
                'euclidean_pixels': results['euclidean_pixels'],
                'rmse_pixels': results['rmse_pixels'],
                'mae_pixels': results['mae_pixels'],
                'clinical_assessment': results['clinical_assessment'],
                'rmse_normalized': results['metrics']['rmse'],
                'mae_normalized': results['metrics']['mae'],
                'mse_normalized': results['metrics']['mse']
            } for method, results in evaluation_results['ensemble_results'].items()
        ])

        # Guardar archivos
        individual_df.to_csv(self.results_dir / 'individual_models_metrics.csv', index=False)
        ensemble_df.to_csv(self.results_dir / 'ensemble_methods_metrics.csv', index=False)

        # Guardar resumen en JSON
        with open(self.results_dir / 'ensemble_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"✓ Resultados guardados en: {self.results_dir}")

    def run_evaluation(self):
        """
        Ejecutar evaluación completa del ensemble
        """
        print("\n🚀 Iniciando evaluación del ensemble...")

        # Cargar componentes
        self.load_ensemble_metadata()
        self.load_ensemble_models()
        self.setup_data()

        # Evaluación principal
        evaluation_results = self.evaluate_ensemble()

        # Análisis y comparación
        best_method = min(
            evaluation_results['ensemble_results'].keys(),
            key=lambda m: evaluation_results['ensemble_results'][m]['euclidean_pixels']
        )
        best_ensemble_error = evaluation_results['ensemble_results'][best_method]['euclidean_pixels']

        # Mejor modelo individual
        best_individual_error = min(
            r['euclidean_pixels'] for r in evaluation_results['individual_results']
        )

        # Mostrar resumen
        print(f"\n" + "="*70)
        print("🎉 EVALUACIÓN ENSEMBLE COMPLETADA")
        print("="*70)
        print(f"📊 Mejor modelo individual: {best_individual_error:.2f} píxeles")
        print(f"🏆 Mejor ensemble ({best_method}): {best_ensemble_error:.2f} píxeles")
        print(f"📈 Mejora del ensemble: {best_individual_error - best_ensemble_error:.2f} píxeles")
        print(f"🎯 Estado del objetivo (<10px): {'✅ ALCANZADO' if best_ensemble_error < 10 else '⚠ NO ALCANZADO'}")

        # Visualizaciones
        self.create_ensemble_visualizations(evaluation_results)

        # Guardar resultados
        self.save_ensemble_results(evaluation_results)

        print(f"\n✅ Evaluación ensemble finalizada!")
        print(f"📁 Resultados en: {self.results_dir}")

        return evaluation_results


def main():
    """
    Función principal de evaluación ensemble
    """
    import argparse

    parser = argparse.ArgumentParser(description='Evaluar ensemble de modelos de landmarks')
    parser.add_argument('--ensemble_dir', type=str, default='checkpoints/ensemble',
                       help='Directorio con modelos del ensemble')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Ruta al archivo de configuración')

    args = parser.parse_args()

    # Verificar archivos
    if not os.path.exists(args.ensemble_dir):
        print(f"❌ Directorio ensemble no encontrado: {args.ensemble_dir}")
        print("💡 Ejecuta primero: python main.py train_ensemble")
        return

    if not os.path.exists(args.config):
        print(f"❌ Configuración no encontrada: {args.config}")
        return

    # Ejecutar evaluación
    evaluator = EnsembleEvaluator(args.ensemble_dir, args.config)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
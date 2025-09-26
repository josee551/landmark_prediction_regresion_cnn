"""
Utilidades para entrenamiento de modelos de regresión de landmarks
"""

import torch
import torch.nn as nn
import numpy as np
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Importar nuevas funciones de pérdida geométricas
from ..models.losses import create_loss_function, CompleteLandmarkLoss
from ..models.geometric_utils import GeometricAnalyzer, compute_anatomical_consistency_score


class LandmarkLoss(nn.Module):
    """
    Función de pérdida especializada para regresión de landmarks

    Combina MSE Loss con regularización opcional
    """

    def __init__(self, loss_type: str = "mse", reduction: str = "mean"):
        """
        Args:
            loss_type: Tipo de pérdida ("mse", "l1", "smooth_l1")
            reduction: Tipo de reducción ("mean", "sum", "none")
        """
        super(LandmarkLoss, self).__init__()
        self.loss_type = loss_type.lower()
        self.reduction = reduction

        if self.loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif self.loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction=reduction)
        elif self.loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss(reduction=reduction)
        else:
            raise ValueError(f"Tipo de pérdida no soportado: {loss_type}")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcular pérdida entre predicciones y targets

        Args:
            predictions: Coordenadas predichas (batch_size, num_coords)
            targets: Coordenadas verdaderas (batch_size, num_coords)

        Returns:
            Pérdida calculada
        """
        return self.loss_fn(predictions, targets)


class LandmarkMetrics:
    """
    Métricas para evaluación de regresión de landmarks
    """

    @staticmethod
    def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Root Mean Square Error

        Args:
            predictions: Predicciones del modelo
            targets: Valores verdaderos

        Returns:
            RMSE
        """
        mse = torch.mean((predictions - targets) ** 2)
        return torch.sqrt(mse).item()

    @staticmethod
    def mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Mean Absolute Error

        Args:
            predictions: Predicciones del modelo
            targets: Valores verdaderos

        Returns:
            MAE
        """
        return torch.mean(torch.abs(predictions - targets)).item()

    @staticmethod
    def euclidean_distance_per_landmark(predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
        """
        Calcular distancia euclidiana promedio por landmark

        Args:
            predictions: Predicciones (batch_size, num_coords)
            targets: Targets (batch_size, num_coords)

        Returns:
            Array con distancia promedio por landmark
        """
        # Reshape para obtener coordenadas (x,y) por landmark
        pred_landmarks = predictions.view(-1, predictions.shape[1] // 2, 2)  # (batch, num_landmarks, 2)
        target_landmarks = targets.view(-1, targets.shape[1] // 2, 2)

        # Calcular distancia euclidiana por landmark
        distances = torch.sqrt(torch.sum((pred_landmarks - target_landmarks) ** 2, dim=2))  # (batch, num_landmarks)

        # Promedio por landmark a través del batch
        return torch.mean(distances, dim=0).cpu().numpy()

    @staticmethod
    def calculate_all_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Calcular todas las métricas

        Args:
            predictions: Predicciones del modelo
            targets: Valores verdaderos

        Returns:
            Diccionario con todas las métricas
        """
        metrics = {
            'rmse': LandmarkMetrics.rmse(predictions, targets),
            'mae': LandmarkMetrics.mae(predictions, targets),
            'mse': torch.mean((predictions - targets) ** 2).item()
        }

        # Agregar distancia euclidiana promedio
        euclidean_distances = LandmarkMetrics.euclidean_distance_per_landmark(predictions, targets)
        metrics['mean_euclidean_distance'] = np.mean(euclidean_distances)

        return metrics


class EarlyStopping:
    """
    Early stopping para evitar overfitting
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, monitor: str = "val_loss"):
        """
        Args:
            patience: Número de épocas sin mejora antes de parar
            min_delta: Cambio mínimo para considerar como mejora
            monitor: Métrica a monitorear
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_score = None
        self.epochs_without_improvement = 0
        self.should_stop = False

    def __call__(self, current_score: float) -> bool:
        """
        Verificar si se debe parar el entrenamiento

        Args:
            current_score: Score actual de la métrica monitoreada

        Returns:
            True si se debe parar, False en caso contrario
        """
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score - self.min_delta:
            # Mejora detectada
            self.best_score = current_score
            self.epochs_without_improvement = 0
        else:
            # No hay mejora
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            return True

        return False


class TrainingLogger:
    """
    Logger para entrenamiento con TensorBoard y archivos de log
    """

    def __init__(self, log_dir: str, experiment_name: str):
        """
        Args:
            log_dir: Directorio base para logs
            experiment_name: Nombre del experimento
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name

        # Crear directorio de logs
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Configurar TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.experiment_dir))

        # Historial de métricas
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }

    def log_metrics(self, epoch: int, train_loss: float, val_loss: float,
                   train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """
        Registrar métricas de una época

        Args:
            epoch: Número de época
            train_loss: Pérdida de entrenamiento
            val_loss: Pérdida de validación
            train_metrics: Métricas de entrenamiento
            val_metrics: Métricas de validación
        """
        # TensorBoard logging
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)

        for metric_name, value in train_metrics.items():
            self.writer.add_scalar(f'Metrics/Train_{metric_name}', value, epoch)

        for metric_name, value in val_metrics.items():
            self.writer.add_scalar(f'Metrics/Val_{metric_name}', value, epoch)

        # Guardar en historial
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['train_metrics'].append(train_metrics)
        self.metrics_history['val_metrics'].append(val_metrics)

    def save_metrics_plot(self):
        """
        Guardar gráfico de evolución de métricas
        """
        epochs = range(1, len(self.metrics_history['train_loss']) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Pérdida
        axes[0, 0].plot(epochs, self.metrics_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(epochs, self.metrics_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss Evolution')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # RMSE
        train_rmse = [m.get('rmse', 0) for m in self.metrics_history['train_metrics']]
        val_rmse = [m.get('rmse', 0) for m in self.metrics_history['val_metrics']]
        axes[0, 1].plot(epochs, train_rmse, label='Train RMSE')
        axes[0, 1].plot(epochs, val_rmse, label='Val RMSE')
        axes[0, 1].set_title('RMSE Evolution')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # MAE
        train_mae = [m.get('mae', 0) for m in self.metrics_history['train_metrics']]
        val_mae = [m.get('mae', 0) for m in self.metrics_history['val_metrics']]
        axes[1, 0].plot(epochs, train_mae, label='Train MAE')
        axes[1, 0].plot(epochs, val_mae, label='Val MAE')
        axes[1, 0].set_title('MAE Evolution')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Distancia Euclidiana
        train_euclidean = [m.get('mean_euclidean_distance', 0) for m in self.metrics_history['train_metrics']]
        val_euclidean = [m.get('mean_euclidean_distance', 0) for m in self.metrics_history['val_metrics']]
        axes[1, 1].plot(epochs, train_euclidean, label='Train Euclidean')
        axes[1, 1].plot(epochs, val_euclidean, label='Val Euclidean')
        axes[1, 1].set_title('Mean Euclidean Distance Evolution')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Distance')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plot_path = self.experiment_dir / 'metrics_evolution.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Gráfico de métricas guardado en: {plot_path}")

    def close(self):
        """
        Cerrar el logger
        """
        self.writer.close()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Cargar configuración desde archivo YAML

    Args:
        config_path: Ruta al archivo de configuración

    Returns:
        Diccionario con configuración
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Guardar configuración en archivo YAML

    Args:
        config: Configuración a guardar
        save_path: Ruta donde guardar
    """
    with open(save_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, indent=2)


def setup_device(use_gpu: bool = True, gpu_id: int = 0) -> torch.device:
    """
    Configurar dispositivo de entrenamiento

    Args:
        use_gpu: Si usar GPU
        gpu_id: ID de la GPU

    Returns:
        Dispositivo PyTorch
    """
    if use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"✓ Usando GPU: {torch.cuda.get_device_name(gpu_id)}")
        print(f"✓ Memoria GPU disponible: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device('cpu')
        print("⚠ Usando CPU para entrenamiento")

    return device


class Timer:
    """
    Utilidad para medir tiempo de entrenamiento
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """Iniciar cronómetro"""
        self.start_time = time.time()

    def stop(self):
        """Parar cronómetro"""
        self.end_time = time.time()

    def elapsed(self) -> float:
        """
        Obtener tiempo transcurrido en segundos

        Returns:
            Tiempo transcurrido
        """
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def formatted_elapsed(self) -> str:
        """
        Obtener tiempo transcurrido formateado

        Returns:
            Tiempo formateado como string
        """
        elapsed = self.elapsed()
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class GeometricLandmarkMetrics:
    """
    Métricas geométricas especializadas para landmarks anatómicos

    Incluye análisis de simetría, coherencia anatómica y validación estructural
    """

    def __init__(self):
        self.geometric_analyzer = GeometricAnalyzer()

    def compute_all_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute todas las métricas geométricas y estándar

        Args:
            predictions: Coordenadas predichas (batch_size, 30)
            targets: Coordenadas verdaderas (batch_size, 30)

        Returns:
            Diccionario completo con todas las métricas
        """
        metrics = {}

        # Métricas estándar
        metrics['rmse'] = LandmarkMetrics.rmse(predictions, targets)
        metrics['mae'] = LandmarkMetrics.mae(predictions, targets)

        # Distancias por landmark en píxeles (asumir imagen 224x224)
        euclidean_distances = LandmarkMetrics.euclidean_distance_per_landmark(predictions, targets)
        pixel_distances = euclidean_distances * 224  # Convertir a píxeles

        metrics['pixel_error_mean'] = np.mean(pixel_distances)
        metrics['pixel_error_std'] = np.std(pixel_distances)
        metrics['pixel_error_max'] = np.max(pixel_distances)

        # Métricas geométricas avanzadas
        with torch.no_grad():
            geometric_metrics = self._compute_geometric_metrics(predictions, targets)
            metrics.update(geometric_metrics)

        return metrics

    def _compute_geometric_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute métricas geométricas específicas"""

        # Análisis geométrico de predicciones
        pred_analysis = self.geometric_analyzer.analyze_batch(predictions)
        target_analysis = self.geometric_analyzer.analyze_batch(targets)

        geometric_metrics = {
            # Simetría bilateral
            'symmetry_error': pred_analysis['symmetry_error_mean'].mean().item(),
            'symmetry_consistency': pred_analysis['bilateral_consistency'].mean().item(),

            # Validez anatómica
            'anatomical_validity': pred_analysis['anatomical_validity'].mean().item(),
            'constraint_violations': pred_analysis['total_violations'].mean().item(),

            # Métricas de forma
            'cardiothoracic_ratio': pred_analysis['cardiothoracic_ratio'].mean().item(),
            'thoracic_aspect_ratio': pred_analysis['aspect_ratio'].mean().item(),
            'mean_dispersion': pred_analysis['mean_dispersion'].mean().item(),

            # Preservación de distancias críticas
            'distance_preservation_error': self._compute_distance_preservation_error(
                pred_analysis, target_analysis
            )
        }

        return geometric_metrics

    def _compute_distance_preservation_error(
        self,
        pred_analysis: Dict,
        target_analysis: Dict
    ) -> float:
        """Compute error en preservación de distancias anatómicas"""

        distance_errors = []
        distance_names = [
            'distance_mediastino_superior_inferior',
            'distance_ancho_toracico_superior',
            'distance_ancho_toracico_medio'
        ]

        for name in distance_names:
            if name in pred_analysis and name in target_analysis:
                pred_dist = pred_analysis[name]
                target_dist = target_analysis[name]
                error = torch.mean(torch.abs(pred_dist - target_dist)).item()
                distance_errors.append(error)

        return np.mean(distance_errors) if distance_errors else 0.0


class EnhancedTrainingLogger(TrainingLogger):
    """
    Logger extendido con soporte para métricas geométricas
    """

    def __init__(self, log_dir: str, experiment_name: str):
        super().__init__(log_dir, experiment_name)

        # Métricas geométricas adicionales
        self.geometric_metrics_history = {
            'train_geometric': [],
            'val_geometric': []
        }

    def log_geometric_metrics(
        self,
        epoch: int,
        train_geometric: Dict[str, float],
        val_geometric: Dict[str, float]
    ):
        """
        Log métricas geométricas específicas

        Args:
            epoch: Número de época
            train_geometric: Métricas geométricas de entrenamiento
            val_geometric: Métricas geométricas de validación
        """
        # TensorBoard logging para métricas geométricas
        for metric_name, value in train_geometric.items():
            self.writer.add_scalar(f'Geometric/Train_{metric_name}', value, epoch)

        for metric_name, value in val_geometric.items():
            self.writer.add_scalar(f'Geometric/Val_{metric_name}', value, epoch)

        # Guardar en historial
        self.geometric_metrics_history['train_geometric'].append(train_geometric)
        self.geometric_metrics_history['val_geometric'].append(val_geometric)

    def save_geometric_metrics_plot(self):
        """Guardar gráficos específicos de métricas geométricas"""
        if not self.geometric_metrics_history['train_geometric']:
            return

        epochs = range(1, len(self.geometric_metrics_history['train_geometric']) + 1)

        # Crear subplot para métricas geométricas clave
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Métricas Geométricas - {self.experiment_name}')

        # 1. Simetría bilateral
        train_symmetry = [m.get('symmetry_consistency', 0)
                         for m in self.geometric_metrics_history['train_geometric']]
        val_symmetry = [m.get('symmetry_consistency', 0)
                       for m in self.geometric_metrics_history['val_geometric']]

        axes[0, 0].plot(epochs, train_symmetry, 'b-', label='Train')
        axes[0, 0].plot(epochs, val_symmetry, 'r-', label='Validation')
        axes[0, 0].set_title('Consistencia de Simetría Bilateral')
        axes[0, 0].set_xlabel('Épocas')
        axes[0, 0].set_ylabel('Score [0-1]')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 2. Validez anatómica
        train_validity = [m.get('anatomical_validity', 0)
                         for m in self.geometric_metrics_history['train_geometric']]
        val_validity = [m.get('anatomical_validity', 0)
                       for m in self.geometric_metrics_history['val_geometric']]

        axes[0, 1].plot(epochs, train_validity, 'b-', label='Train')
        axes[0, 1].plot(epochs, val_validity, 'r-', label='Validation')
        axes[0, 1].set_title('Validez Anatómica')
        axes[0, 1].set_xlabel('Épocas')
        axes[0, 1].set_ylabel('Score [0-1]')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 3. Error de simetría
        train_sym_error = [m.get('symmetry_error', 0)
                          for m in self.geometric_metrics_history['train_geometric']]
        val_sym_error = [m.get('symmetry_error', 0)
                        for m in self.geometric_metrics_history['val_geometric']]

        axes[1, 0].plot(epochs, train_sym_error, 'b-', label='Train')
        axes[1, 0].plot(epochs, val_sym_error, 'r-', label='Validation')
        axes[1, 0].set_title('Error de Simetría')
        axes[1, 0].set_xlabel('Épocas')
        axes[1, 0].set_ylabel('Error')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 4. Ratio cardiotorácico
        train_ct_ratio = [m.get('cardiothoracic_ratio', 0)
                         for m in self.geometric_metrics_history['train_geometric']]
        val_ct_ratio = [m.get('cardiothoracic_ratio', 0)
                       for m in self.geometric_metrics_history['val_geometric']]

        axes[1, 1].plot(epochs, train_ct_ratio, 'b-', label='Train')
        axes[1, 1].plot(epochs, val_ct_ratio, 'r-', label='Validation')
        axes[1, 1].set_title('Ratio Cardiotorácico')
        axes[1, 1].set_xlabel('Épocas')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()

        # Guardar gráfico
        plot_path = self.experiment_dir / 'geometric_metrics_evolution.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Gráfico de métricas geométricas guardado en: {plot_path}")


def create_enhanced_loss_function(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function para crear función de pérdida mejorada según configuración

    Args:
        config: Configuración completa del experimento

    Returns:
        Función de pérdida configurada (estándar o geométrica)
    """
    loss_config = config.get('loss', {})

    # Si se especifica usar funciones geométricas
    if loss_config.get('use_geometric', False):
        return create_loss_function(loss_config)
    else:
        # Usar función de pérdida legacy
        loss_type = loss_config.get('type', 'mse')
        return LandmarkLoss(loss_type=loss_type)


def validate_geometric_predictions(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    validity_threshold: float = 0.7
) -> Dict[str, float]:
    """
    Validar coherencia geométrica de las predicciones del modelo

    Args:
        model: Modelo entrenado
        dataloader: DataLoader con datos de validación
        device: Dispositivo de cómputo
        validity_threshold: Umbral de validez anatómica

    Returns:
        Diccionario con estadísticas de validación geométrica
    """
    model.eval()
    geometric_metrics = GeometricLandmarkMetrics()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (images, targets, _) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)

            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())

    # Concatenar todas las predicciones
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute métricas geométricas
    metrics = geometric_metrics.compute_all_metrics(all_predictions, all_targets)

    # Compute estadísticas de validez
    consistency_scores = compute_anatomical_consistency_score(all_predictions)
    valid_predictions = (consistency_scores >= validity_threshold).float()

    validation_stats = {
        'geometric_validity_rate': valid_predictions.mean().item(),
        'mean_consistency_score': consistency_scores.mean().item(),
        'std_consistency_score': consistency_scores.std().item(),
        'min_consistency_score': consistency_scores.min().item(),
        'max_consistency_score': consistency_scores.max().item()
    }

    # Combinar con métricas geométricas
    validation_stats.update(metrics)

    return validation_stats
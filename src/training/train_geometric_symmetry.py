"""
Entrenamiento Fase 3 con Symmetry Loss: Wing Loss + Bilateral Symmetry Constraints

Este script implementa:
- Combined loss: Wing Loss + Symmetry Loss
- Fine-tuning completo con learning rates diferenciados
- Symmetry enforcement usando mediastinal axis
- Objetivo: 10.91px → ≤9.3px con bilateral constraints
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import time

# Importar módulos del proyecto
from src.data.dataset import LandmarkDataset, create_dataloaders
from src.models.resnet_regressor import ResNetLandmarkRegressor
from src.models.losses import WingLoss, SymmetryLoss
from src.training.utils import (
    load_config, save_config, setup_device, Timer,
    GeometricLandmarkMetrics, EnhancedTrainingLogger,
    EarlyStopping, validate_geometric_predictions
)


def compute_pixel_error(predictions: torch.Tensor, targets: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """
    Compute pixel error between predictions and targets

    Args:
        predictions: Predicted landmarks (batch_size, 30)
        targets: Ground truth landmarks (batch_size, 30)
        image_size: Image size for denormalization

    Returns:
        Pixel errors for each sample (batch_size,)
    """
    # Reshape to (batch_size, 15, 2)
    pred_reshaped = predictions.view(-1, 15, 2)
    target_reshaped = targets.view(-1, 15, 2)

    # Compute euclidean distance per landmark
    distances = torch.norm(pred_reshaped - target_reshaped, dim=2)  # (batch_size, 15)

    # Convert to pixels and compute mean error per sample
    pixel_distances = distances * image_size
    mean_pixel_error = torch.mean(pixel_distances, dim=1)  # (batch_size,)

    return mean_pixel_error


class CombinedWingSymmetryLoss(nn.Module):
    """
    Combined loss function: Wing Loss + Symmetry Loss

    Combines the precision of Wing Loss with bilateral symmetry constraints
    for anatomically-aware landmark regression.
    """

    def __init__(self, wing_omega: float = 10.0, wing_epsilon: float = 2.0,
                 symmetry_weight: float = 0.3, use_mediastinal_axis: bool = True):
        """
        Args:
            wing_omega: Wing loss omega parameter
            wing_epsilon: Wing loss epsilon parameter
            symmetry_weight: Weight for symmetry loss component
            use_mediastinal_axis: Use anatomical mediastinal axis for symmetry
        """
        super(CombinedWingSymmetryLoss, self).__init__()

        self.wing_loss = WingLoss(omega=wing_omega, epsilon=wing_epsilon)
        self.symmetry_loss = SymmetryLoss(
            symmetry_weight=1.0,  # Will be weighted externally
            use_mediastinal_axis=use_mediastinal_axis
        )
        self.symmetry_weight = symmetry_weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor,
                return_components: bool = False) -> torch.Tensor:
        """
        Compute combined Wing + Symmetry loss

        Args:
            prediction: Predicted landmarks (batch_size, 30)
            target: Ground truth landmarks (batch_size, 30)
            return_components: Return individual loss components

        Returns:
            Combined loss (or tuple with components if return_components=True)
        """
        # 1. Wing Loss (main precision loss)
        wing_loss_value = self.wing_loss(prediction, target)

        # 2. Symmetry Loss (bilateral constraints)
        symmetry_loss_value = self.symmetry_loss(prediction)

        # 3. Combined loss
        total_loss = wing_loss_value + self.symmetry_weight * symmetry_loss_value

        if return_components:
            return total_loss, {
                'wing_loss': wing_loss_value.item(),
                'symmetry_loss': symmetry_loss_value.item(),
                'total_loss': total_loss.item()
            }
        else:
            return total_loss


class GeometricSymmetryTrainer:
    """
    Trainer especializado para Fase 3: Wing Loss + Symmetry Loss
    """

    def __init__(self, config_path: str):
        """
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = load_config(config_path)
        self.device = setup_device(
            use_gpu=self.config.get('device', {}).get('use_gpu', True),
            gpu_id=self.config.get('device', {}).get('gpu_id', 0)
        )

        # Configurar reproducibilidad
        self._setup_reproducibility()

        # Inicializar componentes
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.logger = None
        self.geometric_metrics = GeometricLandmarkMetrics()

        print("=" * 80)
        print("🚀 INICIANDO ENTRENAMIENTO FASE 3: WING LOSS + SYMMETRY LOSS")
        print("=" * 80)
        print(f"📋 Configuración cargada desde: {config_path}")
        print(f"🎯 Objetivo: Mejorar de 10.91px → ≤9.3px con symmetry constraints")
        print(f"⚡ Dispositivo: {self.device}")

    def _setup_reproducibility(self):
        """Configurar semillas para reproducibilidad"""
        seed = self.config.get('reproducibility', {}).get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if self.config.get('reproducibility', {}).get('deterministic', True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def setup_data_loaders(self):
        """Configurar data loaders"""
        print("\n📊 Configurando data loaders...")

        data_config = self.config['data']

        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            annotations_file=data_config['coordenadas_path'],
            images_dir=data_config['dataset_path'],
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory'],
            # Parámetros de split
            train_ratio=data_config['train_split'],
            val_ratio=data_config['val_split'],
            test_ratio=data_config['test_split'],
            random_seed=data_config['random_seed']
        )

        print(f"✓ Train samples: {len(self.train_loader.dataset)}")
        print(f"✓ Validation samples: {len(self.val_loader.dataset)}")
        print(f"✓ Test samples: {len(self.test_loader.dataset)}")

    def setup_model(self):
        """Cargar modelo desde checkpoint de Fase 2 (mejor modelo)"""
        print("\n🏗️ Cargando modelo desde mejor checkpoint (Phase 2)...")

        # Usar el mejor checkpoint: geometric_phase2_wing_loss.pt (10.91px)
        checkpoint_path = Path(self.config['checkpoints']['save_dir']) / "geometric_phase2_wing_loss.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint base no encontrado: {checkpoint_path}")

        # Cargar modelo desde checkpoint
        self.model, checkpoint = ResNetLandmarkRegressor.load_from_checkpoint(
            str(checkpoint_path), map_location=str(self.device)
        )
        self.model = self.model.to(self.device)

        # IMPORTANTE: Asegurar que backbone esté descongelado
        self.model.unfreeze_backbone()

        # Mostrar información del modelo
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"✓ Modelo cargado desde: {checkpoint_path}")
        print(f"✓ Época de checkpoint: {checkpoint.get('epoch', 'N/A')}")

        # Handle best_val_loss formatting
        best_val_loss = checkpoint.get('best_val_loss', None)
        if best_val_loss is not None:
            print(f"✓ Error de checkpoint: {best_val_loss:.4f}")
        else:
            print(f"✓ Error de checkpoint: N/A")

        print(f"✓ Parámetros totales: {total_params:,}")
        print(f"✓ Parámetros entrenables: {trainable_params:,}")

    def setup_loss_and_optimizer(self):
        """Configurar función de pérdida combinada y optimizador"""
        print("\n⚡ Configurando Wing Loss + Symmetry Loss combinado...")

        # Combined loss function
        loss_config = self.config.get('loss', {})
        self.criterion = CombinedWingSymmetryLoss(
            wing_omega=loss_config.get('wing_omega', 10.0),
            wing_epsilon=loss_config.get('wing_epsilon', 2.0),
            symmetry_weight=loss_config.get('symmetry_weight', 0.3),
            use_mediastinal_axis=True
        )

        print(f"✓ Wing Loss omega: {self.criterion.wing_loss.omega}")
        print(f"✓ Wing Loss epsilon: {self.criterion.wing_loss.epsilon}")
        print(f"✓ Symmetry weight: {self.criterion.symmetry_weight}")
        print(f"✓ Mediastinal axis: True")

        # Configurar optimizador con learning rates diferenciados
        training_config = self.config.get('training_symmetry', self.config.get('training_phase2', {}))

        # Obtener parámetros del backbone y cabeza
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if 'regression_head' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        # Configurar optimizador con learning rates diferenciados
        optimizer_type = training_config.get('optimizer', 'adam').lower()
        backbone_lr = training_config.get('backbone_lr', 0.00002)
        head_lr = training_config.get('head_lr', 0.0002)
        weight_decay = training_config.get('weight_decay', 0.00005)

        if optimizer_type == 'adam':
            self.optimizer = optim.Adam([
                {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
                {'params': head_params, 'lr': head_lr, 'weight_decay': weight_decay}
            ])
        else:
            raise ValueError(f"Optimizador no soportado: {optimizer_type}")

        print(f"✓ Optimizador: {optimizer_type}")
        print(f"✓ Backbone LR: {backbone_lr}")
        print(f"✓ Head LR: {head_lr}")
        print(f"✓ Weight decay: {weight_decay}")

        # Configurar scheduler
        scheduler_type = training_config.get('scheduler', 'cosine_annealing')
        epochs = training_config.get('epochs', 70)

        if scheduler_type == 'cosine_annealing':
            min_lr = training_config.get('min_lr', 0.000002)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=min_lr
            )
            print(f"✓ Scheduler: Cosine Annealing (min_lr: {min_lr})")
        else:
            self.scheduler = None
            print(f"✓ Scheduler: None")

    def setup_logging(self):
        """Configurar logging y métricas"""
        print("\n📊 Configurando logging...")

        logging_config = self.config.get('logging', {})

        # Configurar logger específico para symmetry training
        log_dir = Path(logging_config.get('log_dir', 'logs')) / "geometric_symmetry_phase3"

        self.logger = EnhancedTrainingLogger(
            log_dir=str(log_dir),
            experiment_name="symmetry_loss_training"
        )

        # Early stopping
        early_stopping_config = self.config.get('training_symmetry', {}).get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=early_stopping_config.get('patience', 15),
            min_delta=early_stopping_config.get('min_delta', 0.0005),
            monitor=early_stopping_config.get('monitor', 'val_loss')
        )

        print(f"✓ Log directory: {log_dir}")
        print(f"✓ Early stopping patience: {self.early_stopping.patience}")

    def train_epoch(self, epoch: int) -> dict:
        """Entrenar una época"""
        self.model.train()

        total_loss = 0.0
        total_wing_loss = 0.0
        total_symmetry_loss = 0.0
        total_pixel_error = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (images, landmarks, _) in enumerate(progress_bar):
            images = images.to(self.device)
            landmarks = landmarks.to(self.device)

            # Forward pass
            predictions = self.model(images)

            # Calcular loss con componentes
            loss, loss_components = self.criterion(predictions, landmarks, return_components=True)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Calcular métricas en píxeles
            pixel_errors = compute_pixel_error(predictions, landmarks, image_size=224)
            mean_pixel_error = torch.mean(pixel_errors)

            # Acumular métricas
            total_loss += loss.item()
            total_wing_loss += loss_components['wing_loss']
            total_symmetry_loss += loss_components['symmetry_loss']
            total_pixel_error += mean_pixel_error.item()

            # Actualizar progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Wing': f"{loss_components['wing_loss']:.4f}",
                'Sym': f"{loss_components['symmetry_loss']:.4f}",
                'Pixel': f"{mean_pixel_error.item():.2f}px"
            })

        # Métricas promedio de la época
        avg_metrics = {
            'train_loss': total_loss / num_batches,
            'train_wing_loss': total_wing_loss / num_batches,
            'train_symmetry_loss': total_symmetry_loss / num_batches,
            'train_pixel_error_mean': total_pixel_error / num_batches
        }

        return avg_metrics

    def validate_epoch(self, epoch: int) -> dict:
        """Validar una época"""
        self.model.eval()

        total_loss = 0.0
        total_wing_loss = 0.0
        total_symmetry_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, landmarks, _ in self.val_loader:
                images = images.to(self.device)
                landmarks = landmarks.to(self.device)

                predictions = self.model(images)
                loss, loss_components = self.criterion(predictions, landmarks, return_components=True)

                total_loss += loss.item()
                total_wing_loss += loss_components['wing_loss']
                total_symmetry_loss += loss_components['symmetry_loss']

                all_predictions.append(predictions.cpu())
                all_targets.append(landmarks.cpu())

        # Concatenar todas las predicciones
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calcular métricas geométricas completas
        geometric_metrics = self.geometric_metrics.compute_all_metrics(
            all_predictions, all_targets
        )

        # Métricas de validación
        val_metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'val_wing_loss': total_wing_loss / len(self.val_loader),
            'val_symmetry_loss': total_symmetry_loss / len(self.val_loader),
            **geometric_metrics
        }

        # Agregar alias para compatibility
        if 'pixel_error_mean' in geometric_metrics:
            val_metrics['val_pixel_error_mean'] = geometric_metrics['pixel_error_mean']

        return val_metrics

    def train(self):
        """Ejecutar entrenamiento completo"""
        print("\n🚀 Iniciando entrenamiento...")

        # Configurar todo
        self.setup_data_loaders()
        self.setup_model()
        self.setup_loss_and_optimizer()
        self.setup_logging()

        # Parámetros de entrenamiento
        training_config = self.config.get('training_symmetry', self.config.get('training_phase2', {}))
        epochs = training_config.get('epochs', 70)

        print(f"\n🎯 Configuración de entrenamiento:")
        print(f"✓ Épocas: {epochs}")
        print(f"✓ Batch size: {self.config['data']['batch_size']}")
        print(f"✓ Loss: Wing Loss + Symmetry Loss (weight: {self.criterion.symmetry_weight})")

        # Mejores métricas
        best_val_loss = float('inf')
        best_pixel_error = float('inf')
        best_epoch = 0

        # Timer para entrenamiento
        training_timer = Timer()
        training_timer.start()

        try:
            # Note: Anomaly detection disabled for performance
            # torch.autograd.set_detect_anomaly(True)

            for epoch in range(1, epochs + 1):
                epoch_timer = Timer()
                epoch_timer.start()

                # Entrenamiento
                train_metrics = self.train_epoch(epoch)

                # Validación
                val_metrics = self.validate_epoch(epoch)

                # Actualizar scheduler
                if self.scheduler:
                    self.scheduler.step()

                epoch_time = epoch_timer.stop()

                # Logging
                all_metrics = {**train_metrics, **val_metrics}
                self.logger.log_metrics(
                    epoch=epoch,
                    train_loss=train_metrics['train_loss'],
                    val_loss=val_metrics['val_loss'],
                    train_metrics=train_metrics,
                    val_metrics=val_metrics
                )

                # Guardar mejor modelo
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    best_pixel_error = val_metrics['val_pixel_error_mean']
                    best_epoch = epoch

                    # Guardar checkpoint
                    checkpoint_path = Path(self.config['checkpoints']['save_dir']) / "geometric_symmetry.pt"
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                        'best_val_loss': best_val_loss,
                        'best_pixel_error': best_pixel_error,
                        'config': self.config,
                        'geometric_metrics': val_metrics
                    }, checkpoint_path)

                # Early stopping
                if self.early_stopping(val_metrics['val_loss']):
                    print(f"\n⏹️ Early stopping activado en época {epoch}")
                    break

                # Imprimir progreso
                print(f"\nÉpoca {epoch}/{epochs} - {epoch_time:.1f}s")
                print(f"  Train Loss: {train_metrics['train_loss']:.4f} | Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"  Wing Loss: {train_metrics['train_wing_loss']:.4f} | Symmetry Loss: {train_metrics['train_symmetry_loss']:.4f}")
                print(f"  Pixel Error: {val_metrics['val_pixel_error_mean']:.2f}px | Best: {best_pixel_error:.2f}px (epoch {best_epoch})")
                print(f"  Symmetry Consistency: {val_metrics.get('val_symmetry_consistency', 0):.3f}")

        except KeyboardInterrupt:
            print("\n⚠️ Entrenamiento interrumpido por usuario")

        training_time = training_timer.stop()

        # Resultados finales
        print("\n" + "=" * 80)
        print("🎉 ENTRENAMIENTO COMPLETADO")
        print("=" * 80)
        print(f"⏱️ Tiempo total: {training_time:.1f}s")
        print(f"🏆 Mejor época: {best_epoch}")
        print(f"📊 Mejor error: {best_pixel_error:.2f}px")
        print(f"🎯 Objetivo alcanzado: {'✅ SÍ' if best_pixel_error <= 9.3 else '❌ NO'} (target: ≤9.3px)")

        return {
            'best_epoch': best_epoch,
            'best_pixel_error': best_pixel_error,
            'best_val_loss': best_val_loss,
            'training_time': training_time,
            'success': best_pixel_error <= 9.3
        }


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Entrenamiento Fase 3: Wing Loss + Symmetry Loss")
    parser.add_argument('--config', type=str, default='configs/config_geometric.yaml',
                       help='Ruta al archivo de configuración')

    args = parser.parse_args()

    try:
        trainer = GeometricSymmetryTrainer(args.config)
        results = trainer.train()

        print(f"\n✅ Entrenamiento finalizado exitosamente")
        print(f"📈 Resultado: {results['best_pixel_error']:.2f}px en época {results['best_epoch']}")

        return 0 if results['success'] else 1

    except Exception as e:
        print(f"\n❌ Error durante entrenamiento: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
"""
Entrenamiento Fase 2 de Feature Engineering Geométrico: Coordinate Attention

Este script implementa la segunda fase del plan de mejoras geométricas:
- Carga del checkpoint baseline de 10.91px (geometric_phase2_wing_loss.pt)
- Implementación de Coordinate Attention para mejor localización espacial
- Learning rates diferenciados optimizados para attention module
- Objetivo: Mejorar de 10.91px → 9.8px con attention espacial

Arquitectura:
Input → ResNet Backbone → Coordinate Attention → Global Avg Pool → Regression Head → Output
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
import warnings
from typing import Dict, Tuple, Optional

# Importar módulos del proyecto
from src.data.dataset import LandmarkDataset, create_dataloaders
from src.models.resnet_regressor import ResNetLandmarkRegressor, ResNetWithCoordinateAttention
from src.training.utils import (
    load_config, save_config, setup_device, Timer,
    GeometricLandmarkMetrics, EnhancedTrainingLogger,
    create_enhanced_loss_function, EarlyStopping,
    validate_geometric_predictions
)


class GeometricAttentionTrainer:
    """
    Trainer especializado para Fase 2: Coordinate Attention + Wing Loss

    Esta clase implementa:
    - Carga de checkpoint baseline (10.91px)
    - ResNetWithCoordinateAttention con attention espacial
    - Optimizador de 3 grupos con LRs diferenciados
    - Wing Loss + métricas geométricas
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
        self.baseline_model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.logger = None
        self.geometric_metrics = GeometricLandmarkMetrics()

        # Métricas de tracking
        self.baseline_error = 10.91  # Error del checkpoint a cargar
        self.target_error = 9.8      # Objetivo de la fase

        print("=" * 80)
        print("🎯 FASE 2: COORDINATE ATTENTION + WING LOSS")
        print("=" * 80)
        print(f"📋 Configuración cargada desde: {config_path}")
        print(f"📏 Baseline: {self.baseline_error:.2f}px (geometric_phase2_wing_loss.pt)")
        print(f"🎯 Objetivo: {self.target_error:.2f}px (mejora esperada: -{self.baseline_error - self.target_error:.2f}px)")
        print(f"⚡ Dispositivo: {self.device}")
        print(f"🧠 Arquitectura: ResNet-18 + Coordinate Attention + Wing Loss")

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

    def load_baseline_checkpoint(self) -> Dict:
        """Cargar checkpoint baseline y extraer información"""
        print("\n🔄 Cargando checkpoint baseline...")

        # Ruta al checkpoint de baseline
        checkpoint_path = Path(self.config['checkpoints']['save_dir']) / "geometric_phase2_wing_loss.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint baseline no encontrado: {checkpoint_path}\n"
                f"Ejecuta primero: python src/training/train_geometric_phase2.py"
            )

        # Cargar checkpoint
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
            print(f"✓ Checkpoint cargado desde: {checkpoint_path}")

            if 'metrics' in checkpoint and 'pixel_error_mean' in checkpoint['metrics']:
                actual_baseline = checkpoint['metrics']['pixel_error_mean']
                print(f"✓ Error baseline confirmado: {actual_baseline:.2f}px")
                self.baseline_error = actual_baseline

            print(f"✓ Época del checkpoint: {checkpoint.get('epoch', 'N/A')}")
            return checkpoint

        except Exception as e:
            raise RuntimeError(f"Error cargando checkpoint: {e}")

    def setup_model(self):
        """Configurar modelo con Coordinate Attention"""
        print("\n🏗️ Configurando modelo con Coordinate Attention...")

        # Cargar checkpoint baseline
        baseline_checkpoint = self.load_baseline_checkpoint()

        # Crear modelo con attention
        self.model = ResNetWithCoordinateAttention(
            num_landmarks=self.config['model']['num_landmarks'],
            pretrained=self.config['model']['pretrained'],
            freeze_backbone=False,  # Para fine-tuning completo
            dropout_rate=self.config['model']['dropout_rate'],
            use_coordinate_attention=True,  # Activar attention
            attention_reduction=self.config['model']['attention_reduction']
        )

        # Transferir pesos del modelo base (sin attention)
        self._transfer_weights_from_baseline(baseline_checkpoint)

        self.model = self.model.to(self.device)

        # Mostrar información del modelo
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        attention_params = sum(p.numel() for p in self.model.coordinate_attention.parameters())

        print(f"✓ Arquitectura: ResNetWithCoordinateAttention")
        print(f"✓ Parámetros totales: {total_params:,}")
        print(f"✓ Parámetros entrenables: {trainable_params:,}")
        print(f"✓ Parámetros de attention: {attention_params:,}")
        print(f"✓ Coordinate Attention habilitado: {self.model.use_coordinate_attention}")
        print(f"✓ Reduction ratio: {self.model.attention_reduction}")

    def _transfer_weights_from_baseline(self, baseline_checkpoint: Dict):
        """
        Transferir pesos del modelo baseline al modelo con attention

        Args:
            baseline_checkpoint: Checkpoint del modelo baseline
        """
        print("\n🔄 Transfiriendo pesos desde modelo baseline...")

        try:
            baseline_state = baseline_checkpoint['model_state_dict']
            current_state = self.model.state_dict()

            transferred_keys = []
            skipped_keys = []

            for key, value in baseline_state.items():
                if key in current_state and current_state[key].shape == value.shape:
                    current_state[key] = value
                    transferred_keys.append(key)
                else:
                    skipped_keys.append(key)

            # Cargar estado actualizado
            self.model.load_state_dict(current_state)

            print(f"✓ Parámetros transferidos: {len(transferred_keys)}")
            print(f"⚠️ Parámetros omitidos: {len(skipped_keys)}")

            if skipped_keys:
                print("  Omitidos (nuevos en attention):")
                for key in skipped_keys[:5]:  # Mostrar solo primeros 5
                    print(f"    - {key}")
                if len(skipped_keys) > 5:
                    print(f"    ... y {len(skipped_keys) - 5} más")

        except Exception as e:
            warnings.warn(f"Error en transferencia de pesos: {e}")
            print("⚠️ Iniciando desde cero (sin transferencia)")

    def setup_loss_and_optimizer(self):
        """Configurar función de pérdida y optimizador con 3 grupos LR"""
        print("\n⚡ Configurando función de pérdida y optimizador...")

        # Función de pérdida geométrica (Wing Loss)
        self.criterion = create_enhanced_loss_function(self.config)

        # Verificar Wing Loss
        loss_type = self.config['loss']['type']
        print(f"✓ Función de pérdida: {loss_type}")

        if hasattr(self.criterion, 'omega'):
            print(f"✓ Wing Loss - omega: {self.criterion.omega}, epsilon: {self.criterion.epsilon}")

        # Configurar optimizador con 3 grupos de learning rates
        self._setup_three_group_optimizer()

        # Scheduler
        training_config = self.config['training_phase2']
        if training_config['scheduler'] == 'cosine_annealing':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['epochs'],
                eta_min=training_config.get('min_lr', 0.000001)
            )
        else:
            self.scheduler = None

        print(f"✓ Optimizador: Adam con 3 grupos LR")
        if self.scheduler:
            print(f"✓ Scheduler: {training_config['scheduler']}")

    def _setup_three_group_optimizer(self):
        """Configurar optimizador con 3 grupos: backbone, attention, head"""

        # Learning rates diferenciados para attention
        backbone_lr = 0.00001   # Más conservador para backbone entrenado
        attention_lr = 0.0001   # LR medio para nuevo módulo
        head_lr = 0.0002        # Mantener LR exitoso para head

        # Separar parámetros por grupos
        backbone_params = []
        attention_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if 'coordinate_attention' in name:
                attention_params.append(param)
            elif 'regression_head' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        # Crear optimizador con 3 grupos
        self.optimizer = optim.Adam([
            {
                'params': backbone_params,
                'lr': backbone_lr,
                'weight_decay': self.config['training_phase2']['weight_decay']
            },
            {
                'params': attention_params,
                'lr': attention_lr,
                'weight_decay': self.config['training_phase2']['weight_decay'] * 0.5  # Menos regularización para nuevo módulo
            },
            {
                'params': head_params,
                'lr': head_lr,
                'weight_decay': self.config['training_phase2']['weight_decay']
            }
        ])

        print(f"  - Backbone LR: {backbone_lr:.2e} ({len(backbone_params)} grupos)")
        print(f"  - Attention LR: {attention_lr:.2e} ({len(attention_params)} grupos)")
        print(f"  - Head LR: {head_lr:.2e} ({len(head_params)} grupos)")

    def setup_logging(self):
        """Configurar logging"""
        print("\n📝 Configurando logging...")

        logging_config = self.config['logging']

        # Nombre específico para esta fase
        experiment_name = "geometric_attention_phase2"

        self.logger = EnhancedTrainingLogger(
            log_dir=logging_config['log_dir'],
            experiment_name=experiment_name
        )

        # Early stopping mejorado para attention
        self.early_stopping = EarlyStopping(
            patience=15,      # Más paciencia para convergencia de attention
            min_delta=0.0005, # Delta más fino
            monitor="val_loss"
        )

        print(f"✓ Logs guardados en: {self.logger.experiment_dir}")
        print(f"✓ Early stopping: patience=15, min_delta=0.0005")

    def train_epoch(self, epoch: int) -> Tuple[float, Dict]:
        """Entrenar una época"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        # Métricas geométricas para acumulación
        all_predictions = []
        all_targets = []

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:2d}/65 [Attention]",
            leave=False
        )

        for batch_idx, (images, targets, _) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            try:
                predictions = self.model(images)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️ GPU memory error en batch {batch_idx}, reduciendo batch size...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            # Calcular pérdida
            loss = self.criterion(predictions, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping para estabilidad de attention
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

            # Acumular predicciones para métricas geométricas
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(targets.detach().cpu())

            # Actualizar barra de progreso
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.6f}",
                'Avg': f"{total_loss / (batch_idx + 1):.6f}",
                'Mem': f"{torch.cuda.memory_reserved() / 1024**3:.1f}GB" if torch.cuda.is_available() else "N/A"
            })

        # Calcular métricas de entrenamiento
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        train_metrics = self.geometric_metrics.compute_all_metrics(all_predictions, all_targets)
        avg_loss = total_loss / num_batches

        return avg_loss, train_metrics

    def validate_epoch(self, epoch: int) -> Tuple[float, Dict]:
        """Validar una época"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets, _ in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(images)
                loss = self.criterion(predictions, targets)

                total_loss += loss.item()

                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

        # Calcular métricas de validación
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        val_metrics = self.geometric_metrics.compute_all_metrics(all_predictions, all_targets)
        avg_loss = total_loss / num_batches

        return avg_loss, val_metrics

    def compare_with_baseline(self, current_error: float) -> Dict:
        """Comparar rendimiento actual con baseline"""
        improvement = self.baseline_error - current_error
        improvement_percent = (improvement / self.baseline_error) * 100
        target_progress = (self.baseline_error - current_error) / (self.baseline_error - self.target_error) * 100

        return {
            'baseline_error': self.baseline_error,
            'current_error': current_error,
            'improvement': improvement,
            'improvement_percent': improvement_percent,
            'target_error': self.target_error,
            'target_progress': min(100.0, max(0.0, target_progress)),
            'target_achieved': current_error <= self.target_error
        }

    def train(self):
        """Ejecutar entrenamiento completo de Coordinate Attention"""
        print("\n🚀 Iniciando entrenamiento con Coordinate Attention...")

        # Configurar todos los componentes
        self.setup_data_loaders()
        self.setup_model()
        self.setup_loss_and_optimizer()
        self.setup_logging()

        # Métricas iniciales
        print("\n📊 Evaluación inicial con Coordinate Attention...")
        initial_val_loss, initial_val_metrics = self.validate_epoch(0)
        initial_comparison = self.compare_with_baseline(initial_val_metrics['pixel_error_mean'])

        print(f"📏 Error inicial: {initial_val_metrics['pixel_error_mean']:.2f}px")
        print(f"📊 Comparación con baseline:")
        print(f"  - Baseline: {initial_comparison['baseline_error']:.2f}px")
        print(f"  - Actual: {initial_comparison['current_error']:.2f}px")
        print(f"  - Diferencia: {initial_comparison['improvement']:+.2f}px")
        print(f"🎯 Progreso hacia objetivo: {initial_comparison['target_progress']:.1f}%")

        # Variables de tracking
        best_val_loss = float('inf')
        best_pixel_error = float('inf')
        best_comparison = None
        training_timer = Timer()
        training_timer.start()

        epochs = 65  # Más épocas para convergencia de attention

        try:
            for epoch in range(1, epochs + 1):
                # Entrenar
                train_loss, train_metrics = self.train_epoch(epoch)

                # Validar
                val_loss, val_metrics = self.validate_epoch(epoch)

                # Actualizar scheduler
                if self.scheduler:
                    self.scheduler.step()

                # Comparación con baseline
                current_comparison = self.compare_with_baseline(val_metrics['pixel_error_mean'])

                # Logging
                self.logger.log_metrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics
                )

                # Logging geométrico y de attention
                if self.config['logging']['log_geometric_metrics']:
                    self.logger.log_geometric_metrics(
                        epoch=epoch,
                        train_geometric=train_metrics,
                        val_geometric=val_metrics
                    )

                # Guardar mejor modelo
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_pixel_error = val_metrics['pixel_error_mean']
                    best_comparison = current_comparison

                    checkpoint_path = Path(self.config['checkpoints']['save_dir']) / "geometric_attention.pt"
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                    # Guardar checkpoint con información de attention
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': val_loss,
                        'metrics': val_metrics,
                        'comparison': current_comparison,
                        'config': self.config,
                        'architecture': 'ResNetWithCoordinateAttention',
                        'baseline_checkpoint': 'geometric_phase2_wing_loss.pt'
                    }, str(checkpoint_path))

                # Mostrar progreso
                if epoch % 5 == 0:
                    lrs = [group['lr'] for group in self.optimizer.param_groups]
                    print(f"\nÉpoca {epoch:3d}/{epochs}")
                    print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
                    print(f"  Error: {val_metrics['pixel_error_mean']:.2f}px "
                          f"(Best: {best_pixel_error:.2f}px)")
                    print(f"  Mejora vs baseline: {current_comparison['improvement']:+.2f}px "
                          f"({current_comparison['improvement_percent']:+.1f}%)")
                    print(f"  Progreso objetivo: {current_comparison['target_progress']:.1f}%")
                    print(f"  LR: Backbone={lrs[0]:.2e}, Attention={lrs[1]:.2e}, Head={lrs[2]:.2e}")
                    print(f"  Simetría: {val_metrics['symmetry_consistency']:.3f} | "
                          f"Validez: {val_metrics['anatomical_validity']:.3f}")

                # Early stopping
                if self.early_stopping(val_loss):
                    print(f"\n⏹️ Early stopping en época {epoch}")
                    break

        except KeyboardInterrupt:
            print("\n⚠️ Entrenamiento interrumpido por usuario")

        finally:
            training_timer.stop()
            total_time = training_timer.formatted_elapsed()

            print("\n" + "=" * 80)
            print("✅ ENTRENAMIENTO COORDINATE ATTENTION COMPLETADO")
            print("=" * 80)
            print(f"⏱️ Tiempo total: {total_time}")

            if best_comparison:
                print(f"📏 Mejor error: {best_pixel_error:.2f}px")
                print(f"📊 Comparación final:")
                print(f"  - Baseline: {best_comparison['baseline_error']:.2f}px")
                print(f"  - Mejor resultado: {best_comparison['current_error']:.2f}px")
                print(f"  - Mejora total: {best_comparison['improvement']:+.2f}px")
                print(f"  - Mejora porcentual: {best_comparison['improvement_percent']:+.1f}%")
                print(f"🎯 Progreso hacia objetivo: {best_comparison['target_progress']:.1f}%")

                if best_comparison['target_achieved']:
                    print("🎉 ¡OBJETIVO COORDINATE ATTENTION ALCANZADO!")
                else:
                    remaining = best_comparison['current_error'] - self.target_error
                    print(f"⚠️ Objetivo no alcanzado (faltan {remaining:.2f}px)")

            # Guardar resultados
            results = {
                'training_time': total_time,
                'best_pixel_error': float(best_pixel_error),
                'baseline_error': float(self.baseline_error),
                'target_error': float(self.target_error),
                'improvement': float(best_comparison['improvement'] if best_comparison else 0),
                'improvement_percent': float(best_comparison['improvement_percent'] if best_comparison else 0),
                'target_achieved': bool(best_comparison['target_achieved'] if best_comparison else False),
                'final_epoch': epoch,
                'architecture': 'ResNetWithCoordinateAttention'
            }

            results_path = self.logger.experiment_dir / 'attention_results.yaml'
            save_config(results, str(results_path))

            # Generar gráficos
            self.logger.save_metrics_plot()
            self.logger.save_geometric_metrics_plot()
            self.logger.close()


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Entrenamiento Fase 2: Coordinate Attention + Wing Loss')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config_geometric.yaml',
        help='Ruta al archivo de configuración'
    )

    args = parser.parse_args()

    # Verificar que existe el checkpoint baseline
    baseline_path = Path("checkpoints/geometric_phase2_wing_loss.pt")
    if not baseline_path.exists():
        print("❌ Error: Checkpoint baseline no encontrado")
        print(f"   Esperado: {baseline_path}")
        print("\n🔧 Ejecuta primero:")
        print("   python src/training/train_geometric_phase2.py")
        return

    # Crear trainer
    trainer = GeometricAttentionTrainer(args.config)

    # Entrenamiento completo
    trainer.train()

    print("\n🎯 FASE COORDINATE ATTENTION COMPLETADA")
    print("🔜 Siguiente fase: Symmetry Loss para mejorar consistencia bilateral")


if __name__ == "__main__":
    main()
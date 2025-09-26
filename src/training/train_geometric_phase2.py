"""
Entrenamiento Fase 2 con mejoras geométricas: Fine-tuning completo con Wing Loss

Este script implementa:
- Fine-tuning completo (backbone + cabeza)
- Wing Loss para precisión sub-píxel
- Learning rates diferenciados optimizados
- Partiendo del checkpoint de Fase 1
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
from src.training.utils import (
    load_config, save_config, setup_device, Timer,
    GeometricLandmarkMetrics, EnhancedTrainingLogger,
    create_enhanced_loss_function, EarlyStopping,
    validate_geometric_predictions
)


class GeometricPhase2Trainer:
    """
    Trainer especializado para Fase 2 con Wing Loss y fine-tuning completo
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
        print("🚀 INICIANDO ENTRENAMIENTO FASE 2: FINE-TUNING CON WING LOSS")
        print("=" * 80)
        print(f"📋 Configuración cargada desde: {config_path}")
        print(f"🎯 Objetivo: Mejorar de 11.34px → 10.5px con fine-tuning + Wing Loss")
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
        """Cargar modelo desde checkpoint de Fase 1"""
        print("\n🏗️ Cargando modelo desde Fase 1...")

        # Ruta al checkpoint de Fase 1
        checkpoint_path = Path(self.config['checkpoints']['save_dir']) / \
                         self.config['checkpoints']['phase1_name']

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint de Fase 1 no encontrado: {checkpoint_path}")

        # Cargar modelo desde checkpoint
        self.model, checkpoint = ResNetLandmarkRegressor.load_from_checkpoint(
            str(checkpoint_path), map_location=str(self.device)
        )
        self.model = self.model.to(self.device)

        # IMPORTANTE: Descongelar backbone para fine-tuning
        self.model.unfreeze_backbone()

        # Mostrar información del modelo
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"✓ Modelo cargado desde: {checkpoint_path}")
        print(f"✓ Época de checkpoint: {checkpoint.get('epoch', 'N/A')}")
        print(f"✓ Parámetros totales: {total_params:,}")
        print(f"✓ Parámetros entrenables: {trainable_params:,}")
        print(f"✓ Backbone descongelado: {trainable_params == total_params}")

    def setup_loss_and_optimizer(self):
        """Configurar función de pérdida y optimizador con LRs diferenciados"""
        print("\n⚡ Configurando función de pérdida y optimizador...")

        # Función de pérdida geométrica (Wing Loss)
        self.criterion = create_enhanced_loss_function(self.config)

        # Verificar que estamos usando Wing Loss
        loss_type = self.config['loss']['type']
        print(f"✓ Función de pérdida: {loss_type}")

        if hasattr(self.criterion, 'omega'):
            print(f"✓ Wing Loss - omega: {self.criterion.omega}, epsilon: {self.criterion.epsilon}")

        # Configurar optimizador con learning rates diferenciados para Fase 2
        training_config = self.config['training_phase2']

        # Separar parámetros de backbone y cabeza
        backbone_params = self.model.get_backbone_parameters()
        head_params = self.model.get_head_parameters()

        # Optimizador con LRs diferenciados
        self.optimizer = optim.Adam([
            {'params': backbone_params, 'lr': training_config['backbone_lr']},
            {'params': head_params, 'lr': training_config['head_lr']}
        ], weight_decay=training_config['weight_decay'])

        # Scheduler
        if training_config['scheduler'] == 'cosine_annealing':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['epochs'],
                eta_min=training_config['min_lr']
            )
        else:
            self.scheduler = None

        print(f"✓ Optimizador: Adam con LRs diferenciados")
        print(f"  - Backbone LR: {training_config['backbone_lr']}")
        print(f"  - Head LR: {training_config['head_lr']}")
        if self.scheduler:
            print(f"✓ Scheduler: {training_config['scheduler']}")

    def setup_logging(self):
        """Configurar logging"""
        print("\n📝 Configurando logging...")

        logging_config = self.config['logging']

        # Cambiar nombre del experimento para Fase 2
        experiment_name = "geometric_wing_loss_phase2"

        self.logger = EnhancedTrainingLogger(
            log_dir=logging_config['log_dir'],
            experiment_name=experiment_name
        )

        # Early stopping
        early_stopping_config = self.config['training_phase2']['early_stopping']
        self.early_stopping = EarlyStopping(
            patience=early_stopping_config['patience'],
            min_delta=early_stopping_config['min_delta'],
            monitor=early_stopping_config['monitor']
        )

        print(f"✓ Logs guardados en: {self.logger.experiment_dir}")

    def train_epoch(self, epoch: int):
        """Entrenar una época"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        # Métricas geométricas para acumulación
        all_predictions = []
        all_targets = []

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Época {epoch:2d}/{self.config['training_phase2']['epochs']}",
            leave=False
        )

        for batch_idx, (images, targets, _) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)

            # Calcular pérdida
            loss = self.criterion(predictions, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Acumular predicciones para métricas geométricas
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(targets.detach().cpu())

            # Actualizar barra de progreso
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.6f}",
                'Avg': f"{total_loss / (batch_idx + 1):.6f}"
            })

        # Calcular métricas de entrenamiento
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        train_metrics = self.geometric_metrics.compute_all_metrics(all_predictions, all_targets)
        avg_loss = total_loss / num_batches

        return avg_loss, train_metrics

    def validate_epoch(self, epoch: int):
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

    def train(self):
        """Ejecutar entrenamiento completo de Fase 2"""
        print("\n🚀 Iniciando entrenamiento Fase 2...")

        # Configurar todos los componentes
        self.setup_data_loaders()
        self.setup_model()
        self.setup_loss_and_optimizer()
        self.setup_logging()

        # Métricas iniciales
        print("\n📊 Evaluación inicial...")
        initial_val_loss, initial_val_metrics = self.validate_epoch(0)

        print(f"📏 Error inicial: {initial_val_metrics['pixel_error_mean']:.2f} píxeles")
        print(f"🎯 Objetivo Fase 2: 10.5 píxeles (-0.8px mejora desde 11.34px)")
        print(f"🔄 Simetría inicial: {initial_val_metrics['symmetry_consistency']:.3f}")
        print(f"🏥 Validez anatómica inicial: {initial_val_metrics['anatomical_validity']:.3f}")

        # Variables de tracking
        best_val_loss = float('inf')
        best_pixel_error = float('inf')
        training_timer = Timer()
        training_timer.start()

        epochs = self.config['training_phase2']['epochs']

        try:
            for epoch in range(1, epochs + 1):
                # Entrenar
                train_loss, train_metrics = self.train_epoch(epoch)

                # Validar
                val_loss, val_metrics = self.validate_epoch(epoch)

                # Actualizar scheduler
                if self.scheduler:
                    self.scheduler.step()

                # Logging
                self.logger.log_metrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics
                )

                # Logging geométrico
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

                    checkpoint_path = Path(self.config['checkpoints']['save_dir']) / \
                                    "geometric_phase2_wing_loss.pt"
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                    self.model.save_checkpoint(
                        filepath=str(checkpoint_path),
                        epoch=epoch,
                        optimizer_state=self.optimizer.state_dict(),
                        loss=val_loss,
                        metrics=val_metrics
                    )

                # Mostrar progreso
                if epoch % self.config['logging']['log_interval'] == 0:
                    backbone_lr = self.optimizer.param_groups[0]['lr']
                    head_lr = self.optimizer.param_groups[1]['lr']
                    print(f"\nÉpoca {epoch:3d}/{epochs}")
                    print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
                    print(f"  Error: {val_metrics['pixel_error_mean']:.2f}px "
                          f"(Best: {best_pixel_error:.2f}px)")
                    print(f"  Simetría: {val_metrics['symmetry_consistency']:.3f} | "
                          f"Validez: {val_metrics['anatomical_validity']:.3f}")
                    print(f"  LR: Backbone={backbone_lr:.2e}, Head={head_lr:.2e}")

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
            print("✅ ENTRENAMIENTO FASE 2 COMPLETADO")
            print("=" * 80)
            print(f"⏱️ Tiempo total: {total_time}")
            print(f"📏 Mejor error: {best_pixel_error:.2f} píxeles")
            print(f"🎯 Mejora lograda: {11.34 - best_pixel_error:.2f}px")

            if best_pixel_error <= 10.5:
                print("🎉 ¡OBJETIVO FASE 2 ALCANZADO!")
            else:
                print(f"⚠️ Objetivo no alcanzado (esperado ≤10.5px)")

            # Guardar configuración y resultados
            results = {
                'training_time': total_time,
                'best_pixel_error': float(best_pixel_error),
                'improvement': float(11.34 - best_pixel_error),
                'target_achieved': bool(best_pixel_error <= 10.5),
                'final_epoch': epoch,
                'baseline_error': 11.34
            }

            results_path = self.logger.experiment_dir / 'phase2_results.yaml'
            save_config(results, str(results_path))

            # Generar gráficos
            self.logger.save_metrics_plot()
            self.logger.save_geometric_metrics_plot()
            self.logger.close()


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Entrenamiento Fase 2 con fine-tuning + Wing Loss')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config_geometric.yaml',
        help='Ruta al archivo de configuración'
    )

    args = parser.parse_args()

    # Crear trainer
    trainer = GeometricPhase2Trainer(args.config)

    # Entrenamiento completo
    trainer.train()

    print("\n🎯 FASE 2 COMPLETADA - Mejoras geométricas implementadas")


if __name__ == "__main__":
    main()
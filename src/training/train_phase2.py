#!/usr/bin/env python3
"""
Entrenamiento Fase 2: Fine-tuning completo
Descongelar backbone y entrenar toda la red con learning rates diferenciados
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Agregar src al path para imports
sys.path.append(str(Path(__file__).parent.parent))

from models.resnet_regressor import ResNetLandmarkRegressor
from data.dataset import create_dataloaders
from training.utils import (
    LandmarkLoss, LandmarkMetrics, EarlyStopping, TrainingLogger,
    load_config, setup_device, Timer
)


class Phase2Trainer:
    """
    Entrenador para Fase 2: Fine-tuning completo

    En esta fase:
    1. Cargar modelo preentrenado de Fase 1
    2. Descongelar backbone completo
    3. Usar learning rates diferenciados (backbone: bajo, cabeza: alto)
    4. Entrenar más épocas con early stopping
    5. Usar CosineAnnealingLR para mejor convergencia
    """

    def __init__(self, config: dict, phase1_checkpoint: str):
        """
        Args:
            config: Configuración del entrenamiento
            phase1_checkpoint: Ruta al checkpoint de Fase 1
        """
        self.config = config
        self.phase1_checkpoint = phase1_checkpoint
        self.device = setup_device(
            use_gpu=config['device']['use_gpu'],
            gpu_id=config['device']['gpu_id']
        )

        # Crear directorios necesarios
        self.checkpoint_dir = Path(config['logging']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Inicializar componentes
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.logger = None
        self.early_stopping = None

        print("="*60)
        print("ENTRENAMIENTO FASE 2: FINE-TUNING COMPLETO")
        print("="*60)

    def setup_model(self):
        """
        Configurar modelo para Fase 2: cargar desde Fase 1 y descongelar
        """
        print(f"\n🔧 Cargando modelo desde Fase 1: {self.phase1_checkpoint}")

        if not os.path.exists(self.phase1_checkpoint):
            raise FileNotFoundError(f"Checkpoint de Fase 1 no encontrado: {self.phase1_checkpoint}")

        # Cargar modelo desde checkpoint de Fase 1
        self.model, checkpoint = ResNetLandmarkRegressor.load_from_checkpoint(
            self.phase1_checkpoint,
            map_location=self.device
        )

        # Descongelar backbone para fine-tuning
        self.model.unfreeze_backbone()

        # Mover a dispositivo
        self.model = self.model.to(self.device)

        print("✓ Modelo cargado y configurado para Fase 2")
        print(f"✓ Época inicial de Fase 1: {checkpoint['epoch']}")
        print(f"✓ Mejor pérdida de Fase 1: {checkpoint.get('loss', 'N/A')}")

        # Mostrar estadísticas de parámetros
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"✓ Parámetros totales: {total_params:,}")
        print(f"✓ Parámetros entrenables: {trainable_params:,}")

    def setup_data(self):
        """
        Configurar datasets y dataloaders para Fase 2
        """
        print("\n📊 Configurando datos...")

        # Crear dataloaders con batch size más pequeño para fine-tuning
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            annotations_file=self.config['data']['annotations_file'],
            images_dir=self.config['data']['images_dir'],
            batch_size=self.config['training_phase2']['batch_size'],
            num_workers=self.config['device']['num_workers'],
            pin_memory=self.config['device']['pin_memory'],
            train_ratio=self.config['split']['train_ratio'],
            val_ratio=self.config['split']['val_ratio'],
            test_ratio=self.config['split']['test_ratio'],
            random_seed=self.config['split']['random_seed']
        )

        print("✓ Datos configurados para Fase 2")

    def setup_training_components(self):
        """
        Configurar optimizador con learning rates diferenciados, scheduler, etc.
        """
        print("\n⚙️ Configurando componentes de entrenamiento para Fase 2...")

        # Función de pérdida
        self.criterion = LandmarkLoss(
            loss_type=self.config['loss']['type'],
            reduction=self.config['loss']['reduction']
        )

        # Optimizador con learning rates diferenciados
        # Backbone: learning rate bajo
        # Cabeza: learning rate más alto
        backbone_lr = self.config['training_phase2']['backbone_lr']
        head_lr = self.config['training_phase2']['head_lr']

        param_groups = [
            {
                'params': list(self.model.get_backbone_parameters()),
                'lr': backbone_lr,
                'name': 'backbone'
            },
            {
                'params': list(self.model.get_head_parameters()),
                'lr': head_lr,
                'name': 'head'
            }
        ]

        if self.config['training_phase2']['optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(
                param_groups,
                weight_decay=self.config['training_phase2']['weight_decay']
            )
        else:
            raise ValueError(f"Optimizador no soportado: {self.config['training_phase2']['optimizer']}")

        # Scheduler CosineAnnealingLR
        if self.config['training_phase2']['scheduler'] == 'cosine_annealing':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training_phase2']['epochs'],
                eta_min=self.config['training_phase2']['min_lr']
            )

        # Early stopping con paciencia mayor para fine-tuning
        self.early_stopping = EarlyStopping(
            patience=self.config['early_stopping']['patience'],
            min_delta=self.config['early_stopping']['min_delta'],
            monitor=self.config['early_stopping']['monitor']
        )

        # Logger
        self.logger = TrainingLogger(
            log_dir=self.config['logging']['log_dir'],
            experiment_name="phase2_full_finetuning"
        )

        print("✓ Componentes de entrenamiento configurados")
        print(f"✓ Learning rate backbone: {backbone_lr}")
        print(f"✓ Learning rate cabeza: {head_lr}")
        print(f"✓ Scheduler: {self.config['training_phase2']['scheduler']}")

    def train_epoch(self, epoch: int) -> tuple:
        """
        Entrenar una época con fine-tuning completo

        Args:
            epoch: Número de época

        Returns:
            Tupla de (pérdida_promedio, métricas)
        """
        self.model.train()

        epoch_loss = 0.0
        all_predictions = []
        all_targets = []

        # Barra de progreso
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train - Phase2]")

        for batch_idx, (images, landmarks, metadata) in enumerate(pbar):
            # Mover datos al dispositivo
            images = images.to(self.device)
            landmarks = landmarks.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            predictions = self.model(images)

            # Calcular pérdida
            loss = self.criterion(predictions, landmarks)

            # Backward pass
            loss.backward()

            # Clip gradients para estabilidad en fine-tuning
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimizar
            self.optimizer.step()

            # Acumular métricas
            epoch_loss += loss.item()
            all_predictions.append(predictions.detach())
            all_targets.append(landmarks.detach())

            # Actualizar barra de progreso con learning rates actuales
            current_lr_backbone = self.optimizer.param_groups[0]['lr']
            current_lr_head = self.optimizer.param_groups[1]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'lr_bb': f'{current_lr_backbone:.6f}',
                'lr_head': f'{current_lr_head:.6f}'
            })

        # Calcular métricas de la época
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        avg_loss = epoch_loss / len(self.train_loader)
        metrics = LandmarkMetrics.calculate_all_metrics(all_predictions, all_targets)

        return avg_loss, metrics

    def validate_epoch(self, epoch: int) -> tuple:
        """
        Validar una época

        Args:
            epoch: Número de época

        Returns:
            Tupla de (pérdida_promedio, métricas)
        """
        self.model.eval()

        epoch_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val - Phase2]")

            for batch_idx, (images, landmarks, metadata) in enumerate(pbar):
                # Mover datos al dispositivo
                images = images.to(self.device)
                landmarks = landmarks.to(self.device)

                # Forward pass
                predictions = self.model(images)

                # Calcular pérdida
                loss = self.criterion(predictions, landmarks)

                # Acumular métricas
                epoch_loss += loss.item()
                all_predictions.append(predictions)
                all_targets.append(landmarks)

                # Actualizar barra de progreso
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        # Calcular métricas de la época
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        avg_loss = epoch_loss / len(self.val_loader)
        metrics = LandmarkMetrics.calculate_all_metrics(all_predictions, all_targets)

        return avg_loss, metrics

    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float,
                       train_metrics: dict, val_metrics: dict, is_best: bool = False):
        """
        Guardar checkpoint del modelo

        Args:
            epoch: Época actual
            train_loss: Pérdida de entrenamiento
            val_loss: Pérdida de validación
            train_metrics: Métricas de entrenamiento
            val_metrics: Métricas de validación
            is_best: Si es el mejor modelo hasta ahora
        """
        checkpoint_path = self.checkpoint_dir / f"phase2_epoch_{epoch+1}.pt"

        self.model.save_checkpoint(
            filepath=str(checkpoint_path),
            epoch=epoch,
            optimizer_state=self.optimizer.state_dict(),
            loss=val_loss,
            metrics={
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'scheduler_state': self.scheduler.state_dict() if self.scheduler else None
            }
        )

        if is_best:
            best_path = self.checkpoint_dir / "phase2_best.pt"
            self.model.save_checkpoint(
                filepath=str(best_path),
                epoch=epoch,
                optimizer_state=self.optimizer.state_dict(),
                loss=val_loss,
                metrics={
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'scheduler_state': self.scheduler.state_dict() if self.scheduler else None
                }
            )
            print(f"✓ Nuevo mejor modelo guardado: {best_path}")

    def train(self):
        """
        Ejecutar entrenamiento completo de Fase 2
        """
        print("\n🚀 Iniciando entrenamiento Fase 2...")

        # Configurar componentes
        self.setup_model()
        self.setup_data()
        self.setup_training_components()

        # Variables de entrenamiento
        epochs = self.config['training_phase2']['epochs']
        best_val_loss = float('inf')
        timer = Timer()
        timer.start()

        print(f"\n📈 Fine-tuning por {epochs} épocas...")

        for epoch in range(epochs):
            print(f"\n--- Época {epoch+1}/{epochs} [Fine-tuning] ---")

            # Entrenar
            train_loss, train_metrics = self.train_epoch(epoch)

            # Validar
            val_loss, val_metrics = self.validate_epoch(epoch)

            # Actualizar scheduler
            if self.scheduler:
                self.scheduler.step()

            # Logging
            self.logger.log_metrics(epoch, train_loss, val_loss, train_metrics, val_metrics)

            # Mostrar resultados
            current_lr_backbone = self.optimizer.param_groups[0]['lr']
            current_lr_head = self.optimizer.param_groups[1]['lr']

            print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            print(f"Train RMSE: {train_metrics['rmse']:.6f} | Val RMSE: {val_metrics['rmse']:.6f}")
            print(f"Train MAE: {train_metrics['mae']:.6f} | Val MAE: {val_metrics['mae']:.6f}")
            print(f"LR - Backbone: {current_lr_backbone:.6f} | Head: {current_lr_head:.6f}")

            # Guardar checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            if (epoch + 1) % self.config['logging']['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, train_loss, val_loss, train_metrics, val_metrics, is_best)

            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\n⚠ Early stopping activado en época {epoch+1}")
                break

        timer.stop()

        # Finalizar entrenamiento
        print(f"\n✅ Entrenamiento Fase 2 completado en {timer.formatted_elapsed()}")
        print(f"✅ Mejor pérdida de validación: {best_val_loss:.6f}")

        # Guardar gráficos y cerrar logger
        self.logger.save_metrics_plot()
        self.logger.close()

        return best_val_loss


def main():
    """
    Función principal de entrenamiento Fase 2
    """
    # Cargar configuración
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Archivo de configuración no encontrado: {config_path}")
        return

    config = load_config(config_path)

    # Verificar checkpoint de Fase 1
    phase1_checkpoint = "checkpoints/phase1_best.pt"
    if not os.path.exists(phase1_checkpoint):
        print(f"❌ Checkpoint de Fase 1 no encontrado: {phase1_checkpoint}")
        print("💡 Ejecuta primero el entrenamiento de Fase 1")
        return

    # Crear entrenador y ejecutar
    trainer = Phase2Trainer(config, phase1_checkpoint)

    try:
        best_loss = trainer.train()
        print(f"\n🎉 Fase 2 completada exitosamente!")
        print(f"🎯 Mejor pérdida de validación: {best_loss:.6f}")
        print(f"📁 Checkpoints guardados en: {trainer.checkpoint_dir}")

    except KeyboardInterrupt:
        print("\n⚠ Entrenamiento interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        raise


if __name__ == "__main__":
    main()
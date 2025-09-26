#!/usr/bin/env python3
"""
Entrenamiento Fase 1: Solo cabeza de regresión
Backbone de ResNet-18 congelado, solo entrenar la nueva capa de regresión
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Agregar src al path para imports
sys.path.append(str(Path(__file__).parent.parent))

from models.resnet_regressor import create_model, print_model_summary
from data.dataset import create_dataloaders
from training.utils import (
    LandmarkLoss, LandmarkMetrics, EarlyStopping, TrainingLogger,
    load_config, setup_device, Timer
)


class Phase1Trainer:
    """
    Entrenador para Fase 1: Solo cabeza de regresión

    En esta fase:
    1. Backbone congelado (no se actualiza)
    2. Solo se entrena la cabeza de regresión
    3. Learning rate alto para la nueva capa
    4. Pocas épocas para evitar overfitting de la cabeza
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Configuración del entrenamiento
        """
        self.config = config
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
        print("ENTRENAMIENTO FASE 1: SOLO CABEZA DE REGRESIÓN")
        print("="*60)

    def setup_model(self):
        """
        Configurar modelo para Fase 1
        """
        print("\n🔧 Configurando modelo...")

        # Crear modelo con backbone congelado
        self.model = create_model(
            num_landmarks=self.config['model']['num_landmarks'],
            pretrained=self.config['model']['pretrained'],
            freeze_backbone=True,  # ¡IMPORTANTE! Congelar backbone en Fase 1
            dropout_rate=0.5
        )

        # Mover a dispositivo
        self.model = self.model.to(self.device)

        # Mostrar resumen del modelo
        print_model_summary(self.model)

        print("✓ Modelo configurado para Fase 1")

    def setup_data(self):
        """
        Configurar datasets y dataloaders
        """
        print("\n📊 Configurando datos...")

        # Crear dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
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

        print("✓ Datos configurados")

    def setup_training_components(self):
        """
        Configurar optimizador, scheduler, pérdida, etc.
        """
        print("\n⚙️ Configurando componentes de entrenamiento...")

        # Función de pérdida
        self.criterion = LandmarkLoss(
            loss_type=self.config['loss']['type'],
            reduction=self.config['loss']['reduction']
        )

        # Optimizador - Solo parámetros de la cabeza
        # En Fase 1, el backbone está congelado
        optimizer_params = list(self.model.get_head_parameters())

        if self.config['training_phase1']['optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(
                optimizer_params,
                lr=self.config['training_phase1']['learning_rate'],
                weight_decay=self.config['training_phase1']['weight_decay']
            )
        else:
            raise ValueError(f"Optimizador no soportado: {self.config['training_phase1']['optimizer']}")

        # Scheduler
        if self.config['training_phase1']['scheduler'] == 'step_lr':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=self.config['training_phase1']['step_size'],
                gamma=self.config['training_phase1']['gamma']
            )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['early_stopping']['patience'],
            min_delta=self.config['early_stopping']['min_delta'],
            monitor=self.config['early_stopping']['monitor']
        )

        # Logger
        self.logger = TrainingLogger(
            log_dir=self.config['logging']['log_dir'],
            experiment_name="phase1_head_only"
        )

        print("✓ Componentes de entrenamiento configurados")
        print(f"✓ Optimizando solo parámetros de la cabeza: {sum(p.numel() for p in optimizer_params):,}")

    def train_epoch(self, epoch: int) -> tuple:
        """
        Entrenar una época

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
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")

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

            # Optimizar
            self.optimizer.step()

            # Acumular métricas
            epoch_loss += loss.item()
            all_predictions.append(predictions.detach())
            all_targets.append(landmarks.detach())

            # Actualizar barra de progreso
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

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
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")

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
        checkpoint_path = self.checkpoint_dir / f"phase1_epoch_{epoch+1}.pt"

        self.model.save_checkpoint(
            filepath=str(checkpoint_path),
            epoch=epoch,
            optimizer_state=self.optimizer.state_dict(),
            loss=val_loss,
            metrics={
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
        )

        if is_best:
            best_path = self.checkpoint_dir / "phase1_best.pt"
            self.model.save_checkpoint(
                filepath=str(best_path),
                epoch=epoch,
                optimizer_state=self.optimizer.state_dict(),
                loss=val_loss,
                metrics={
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }
            )
            print(f"✓ Nuevo mejor modelo guardado: {best_path}")

    def train(self):
        """
        Ejecutar entrenamiento completo de Fase 1
        """
        print("\n🚀 Iniciando entrenamiento Fase 1...")

        # Configurar componentes
        self.setup_model()
        self.setup_data()
        self.setup_training_components()

        # Variables de entrenamiento
        epochs = self.config['training_phase1']['epochs']
        best_val_loss = float('inf')
        timer = Timer()
        timer.start()

        print(f"\n📈 Entrenando por {epochs} épocas...")

        for epoch in range(epochs):
            print(f"\n--- Época {epoch+1}/{epochs} ---")

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
            print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            print(f"Train RMSE: {train_metrics['rmse']:.6f} | Val RMSE: {val_metrics['rmse']:.6f}")
            print(f"Train MAE: {train_metrics['mae']:.6f} | Val MAE: {val_metrics['mae']:.6f}")

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
        print(f"\n✅ Entrenamiento Fase 1 completado en {timer.formatted_elapsed()}")
        print(f"✅ Mejor pérdida de validación: {best_val_loss:.6f}")

        # Guardar gráficos y cerrar logger
        self.logger.save_metrics_plot()
        self.logger.close()

        return best_val_loss


def main():
    """
    Función principal de entrenamiento Fase 1
    """
    # Cargar configuración
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Archivo de configuración no encontrado: {config_path}")
        return

    config = load_config(config_path)

    # Crear entrenador y ejecutar
    trainer = Phase1Trainer(config)

    try:
        best_loss = trainer.train()
        print(f"\n🎉 Fase 1 completada exitosamente!")
        print(f"🎯 Mejor pérdida de validación: {best_loss:.6f}")
        print(f"📁 Checkpoints guardados en: {trainer.checkpoint_dir}")

    except KeyboardInterrupt:
        print("\n⚠ Entrenamiento interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        raise


if __name__ == "__main__":
    main()
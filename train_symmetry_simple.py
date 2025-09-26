#!/usr/bin/env python3
"""
Script simplificado para entrenamiento Phase 3: Wing Loss + Symmetry Loss
Versión robusta sin errores de formato
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time

# Importar módulos del proyecto
from src.data.dataset import create_dataloaders
from src.models.resnet_regressor import ResNetLandmarkRegressor
from src.models.losses import WingLoss, SymmetryLoss
from src.training.utils import load_config, setup_device


class SimpleSymmetryTrainer:
    def __init__(self):
        # Configurar device
        self.device = setup_device(use_gpu=True, gpu_id=0)

        # Configurar semillas
        torch.manual_seed(42)
        np.random.seed(42)

        print("🚀 ENTRENAMIENTO PHASE 3: WING LOSS + SYMMETRY LOSS")
        print(f"⚡ Device: {self.device}")

    def create_combined_loss(self):
        """Crear loss combinado Wing + Symmetry"""
        wing_loss = WingLoss(omega=10.0, epsilon=2.0)
        symmetry_loss = SymmetryLoss(symmetry_weight=1.0, use_mediastinal_axis=True)

        def combined_loss_fn(predictions, targets):
            wing = wing_loss(predictions, targets)
            symmetry = symmetry_loss(predictions)
            total = wing + 0.3 * symmetry
            return total, wing.item(), symmetry.item()

        return combined_loss_fn

    def compute_pixel_error(self, predictions, targets):
        """Calcular error en píxeles"""
        pred_reshaped = predictions.view(-1, 15, 2)
        target_reshaped = targets.view(-1, 15, 2)
        distances = torch.norm(pred_reshaped - target_reshaped, dim=2)
        pixel_distances = distances * 224  # Convert to pixels
        return torch.mean(pixel_distances)

    def train(self):
        print("\n📊 Configurando data loaders...")

        # Data loaders
        train_loader, val_loader, test_loader = create_dataloaders(
            annotations_file="data/coordenadas/coordenadas_maestro.csv",
            images_dir="data/dataset",
            batch_size=8,
            num_workers=4,
            pin_memory=True,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )

        print(f"✓ Train: {len(train_loader.dataset)} samples")
        print(f"✓ Val: {len(val_loader.dataset)} samples")

        print("\n🏗️ Cargando modelo...")

        # Cargar modelo
        checkpoint_path = "checkpoints/geometric_phase2_wing_loss.pt"
        model, checkpoint = ResNetLandmarkRegressor.load_from_checkpoint(
            checkpoint_path, map_location=str(self.device)
        )
        model = model.to(self.device)
        model.unfreeze_backbone()

        print(f"✓ Modelo cargado desde: {checkpoint_path}")
        print(f"✓ Parámetros: {sum(p.numel() for p in model.parameters()):,}")

        print("\n⚡ Configurando optimizador...")

        # Loss y optimizer
        criterion = self.create_combined_loss()

        # Parámetros diferenciados
        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if 'regression_head' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': 0.00002, 'weight_decay': 0.00005},
            {'params': head_params, 'lr': 0.0002, 'weight_decay': 0.00005}
        ])

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=70, eta_min=0.000002)

        print("✓ Wing Loss + Symmetry Loss configurado")
        print("✓ Learning rates diferenciados configurados")

        # Entrenamiento
        epochs = 70
        best_val_error = float('inf')
        best_epoch = 0

        print(f"\n🎯 Iniciando entrenamiento ({epochs} épocas)...")

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # Training
            model.train()
            train_loss = 0.0
            train_wing = 0.0
            train_symmetry = 0.0
            train_pixel_error = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

            for images, landmarks, _ in progress_bar:
                images = images.to(self.device)
                landmarks = landmarks.to(self.device)

                optimizer.zero_grad()
                predictions = model(images)

                loss, wing_loss, symmetry_loss = criterion(predictions, landmarks)
                loss.backward()
                optimizer.step()

                pixel_error = self.compute_pixel_error(predictions, landmarks)

                train_loss += loss.item()
                train_wing += wing_loss
                train_symmetry += symmetry_loss
                train_pixel_error += pixel_error.item()

                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Wing': f"{wing_loss:.4f}",
                    'Sym': f"{symmetry_loss:.4f}",
                    'Pixel': f"{pixel_error.item():.2f}px"
                })

            # Validation
            model.eval()
            val_loss = 0.0
            val_pixel_error = 0.0

            with torch.no_grad():
                for images, landmarks, _ in val_loader:
                    images = images.to(self.device)
                    landmarks = landmarks.to(self.device)

                    predictions = model(images)
                    loss, _, _ = criterion(predictions, landmarks)
                    pixel_error = self.compute_pixel_error(predictions, landmarks)

                    val_loss += loss.item()
                    val_pixel_error += pixel_error.item()

            # Métricas promedio
            train_loss /= len(train_loader)
            train_wing /= len(train_loader)
            train_symmetry /= len(train_loader)
            train_pixel_error /= len(train_loader)
            val_loss /= len(val_loader)
            val_pixel_error /= len(val_loader)

            # Scheduler step
            scheduler.step()

            # Guardar mejor modelo
            if val_pixel_error < best_val_error:
                best_val_error = val_pixel_error
                best_epoch = epoch

                checkpoint_path = Path("checkpoints/geometric_symmetry.pt")
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_pixel_error': best_val_error,
                    'val_loss': val_loss
                }, checkpoint_path)

            # Progreso
            elapsed = time.time() - start_time
            print(f"\nÉpoca {epoch}/{epochs} - {elapsed/epoch:.1f}s/epoch")
            print(f"  Train: Loss={train_loss:.4f}, Wing={train_wing:.4f}, Sym={train_symmetry:.4f}")
            print(f"  Val: Loss={val_loss:.4f}, Pixel={val_pixel_error:.2f}px")
            print(f"  Best: {best_val_error:.2f}px (epoch {best_epoch})")

            # Early stopping simple
            if epoch - best_epoch > 15:
                print(f"\n⏹️ Early stopping en época {epoch}")
                break

        total_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("🎉 ENTRENAMIENTO COMPLETADO")
        print("=" * 60)
        print(f"⏱️ Tiempo total: {total_time:.1f}s")
        print(f"🏆 Mejor época: {best_epoch}")
        print(f"📊 Mejor error: {best_val_error:.2f}px")
        print(f"🎯 Objetivo alcanzado: {'✅ SÍ' if best_val_error <= 9.3 else '❌ NO'} (≤9.3px)")
        print(f"💾 Modelo guardado: checkpoints/geometric_symmetry.pt")

        return best_val_error


def main():
    trainer = SimpleSymmetryTrainer()
    final_error = trainer.train()
    return 0 if final_error <= 9.3 else 1


if __name__ == "__main__":
    exit(main())
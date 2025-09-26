#!/usr/bin/env python3
"""
Entrenamiento Ensemble: Múltiples modelos para Bootstrap Aggregating
Entrena varios modelos ResNet-18 con diferentes random seeds para ensemble learning
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Agregar src al path para imports
sys.path.append(str(Path(__file__).parent.parent))

from training.train_phase1 import Phase1Trainer
from training.train_phase2 import Phase2Trainer
from training.utils import load_config, Timer


class EnsembleTrainer:
    """
    Entrenador de Ensemble para landmark prediction

    Estrategia: Bootstrap Aggregating (Bagging)
    - Entrenar múltiples modelos ResNet-18 idénticos
    - Usar diferentes random seeds para diversidad
    - Mismo pipeline de 2 fases para cada modelo
    - Guardado organizado por seed para posterior ensemble
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Configuración del entrenamiento ensemble
        """
        self.config = config
        self.ensemble_config = config.get('ensemble', {})

        # Configuración del ensemble
        self.num_models = self.ensemble_config.get('num_models', 5)
        self.random_seeds = self.ensemble_config.get('random_seeds', [42, 123, 456, 789, 999])

        # Verificar que tenemos suficientes seeds
        if len(self.random_seeds) < self.num_models:
            # Generar seeds adicionales si es necesario
            import random
            random.seed(42)
            additional_seeds = [random.randint(1, 10000) for _ in range(self.num_models - len(self.random_seeds))]
            self.random_seeds.extend(additional_seeds)

        # Usar solo los primeros num_models seeds
        self.random_seeds = self.random_seeds[:self.num_models]

        # Directorios
        self.checkpoint_dir = Path(config['logging']['checkpoint_dir'])
        self.ensemble_dir = self.checkpoint_dir / "ensemble"
        self.ensemble_dir.mkdir(parents=True, exist_ok=True)

        print("="*70)
        print("ENTRENAMIENTO ENSEMBLE: BOOTSTRAP AGGREGATING")
        print("="*70)
        print(f"🎯 Número de modelos: {self.num_models}")
        print(f"🎲 Random seeds: {self.random_seeds}")
        print(f"📁 Directorio ensemble: {self.ensemble_dir}")

    def train_phase1_for_seed(self, seed: int) -> str:
        """
        Entrenar Fase 1 para una seed específica

        Args:
            seed: Random seed para este modelo

        Returns:
            Ruta al checkpoint de Fase 1 generado
        """
        print(f"\n🔄 Entrenando Fase 1 para seed {seed}")

        # Crear configuración específica para este modelo
        model_config = self.config.copy()
        model_config['split']['random_seed'] = seed

        # Crear trainer para Fase 1
        trainer = Phase1Trainer(model_config)

        try:
            # Entrenar Fase 1
            trainer.train()

            # Mover checkpoint con nombre específico por seed
            original_checkpoint = trainer.checkpoint_dir / "phase1_best.pt"
            seed_checkpoint = self.checkpoint_dir / f"phase1_seed_{seed}_best.pt"

            if original_checkpoint.exists():
                # Copiar checkpoint con metadatos de seed
                import torch
                checkpoint = torch.load(original_checkpoint)
                checkpoint['ensemble_info'] = {
                    'random_seed': seed,
                    'phase': 1,
                    'ensemble_size': self.num_models
                }
                torch.save(checkpoint, seed_checkpoint)
                print(f"✓ Fase 1 completada para seed {seed}: {seed_checkpoint}")
                return str(seed_checkpoint)
            else:
                raise FileNotFoundError(f"Checkpoint de Fase 1 no encontrado: {original_checkpoint}")

        except Exception as e:
            print(f"❌ Error entrenando Fase 1 para seed {seed}: {e}")
            raise

    def train_single_model(self, seed: int, model_idx: int) -> tuple:
        """
        Entrenar un solo modelo del ensemble (Fase 1 + Fase 2)

        Args:
            seed: Random seed para este modelo
            model_idx: Índice del modelo (para logging)

        Returns:
            Tupla de (best_val_loss, checkpoint_path)
        """
        print(f"\n" + "="*50)
        print(f"🔄 MODELO {model_idx+1}/{self.num_models} - SEED {seed}")
        print("="*50)

        # Paso 1: Entrenar Fase 1 si no existe
        phase1_checkpoint = self.checkpoint_dir / f"phase1_seed_{seed}_best.pt"

        if not phase1_checkpoint.exists():
            print(f"🔧 Entrenando Fase 1 para seed {seed}...")
            phase1_checkpoint_path = self.train_phase1_for_seed(seed)
        else:
            print(f"✓ Fase 1 ya existe para seed {seed}: {phase1_checkpoint}")
            phase1_checkpoint_path = str(phase1_checkpoint)

        # Paso 2: Entrenar Fase 2
        print(f"🚀 Entrenando Fase 2 para seed {seed}...")

        # Crear configuración específica para este modelo
        model_config = self.config.copy()
        model_config['split']['random_seed'] = seed

        # Crear trainer para Fase 2
        trainer = Phase2Trainer(model_config, phase1_checkpoint_path)

        try:
            # Entrenar modelo
            timer = Timer()
            timer.start()

            best_val_loss = trainer.train()

            timer.stop()

            # Mover checkpoint a directorio ensemble con nombre específico
            original_checkpoint = trainer.checkpoint_dir / "phase2_best.pt"
            ensemble_checkpoint = self.ensemble_dir / f"model_{seed}.pt"

            if original_checkpoint.exists():
                # Copiar checkpoint con metadatos adicionales
                import torch
                checkpoint = torch.load(original_checkpoint)
                checkpoint['ensemble_info'] = {
                    'model_index': model_idx,
                    'random_seed': seed,
                    'training_time': timer.elapsed,
                    'ensemble_size': self.num_models
                }
                torch.save(checkpoint, ensemble_checkpoint)
                print(f"✓ Checkpoint guardado: {ensemble_checkpoint}")
            else:
                raise FileNotFoundError(f"Checkpoint de Fase 2 no encontrado: {original_checkpoint}")

            print(f"✅ Modelo {model_idx+1} completado en {timer.formatted_elapsed()}")
            print(f"📊 Mejor pérdida de validación: {best_val_loss:.6f}")

            return best_val_loss, str(ensemble_checkpoint)

        except Exception as e:
            print(f"❌ Error entrenando modelo {model_idx+1} (seed {seed}): {e}")
            raise

    def train_ensemble(self) -> Dict[str, Any]:
        """
        Entrenar ensemble completo

        Returns:
            Diccionario con estadísticas del ensemble
        """
        print(f"\n🚀 Iniciando entrenamiento ensemble...")

        # Ahora entrenamos automáticamente Fase 1 + Fase 2 para cada seed

        # Variables de tracking
        ensemble_results = []
        total_timer = Timer()
        total_timer.start()

        print(f"\n📈 Entrenando {self.num_models} modelos...")

        # Entrenar cada modelo del ensemble
        for model_idx, seed in enumerate(self.random_seeds):
            try:
                best_loss, checkpoint_path = self.train_single_model(seed, model_idx)

                ensemble_results.append({
                    'model_index': model_idx,
                    'seed': seed,
                    'best_val_loss': best_loss,
                    'checkpoint_path': checkpoint_path,
                    'status': 'completed'
                })

            except Exception as e:
                print(f"⚠ Saltando modelo {model_idx+1} debido a error: {e}")
                ensemble_results.append({
                    'model_index': model_idx,
                    'seed': seed,
                    'best_val_loss': None,
                    'checkpoint_path': None,
                    'status': 'failed',
                    'error': str(e)
                })

        total_timer.stop()

        # Calcular estadísticas
        completed_models = [r for r in ensemble_results if r['status'] == 'completed']
        failed_models = [r for r in ensemble_results if r['status'] == 'failed']

        if completed_models:
            losses = [r['best_val_loss'] for r in completed_models]
            avg_loss = sum(losses) / len(losses)
            min_loss = min(losses)
            max_loss = max(losses)
            std_loss = (sum((l - avg_loss) ** 2 for l in losses) / len(losses)) ** 0.5
        else:
            avg_loss = min_loss = max_loss = std_loss = None

        # Guardar metadata del ensemble
        ensemble_metadata = {
            'num_models_requested': self.num_models,
            'num_models_completed': len(completed_models),
            'num_models_failed': len(failed_models),
            'random_seeds': self.random_seeds,
            'training_time_total': total_timer.elapsed,
            'statistics': {
                'avg_val_loss': avg_loss,
                'min_val_loss': min_loss,
                'max_val_loss': max_loss,
                'std_val_loss': std_loss
            },
            'model_results': ensemble_results,
            'config': self.ensemble_config
        }

        # Guardar metadata
        import json
        metadata_path = self.ensemble_dir / "ensemble_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(ensemble_metadata, f, indent=2, default=str)

        # Mostrar resumen
        print(f"\n" + "="*70)
        print("🎉 ENTRENAMIENTO ENSEMBLE COMPLETADO")
        print("="*70)
        print(f"⏱ Tiempo total: {total_timer.formatted_elapsed()}")
        print(f"✅ Modelos completados: {len(completed_models)}/{self.num_models}")

        if failed_models:
            print(f"❌ Modelos fallidos: {len(failed_models)}")

        if completed_models:
            print(f"📊 Estadísticas de pérdida de validación:")
            print(f"   • Promedio: {avg_loss:.6f}")
            print(f"   • Mínimo: {min_loss:.6f}")
            print(f"   • Máximo: {max_loss:.6f}")
            print(f"   • Desviación estándar: {std_loss:.6f}")

        print(f"📁 Checkpoints guardados en: {self.ensemble_dir}")
        print(f"📋 Metadata guardada en: {metadata_path}")

        if len(completed_models) >= 3:  # Mínimo viable para ensemble
            print("✨ Ensemble listo para evaluación!")
            ensemble_metadata['status'] = 'ready'
        else:
            print("⚠ Ensemble incompleto - se recomienda al menos 3 modelos")
            ensemble_metadata['status'] = 'incomplete'

        return ensemble_metadata


def main():
    """
    Función principal de entrenamiento ensemble
    """
    # Cargar configuración
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Archivo de configuración no encontrado: {config_path}")
        return

    config = load_config(config_path)

    # Verificar configuración de ensemble
    if 'ensemble' not in config:
        print("❌ Configuración de ensemble no encontrada en config.yaml")
        print("💡 Agrega una sección 'ensemble' a la configuración")
        return

    # Crear entrenador ensemble
    trainer = EnsembleTrainer(config)

    try:
        results = trainer.train_ensemble()

        if results.get('status') == 'ready':
            print(f"\n🎯 Ensemble entrenado exitosamente!")
            print(f"🔗 Para evaluar: python main.py evaluate_ensemble")
        elif results.get('status') == 'incomplete':
            print(f"\n⚠ Ensemble parcialmente completado")
        else:
            print(f"\n❌ Entrenamiento ensemble falló")

    except KeyboardInterrupt:
        print("\n⚠ Entrenamiento interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante entrenamiento ensemble: {e}")
        raise


if __name__ == "__main__":
    main()
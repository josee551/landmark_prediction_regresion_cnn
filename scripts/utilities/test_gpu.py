#!/usr/bin/env python3
"""
Script para verificar la compatibilidad de GPU AMD con ROCm y PyTorch
"""

import torch
import sys

def test_gpu_compatibility():
    print("=== Verificación de GPU AMD con ROCm ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")

    # Verificar si ROCm está disponible
    print(f"ROCm disponible: {torch.version.hip is not None}")
    if torch.version.hip:
        print(f"HIP version: {torch.version.hip}")

    # Verificar dispositivos disponibles
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    print(f"Número de dispositivos: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Dispositivo {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memoria total: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

    # Test básico de tensor en GPU
    if torch.cuda.is_available():
        print("\n=== Test de operaciones en GPU ===")
        try:
            device = torch.device('cuda:0')
            print(f"Dispositivo seleccionado: {device}")

            # Crear tensores en GPU
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)

            # Operación matricial
            z = torch.mm(x, y)
            print("✓ Multiplicación de matrices en GPU exitosa")
            print(f"  Forma del resultado: {z.shape}")
            print(f"  Dispositivo del resultado: {z.device}")

            return True

        except Exception as e:
            print(f"✗ Error en operaciones GPU: {e}")
            return False
    else:
        print("No hay GPU disponible para testing")
        return False

if __name__ == "__main__":
    success = test_gpu_compatibility()
    if success:
        print("\n✓ GPU AMD configurada correctamente con ROCm")
    else:
        print("\n✗ Problemas con la configuración de GPU")

    sys.exit(0 if success else 1)
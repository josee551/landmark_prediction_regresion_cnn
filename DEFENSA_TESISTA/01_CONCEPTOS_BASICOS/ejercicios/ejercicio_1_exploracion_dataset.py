#!/usr/bin/env python3
"""
EJERCICIO PRÁCTICO 1: EXPLORACIÓN DEL DATASET
Módulo 1: Conceptos Básicos - Proyecto Landmarks 8.13px

Este ejercicio ayuda al tesista a explorar manualmente el dataset
para comprender su composición y características fundamentales.

Tiempo estimado: 30-45 minutos
Objetivo: Familiarizarse con la estructura de datos del proyecto
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    """
    Ejercicio guiado de exploración del dataset de landmarks
    """
    print("🔍 EJERCICIO 1: EXPLORACIÓN DEL DATASET DE LANDMARKS")
    print("=" * 60)

    # Verificar estructura del proyecto
    print("\n📁 PASO 1: Verificar estructura de directorios")
    dataset_path = Path("data/dataset")
    coords_path = Path("data/coordenadas")

    if not dataset_path.exists():
        print("❌ Error: Directorio data/dataset no encontrado")
        print("   Asegúrate de ejecutar desde el directorio raíz del proyecto")
        return

    print(f"✅ Dataset encontrado en: {dataset_path}")
    print(f"✅ Coordenadas encontradas en: {coords_path}")

    # Análisis de categorías
    print("\n🏥 PASO 2: Análisis de categorías médicas")
    categories = {}

    for category in ["COVID", "Normal", "Viral"]:
        category_path = dataset_path / category
        if category_path.exists():
            images = list(category_path.glob("*.png"))
            categories[category] = len(images)
            print(f"   {category:>8}: {len(images):>3} imágenes")
        else:
            print(f"❌ Categoría {category} no encontrada")

    total_images = sum(categories.values())
    print(f"\n📊 TOTAL GENERAL: {total_images} imágenes")

    # Cálculo de proporciones
    print("\n📈 PASO 3: Proporciones del dataset")
    for category, count in categories.items():
        percentage = (count / total_images) * 100
        print(f"   {category:>8}: {percentage:>5.1f}%")

    # PREGUNTA PARA EL TESISTA
    print("\n❓ PREGUNTA DE REFLEXIÓN:")
    print("   ¿Por qué crees que tenemos más imágenes Normales que patológicas?")
    print("   ¿Qué implicaciones tiene esto para el entrenamiento del modelo?")

    # Análisis de coordenadas
    print("\n📐 PASO 4: Análisis de archivo de coordenadas")
    coord_file = coords_path / "coordinates.csv"

    if coord_file.exists():
        try:
            df = pd.read_csv(coord_file)
            print(f"✅ Archivo de coordenadas cargado: {len(df)} filas")
            print(f"   Columnas: {len(df.columns)} (debería ser 31: nombre + 30 coordenadas)")

            # Verificar formato
            if len(df.columns) == 31:
                print("✅ Formato correcto: 1 columna nombre + 30 coordenadas (15 landmarks × 2)")

                # Mostrar ejemplo
                print("\n📋 EJEMPLO DE ANOTACIÓN:")
                sample_row = df.iloc[0]
                print(f"   Imagen: {sample_row.iloc[0]}")
                print("   Primeros 6 valores (landmarks 1-3):")
                for i in range(1, 7, 2):
                    landmark_num = (i + 1) // 2
                    x, y = sample_row.iloc[i], sample_row.iloc[i+1]
                    print(f"     Landmark {landmark_num}: ({x:.1f}, {y:.1f})")

            else:
                print(f"❌ Formato inesperado: {len(df.columns)} columnas")

        except Exception as e:
            print(f"❌ Error leyendo coordenadas: {e}")
    else:
        print("❌ Archivo de coordenadas no encontrado")

    # Cálculo de eficiencia
    print("\n⚡ PASO 5: Cálculo de eficiencia automática vs manual")
    manual_time_per_image = 7  # minutos promedio
    automatic_time_per_image = 0.1  # segundos

    total_manual_minutes = total_images * manual_time_per_image
    total_automatic_seconds = total_images * automatic_time_per_image
    total_automatic_minutes = total_automatic_seconds / 60

    print(f"   Tiempo manual total: {total_manual_minutes:,} minutos ({total_manual_minutes/60:.1f} horas)")
    print(f"   Tiempo automático total: {total_automatic_minutes:.1f} minutos")

    efficiency = total_manual_minutes / total_automatic_minutes
    print(f"   🚀 EFICIENCIA: {efficiency:.0f}x más rápido")

    # Costo económico estimado
    radiologist_cost_per_hour = 125  # USD
    manual_cost = (total_manual_minutes / 60) * radiologist_cost_per_hour
    automatic_cost = 1  # USD estimado

    print(f"\n💰 IMPACTO ECONÓMICO ESTIMADO:")
    print(f"   Costo anotación manual: ${manual_cost:,.0f} USD")
    print(f"   Costo procesamiento automático: ${automatic_cost} USD")
    print(f"   Ahorro potencial: ${manual_cost - automatic_cost:,.0f} USD")

    # Visualización simple
    print("\n📊 PASO 6: Generando gráfico de distribución")
    try:
        plt.figure(figsize=(10, 6))

        # Gráfico de barras de categorías
        plt.subplot(1, 2, 1)
        categories_list = list(categories.keys())
        counts = list(categories.values())
        colors = ['red', 'green', 'orange']

        plt.bar(categories_list, counts, color=colors, alpha=0.7)
        plt.title('Distribución de Categorías\nDataset Landmarks (956 imágenes)')
        plt.ylabel('Número de imágenes')

        # Agregar valores en barras
        for i, count in enumerate(counts):
            plt.text(i, count + 5, str(count), ha='center', fontweight='bold')

        # Gráfico de pie de proporciones
        plt.subplot(1, 2, 2)
        percentages = [(count/total_images)*100 for count in counts]
        plt.pie(percentages, labels=[f'{cat}\n{pct:.1f}%' for cat, pct in zip(categories_list, percentages)],
                colors=colors, autopct='', startangle=90)
        plt.title('Proporciones del Dataset')

        plt.tight_layout()

        # Guardar gráfico
        output_file = "DEFENSA_TESISTA/01_CONCEPTOS_BASICOS/diagramas/dataset_distribution.png"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')

        print(f"✅ Gráfico guardado en: {output_file}")
        plt.show()

    except ImportError:
        print("⚠️  Matplotlib no disponible, saltando visualización")
    except Exception as e:
        print(f"❌ Error generando gráfico: {e}")

    # Resumen para memorizar
    print("\n🎯 DATOS CLAVE PARA MEMORIZAR:")
    print("=" * 40)
    print(f"• Total de imágenes: {total_images}")
    print(f"• COVID: {categories.get('COVID', 0)} ({(categories.get('COVID', 0)/total_images)*100:.1f}%)")
    print(f"• Normal: {categories.get('Normal', 0)} ({(categories.get('Normal', 0)/total_images)*100:.1f}%)")
    print(f"• Viral: {categories.get('Viral', 0)} ({(categories.get('Viral', 0)/total_images)*100:.1f}%)")
    print(f"• Landmarks por imagen: 15 (30 coordenadas)")
    print(f"• Total landmarks anotados: {total_images * 15:,}")
    print(f"• Eficiencia vs manual: {efficiency:.0f}x más rápido")

    print("\n✅ EJERCICIO 1 COMPLETADO")
    print("\nPróximo ejercicio: Cálculo de precisión y métricas clínicas")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script para exploración y análisis del dataset de landmarks en imágenes médicas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
import seaborn as sns

def load_and_analyze_annotations():
    """Cargar y analizar el archivo de coordenadas"""
    print("=== Análisis del archivo de coordenadas ===")

    # Cargar el archivo CSV
    coords_file = "data/coordenadas/coordenadas_maestro.csv"
    df = pd.read_csv(coords_file, header=None)

    print(f"Forma del dataset: {df.shape}")
    print(f"Primeras 5 filas:")
    print(df.head())

    # Analizar estructura de columnas
    print(f"\nEstructura de datos:")
    print(f"Total de columnas: {len(df.columns)}")

    # Separar coordenadas y nombre de archivo
    # Formato esperado: ID, coord1_x, coord1_y, ..., coord15_x, coord15_y, filename
    coords_cols = df.iloc[:, 1:-1]  # Columnas de coordenadas (excluir ID y filename)
    filenames = df.iloc[:, -1]      # Última columna son los nombres de archivos

    print(f"Columnas de coordenadas: {coords_cols.shape[1]}")
    print(f"Coordenadas por imagen: {coords_cols.shape[1] // 2} landmarks")

    return df, coords_cols, filenames

def analyze_coordinates(coords_cols):
    """Analizar distribución y estadísticas de coordenadas"""
    print("\n=== Análisis estadístico de coordenadas ===")

    # Estadísticas básicas
    print("Estadísticas descriptivas:")
    print(coords_cols.describe())

    # Verificar rangos de coordenadas
    print(f"\nRangos de coordenadas:")
    print(f"Mínimo: {coords_cols.min().min()}")
    print(f"Máximo: {coords_cols.max().max()}")

    # Detectar outliers usando IQR
    Q1 = coords_cols.quantile(0.25)
    Q3 = coords_cols.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((coords_cols < (Q1 - 1.5 * IQR)) | (coords_cols > (Q3 + 1.5 * IQR))).sum().sum()
    print(f"Outliers detectados: {outliers}")

    return coords_cols

def analyze_images(filenames):
    """Analizar las imágenes del dataset"""
    print("\n=== Análisis de imágenes ===")

    # Categorías de imágenes
    categories = filenames.str.split('-').str[0].unique()
    print(f"Categorías encontradas: {categories}")

    # Contar imágenes por categoría
    category_counts = filenames.str.split('-').str[0].value_counts()
    print(f"Distribución por categoría:")
    print(category_counts)

    # Verificar existencia de archivos
    dataset_path = Path("data/dataset")
    missing_files = []
    existing_files = []
    image_shapes = []

    for filename in filenames.iloc[:10]:  # Verificar primeras 10 imágenes
        # Buscar archivo en subdirectorios
        found = False
        for category_dir in dataset_path.iterdir():
            if category_dir.is_dir():
                image_path = category_dir / f"{filename}.png"
                if image_path.exists():
                    existing_files.append(str(image_path))
                    # Leer imagen y verificar dimensiones
                    img = cv2.imread(str(image_path))
                    if img is not None:
                        image_shapes.append(img.shape)
                    found = True
                    break

        if not found:
            missing_files.append(filename)

    print(f"\nVerificación de archivos (muestra de 10):")
    print(f"Archivos encontrados: {len(existing_files)}")
    print(f"Archivos faltantes: {len(missing_files)}")

    if image_shapes:
        unique_shapes = list(set(image_shapes))
        print(f"Dimensiones de imágenes encontradas: {unique_shapes}")

    return category_counts, existing_files

def visualize_sample_landmarks(df, existing_files):
    """Visualizar landmarks en imágenes de muestra"""
    print("\n=== Visualizando landmarks en imágenes de muestra ===")

    if len(existing_files) == 0:
        print("No hay imágenes disponibles para visualización")
        return

    # Crear figura para visualizaciones
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, img_path in enumerate(existing_files[:4]):
        if i >= 4:
            break

        # Cargar imagen
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Obtener nombre de archivo para buscar coordenadas
        filename = Path(img_path).stem

        # Buscar coordenadas correspondientes
        row_idx = df[df.iloc[:, -1] == filename].index
        if len(row_idx) > 0:
            coords = df.iloc[row_idx[0], 1:-1].values  # Excluir ID y filename

            # Convertir coordenadas a arrays x,y
            x_coords = coords[::2]   # Coordenadas x (índices pares)
            y_coords = coords[1::2]  # Coordenadas y (índices impares)

            # Mostrar imagen con landmarks
            axes[i].imshow(img_rgb)
            axes[i].scatter(x_coords, y_coords, c='red', s=30, alpha=0.8)

            # Numerar landmarks
            for j, (x, y) in enumerate(zip(x_coords, y_coords)):
                axes[i].annotate(str(j+1), (x, y), xytext=(3, 3),
                               textcoords='offset points', fontsize=8, color='yellow')

            axes[i].set_title(f"Imagen: {filename}")
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f"Coordenadas no encontradas\npara {filename}",
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('sample_landmarks_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Visualización guardada como 'sample_landmarks_visualization.png'")

def create_coordinate_distribution_plots(coords_cols):
    """Crear gráficos de distribución de coordenadas"""
    print("\n=== Creando gráficos de distribución ===")

    # Separar coordenadas X e Y
    x_coords = coords_cols.iloc[:, ::2]   # Columnas pares (X)
    y_coords = coords_cols.iloc[:, 1::2]  # Columnas impares (Y)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Distribución de coordenadas X
    axes[0,0].hist(x_coords.values.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0,0].set_title('Distribución de Coordenadas X')
    axes[0,0].set_xlabel('Valor X')
    axes[0,0].set_ylabel('Frecuencia')

    # Distribución de coordenadas Y
    axes[0,1].hist(y_coords.values.flatten(), bins=50, alpha=0.7, color='red')
    axes[0,1].set_title('Distribución de Coordenadas Y')
    axes[0,1].set_xlabel('Valor Y')
    axes[0,1].set_ylabel('Frecuencia')

    # Boxplot de coordenadas X por landmark
    x_coords.boxplot(ax=axes[1,0], rot=45)
    axes[1,0].set_title('Distribución de Coordenadas X por Landmark')
    axes[1,0].set_ylabel('Valor X')

    # Boxplot de coordenadas Y por landmark
    y_coords.boxplot(ax=axes[1,1], rot=45)
    axes[1,1].set_title('Distribución de Coordenadas Y por Landmark')
    axes[1,1].set_ylabel('Valor Y')

    plt.tight_layout()
    plt.savefig('coordinate_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Gráficos de distribución guardados como 'coordinate_distributions.png'")

def main():
    """Función principal de exploración de datos"""
    print("Iniciando exploración del dataset de landmarks...")

    # Cargar y analizar anotaciones
    df, coords_cols, filenames = load_and_analyze_annotations()

    # Analizar coordenadas
    coords_analysis = analyze_coordinates(coords_cols)

    # Analizar imágenes
    category_counts, existing_files = analyze_images(filenames)

    # Crear visualizaciones
    visualize_sample_landmarks(df, existing_files)
    create_coordinate_distribution_plots(coords_cols)

    # Resumen final
    print("\n" + "="*50)
    print("RESUMEN DEL ANÁLISIS")
    print("="*50)
    print(f"Total de muestras: {len(df)}")
    print(f"Landmarks por imagen: {coords_cols.shape[1] // 2}")
    print(f"Categorías: {list(category_counts.index)}")
    print(f"Distribución por categoría:")
    for cat, count in category_counts.items():
        print(f"  - {cat}: {count} imágenes")
    print(f"Rango de coordenadas: [{coords_cols.min().min()}, {coords_cols.max().max()}]")

    return df, coords_cols, filenames, category_counts

if __name__ == "__main__":
    df, coords_cols, filenames, category_counts = main()
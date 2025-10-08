# 🖼️ GUÍA DE VISUALIZACIONES - LANDMARK REGRESSION

## 📋 RESUMEN

Esta guía documenta todos los comandos de visualización disponibles en el proyecto de regresión de landmarks médicos, incluyendo el nuevo comando para el modelo Phase 4 Complete Loss.

---

## 🎯 COMANDOS DE VISUALIZACIÓN DISPONIBLES

### **1. Visualización Individual**
```bash
python main.py visualize --checkpoint checkpoints/[modelo].pt --image [ruta_imagen]
```
- **Propósito**: Visualizar predicciones en una imagen específica
- **Uso**: Análisis individual y debug
- **Output**: Visualización única con landmarks

### **2. Visualización Test Set Estándar**
```bash
python main.py visualize_test --checkpoint checkpoints/phase2_best.pt
```
- **Propósito**: Generar visualizaciones del test set para modelos estándar
- **Uso**: Evaluación visual del modelo baseline
- **Output**: Visualizaciones en `evaluation_results/test_predictions/`

### **3. Visualización Test Set Symmetry**
```bash
python main.py visualize_test --checkpoint checkpoints/geometric_symmetry.pt
```
- **Propósito**: Generar visualizaciones del modelo Phase 3 Symmetry
- **Uso**: Evaluar efectividad de restricciones bilaterales
- **Output**: Visualizaciones en `evaluation_results/test_predictions_symmetry/`
- **Auto-detección**: Detecta automáticamente modelo symmetry

### **4. Visualización Test Set Complete Loss (NUEVO)**
```bash
python main.py visualize_test_complete_loss
```
- **Propósito**: Generar visualizaciones del modelo Phase 4 Complete Loss
- **Uso**: Evaluar modelo con máxima precisión clínica
- **Output**: Visualizaciones en `evaluation_results/test_predictions_complete_loss/`
- **Características especiales**: Nombres descriptivos con categoría médica

---

## 🔍 COMANDO PHASE 4 COMPLETE LOSS - DETALLE

### **Sintaxis Completa**
```bash
python main.py visualize_test_complete_loss
```

### **Características Técnicas**
- **Modelo**: `checkpoints/geometric_complete.pt`
- **Error promedio**: 8.13px (excelencia clínica)
- **Total imágenes**: 144 (conjunto de test completo)
- **Tiempo ejecución**: ~1.3 minutos
- **Resolución**: 12x12 inches, 150 DPI

### **Distribución por Categorías**
- **COVID**: 38 imágenes
- **Normal**: 83 imágenes
- **Viral Pneumonia**: 23 imágenes

### **Formato de Nombres Generados**
```
COVID_COVID-1028_error_14.80px.png
Normal_Normal-10078_error_7.14px.png
Viral_Pneumonia_Viral Pneumonia-101_error_12.73px.png
```

### **Estructura del Nombre**
- **Categoría médica**: COVID / Normal / Viral_Pneumonia
- **ID original**: Número identificador del dataset
- **Error específico**: Precisión en píxeles del modelo

### **Elementos Visuales**
- **Ground Truth**: Landmarks en verde con bordes oscuros
- **Predicciones**: Landmarks en rojo con marcadores X
- **Líneas de error**: Conexiones amarillas mostrando desviaciones
- **Grid**: Cuadrícula de referencia para análisis preciso
- **Título**: Formato "Category: ImageID - Error: X.XXpx (Complete Loss Model)"

---

## 📊 COMPARACIÓN DE VISUALIZACIONES

### **Baseline vs Phase 4**
| Aspecto | Baseline | Phase 4 Complete Loss |
|---------|----------|----------------------|
| **Error promedio** | 11.34px | 8.13px |
| **Precisión clínica** | Buena | Excelente |
| **Identificación** | Genérica | Categoría médica |
| **Metadata** | Básica | Completa |

### **Distribución de Calidad Phase 4**
```
Excelente (<5px):     25 muestras (17.4%)
Muy bueno (5-8.5px):  71 muestras (49.3%)
Bueno (8.5-15px):     40 muestras (27.8%)
Aceptable (≥15px):     8 muestras (5.6%)
```

---

## 🛠️ SOLUCIÓN DE PROBLEMAS

### **Problema: Nombres aparecen como "Unknown"**
- **Causa**: Problemas con metadata del DataLoader
- **Solución**: Implementada custom collate function
- **Estado**: ✅ RESUELTO en versión actual

### **Problema: Checkpoint no encontrado**
```bash
❌ Checkpoint Complete Loss no encontrado: checkpoints/geometric_complete.pt
💡 Ejecuta primero: python main.py train_geometric_complete
```
- **Solución**: Entrenar modelo Phase 4 primero

### **Problema: Errores de memoria**
- **Causa**: Batch size muy grande o GPU insuficiente
- **Solución**: Script usa batch_size=1 automáticamente
- **GPU mínima**: 4GB (recomendado 8GB)

---

## 📁 ESTRUCTURA DE OUTPUTS

### **Directorio Base**
```
evaluation_results/
├── test_predictions/                    # Modelos estándar
├── test_predictions_symmetry/           # Phase 3 Symmetry
└── test_predictions_complete_loss/      # Phase 4 Complete Loss
```

### **Contenido Complete Loss**
```
test_predictions_complete_loss/
├── COVID_COVID-1028_error_14.80px.png
├── COVID_COVID-1054_error_12.16px.png
├── Normal_Normal-10078_error_7.14px.png
├── Normal_Normal-1016_error_5.47px.png
├── Viral_Pneumonia_Viral Pneumonia-101_error_12.73px.png
└── ... (144 total)
```

---

## 🚀 CASOS DE USO

### **1. Análisis Clínico**
```bash
# Generar todas las visualizaciones
python main.py visualize_test_complete_loss

# Filtrar por categoría
ls evaluation_results/test_predictions_complete_loss/COVID_* | head -10
ls evaluation_results/test_predictions_complete_loss/Normal_* | head -10
```

### **2. Evaluación de Calidad**
```bash
# Casos excelentes (<5px)
ls evaluation_results/test_predictions_complete_loss/ | grep "error_[0-4]"

# Casos problemáticos (>15px)
ls evaluation_results/test_predictions_complete_loss/ | grep "error_[1-9][5-9]"
```

### **3. Comparación Entre Modelos**
```bash
# Generar visualizaciones de diferentes modelos
python main.py visualize_test --checkpoint checkpoints/geometric_symmetry.pt
python main.py visualize_test_complete_loss

# Comparar directorios
ls evaluation_results/test_predictions_symmetry/ | wc -l
ls evaluation_results/test_predictions_complete_loss/ | wc -l
```

---

## 💡 MEJORAS FUTURAS POSIBLES

### **Filtrado Avanzado**
- Generar visualizaciones solo para errores <5px
- Separar por categoría médica en subdirectorios
- Filtrar por rango de error específico

### **Análisis Comparativo**
- Script para comparar visualizaciones entre modelos
- Métricas por landmark individual
- Análisis de outliers automático

### **Formatos Adicionales**
- Export a PDF con múltiples imágenes por página
- Generación de video con progresión de errores
- Visualizaciones interactivas con HTML

---

## 📞 SOPORTE TÉCNICO

### **Información del Sistema**
- **GPU requerida**: AMD/NVIDIA con 4GB+ VRAM
- **Tiempo típico**: 1-2 minutos para 144 imágenes
- **Espacio en disco**: ~30MB para visualizaciones completas

### **Contacto y Debugging**
- Los errores se reportan automáticamente en consola
- Metadata preservada con custom collate function
- Scripts robustos sin dependencias adicionales

---

*Documentación actualizada para Phase 4 Complete Loss*
*Última actualización: Septiembre 23, 2024*
*Estado: Funcionalidad completa y optimizada*
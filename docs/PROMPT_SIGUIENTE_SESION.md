# 📚 PROMPT PARA PRÓXIMA SESIÓN: COMPRENSIÓN Y DOMINIO DEL PROYECTO

## Prompt Recomendado:

```
Necesito que me ayudes a comprender profundamente el proyecto de "Medical Landmarks Prediction
with Deep Learning" para poder explicarlo con claridad.

Este es un sistema de regresión de landmarks anatómicos en rayos-X de tórax que alcanzó
excelencia clínica (8.29px) usando ResNet-18 y un pipeline de 4 fases con losses geométricos.

Por favor, guíame paso a paso a través de:

1. **Arquitectura del Sistema (30 min)**
   - Explicar cómo funciona el modelo ResNet-18 modificado
   - Detallar la cabeza de regresión personalizada
   - Mostrar el flujo de datos: imagen → features → coordenadas
   - Explicar por qué funciona transfer learning desde ImageNet

2. **Pipeline de 4 Fases (45 min)**
   - Entender la lógica detrás de cada fase
   - Por qué Phase 1 congela el backbone
   - Qué hace Wing Loss diferente a MSE
   - Cómo funciona Symmetry Loss y por qué mejora tanto (+21.4%)
   - Qué añade Distance Preservation Loss en Phase 4
   - Ver ejemplos de código de cada componente

3. **Implementación Práctica (30 min)**
   - Cómo entrenar desde cero el pipeline completo
   - Cómo evaluar y generar visualizaciones
   - Cómo interpretar los resultados (8.29px vs 8.5px target)
   - Dónde están los archivos clave y qué hace cada uno

4. **Conceptos Médicos (20 min)**
   - Qué son los 15 landmarks anatómicos y por qué importan
   - Qué es excelencia clínica (<8.5px) y cómo se mide
   - Pares simétricos bilaterales y eje mediastinal
   - Aplicaciones reales: ICT, detección de asimetrías

5. **Ejercicios Prácticos (30 min)**
   - Analizar 3-5 visualizaciones del test set (buenas y malas)
   - Explicar por qué algunos casos tienen error >15px
   - Modificar un hiperparámetro y predecir el efecto
   - Diseñar un experimento para mejorar a <8px

Usa:
- Analogías simples para conceptos complejos
- Diagramas en ASCII cuando ayude
- Ejemplos de código comentados
- Referencias específicas a archivos del proyecto (líneas de código)

Al final, debería poder explicar este proyecto en una presentación de 15 minutos
cubriendo: problema, solución técnica, resultados y aplicación clínica.

Archivos clave disponibles:
- CLAUDE.md (contexto completo del proyecto)
- PIPELINE_RESULTS.md (resultados recientes)
- main.py (CLI con todos los comandos)
- src/models/losses.py (Wing, Symmetry, Distance Loss)
- checkpoints/geometric_complete.pt (modelo final 8.29px)
```

---

## 🎯 Objetivos de Aprendizaje

Al finalizar la sesión deberías poder:

✅ Explicar la arquitectura completa sin ver código
✅ Justificar cada decisión de diseño (por qué 4 fases, por qué estos losses)
✅ Entrenar y evaluar el modelo desde cero
✅ Interpretar visualizaciones y diagnosticar errores
✅ Presentar el proyecto a audiencia técnica y médica

---

## 📋 Checklist de Preparación

Antes de iniciar la sesión:
- [ ] Lee `CLAUDE.md` (visión general del proyecto)
- [ ] Revisa `PIPELINE_RESULTS.md` (resultados actuales)
- [ ] Explora `evaluation_results/test_predictions_complete_loss/` (ver 5-10 visualizaciones)
- [ ] Ten a mano papel para tomar notas y diagramas

---

## 🔄 Prompt Alternativo (Más Interactivo)

Si prefieres un enfoque más práctico:

```
Quiero aprender este proyecto de landmarks médicos haciéndolo paso a paso.

Empecemos por:
1. Mostrarme una visualización del test set y explicar qué veo
2. Desde ahí, construir hacia atrás: cómo el modelo genera esos puntos
3. Luego profundizar en cada componente técnico según necesite

Prefiero aprender haciendo preguntas y resolviendo problemas reales del código,
no solo teoría. Guíame como si fuera un código review educativo.

Archivos base:
- CLAUDE.md
- PIPELINE_RESULTS.md
- evaluate_complete.py (para entender evaluación)
- train_complete_simple.py (para entender Phase 4)
```

---

## 💡 Tips para Máximo Aprendizaje

1. **Empieza visual:** Abre visualizaciones antes de ver código
2. **Pregunta "por qué":** No solo "qué hace" sino "por qué esta decisión"
3. **Compara fases:** Ver diff entre Phase 2 y Phase 3 para entender Symmetry Loss
4. **Experimenta:** Cambia un peso del loss y predice el efecto
5. **Enseña back:** Explica a Claude lo que entendiste (Feynman technique)

---

## 📊 Métricas de Éxito de la Sesión

Al terminar, deberías poder responder:

- ¿Por qué Wing Loss es mejor que MSE para landmarks?
- ¿Cómo funciona Symmetry Loss matemáticamente?
- ¿Qué landmarks son más difíciles y por qué?
- ¿Cómo se calcula el error de 8.29px?
- ¿Qué harías para mejorar a <8px?

---

## 🎓 Recursos de Referencia Durante la Sesión

```bash
# Comandos útiles para explorar mientras aprendes:

# Ver arquitectura del modelo
python -c "from src.models.resnet_regressor import ResNetLandmarkRegressor; \
           model = ResNetLandmarkRegressor(num_landmarks=15); \
           print(model)"

# Ver una predicción paso a paso (con breakpoints)
python -m pdb evaluate_complete.py

# Comparar checkpoints
python -c "import torch; \
           p3 = torch.load('checkpoints/geometric_symmetry.pt'); \
           p4 = torch.load('checkpoints/geometric_complete.pt'); \
           print(f'P3: {p3[\"metrics\"]}'); \
           print(f'P4: epoch {p4[\"epoch\"]}')"

# Ver distribución de errores por landmark
python -c "from evaluate_complete import *; \
           # código para analizar errores por landmark"
```

---

**Duración estimada:** 2-3 horas
**Nivel:** Intermedio-Avanzado
**Output esperado:** Comprensión completa + capacidad de explicación + ideas de mejora

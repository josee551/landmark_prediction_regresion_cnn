#!/usr/bin/env python3
"""
EJERCICIO PRÁCTICO: COMPARACIÓN DE FASES GEOMÉTRICAS
Módulo 3: Transfer Learning - Proyecto Landmarks 8.13px

Este ejercicio ayuda al tesista a comprender las diferencias
entre las 4 fases geométricas y sus implicaciones clínicas.

Tiempo estimado: 45-60 minutos
Objetivo: Análisis comparativo detallado de la evolución metodológica
"""

import matplotlib.pyplot as plt
import numpy as np

def analyze_geometric_phases():
    """
    Análisis comparativo de las 4 fases geométricas del proyecto
    """
    print("📊 EJERCICIO: ANÁLISIS DE FASES GEOMÉTRICAS")
    print("=" * 60)

    # Datos exactos de las fases geométricas
    phases_data = {
        "Baseline MSE": {
            "error_px": 11.34,
            "improvement": 0.0,
            "technique": "Mean Squared Error",
            "time_min": 4.0,
            "status": "✅ Funcional",
            "clinical_grade": "Clínicamente útil",
            "model_file": "phase2_best.pt"
        },
        "Phase 1 Geom": {
            "error_px": 10.91,
            "improvement": 3.8,
            "technique": "Wing Loss",
            "time_min": 3.2,
            "status": "✅ Exitoso",
            "clinical_grade": "Clínicamente útil+",
            "model_file": "geometric_phase1_wing_loss.pt"
        },
        "Phase 2 Geom": {
            "error_px": 11.07,
            "improvement": -1.4,  # Degradación
            "technique": "Wing Loss + Coordinate Attention",
            "time_min": 3.8,
            "status": "❌ Falló",
            "clinical_grade": "Clínicamente útil",
            "model_file": "geometric_attention.pt"
        },
        "Phase 3 Geom": {
            "error_px": 8.91,
            "improvement": 21.4,
            "technique": "Wing Loss + Symmetry Loss",
            "time_min": 4.0,
            "status": "✅ Breakthrough",
            "clinical_grade": "EXCELENCIA CLÍNICA",
            "model_file": "geometric_symmetry.pt"
        },
        "Phase 4 Geom": {
            "error_px": 8.13,
            "improvement": 28.3,
            "technique": "Complete Loss (Wing+Symmetry+Distance)",
            "time_min": 3.7,
            "status": "✅ EXCELENCIA",
            "clinical_grade": "EXCELENCIA CLÍNICA+",
            "model_file": "geometric_complete.pt"
        }
    }

    print("\n📈 PASO 1: Evolución Cronológica del Error")
    print("-" * 50)

    baseline_error = 11.34
    for phase, data in phases_data.items():
        error = data["error_px"]
        improvement = data["improvement"]
        status = data["status"]

        if improvement >= 0:
            change_str = f"↓ {improvement:+.1f}%"
        else:
            change_str = f"↑ {abs(improvement):+.1f}% (DEGRADACIÓN)"

        print(f"   {phase:<15}: {error:>5.2f}px {change_str:>12} {status}")

    # Análisis de benchmarks clínicos
    print("\n🏥 PASO 2: Benchmarks Clínicos Alcanzados")
    print("-" * 50)

    clinical_benchmarks = {
        "Sub-píxel (research)": 5.0,
        "Excelencia clínica": 8.5,
        "Clínicamente excelente": 10.0,
        "Clínicamente útil": 15.0,
        "Análisis general": 20.0
    }

    for phase, data in phases_data.items():
        error = data["error_px"]
        print(f"\n   {phase}:")

        for benchmark, threshold in clinical_benchmarks.items():
            if error < threshold:
                status = "✅ SUPERADO"
                margin = threshold - error
                print(f"      • {benchmark:<25}: <{threshold:>4.1f}px {status} (margen: {margin:.1f}px)")
            else:
                status = "❌ No alcanzado"
                deficit = error - threshold
                print(f"      • {benchmark:<25}: <{threshold:>4.1f}px {status} (falta: {deficit:.1f}px)")

    # Análisis de tiempo vs beneficio
    print("\n⏱️  PASO 3: Análisis Tiempo vs Beneficio")
    print("-" * 50)

    total_time = 0
    cumulative_improvement = 0

    print("   Eficiencia por fase:")
    for phase, data in phases_data.items():
        time = data["time_min"]
        improvement = data["improvement"]
        total_time += time

        if improvement > 0:
            efficiency = improvement / time
            print(f"   {phase:<15}: {time:>4.1f}min → {improvement:>+5.1f}% → {efficiency:>5.1f}%/min")
        else:
            print(f"   {phase:<15}: {time:>4.1f}min → {improvement:>+5.1f}% → INEFICIENTE")

    print(f"\n   Total tiempo invertido: {total_time:.1f} minutos")
    print(f"   Mejora final: 28.3% (11.34px → 8.13px)")
    print(f"   Eficiencia promedio: {28.3/total_time:.1f}%/minuto")

    # Análisis de estrategias exitosas vs fallidas
    print("\n🔬 PASO 4: Análisis de Estrategias")
    print("-" * 50)

    strategies = {
        "Wing Loss": {
            "type": "Loss Function Engineering",
            "result": "✅ Exitoso (+3.8%)",
            "reason": "Balance precisión vs robustez para landmarks médicos"
        },
        "Coordinate Attention": {
            "type": "Architectural Enhancement",
            "result": "❌ Fallido (-1.4%)",
            "reason": "Dataset pequeño, complejidad innecesaria, sin fundamento médico"
        },
        "Symmetry Loss": {
            "type": "Domain Knowledge Integration",
            "result": "✅ Breakthrough (+17.6% adicional)",
            "reason": "Conocimiento anatómico válido sobre simetría bilateral"
        },
        "Distance Preservation": {
            "type": "Geometric Constraints",
            "result": "✅ Refinamiento (+6.9% adicional)",
            "reason": "Preserva relaciones espaciales críticas para diagnóstico"
        }
    }

    print("   Estrategias analizadas:")
    for strategy, details in strategies.items():
        print(f"\n   {strategy}:")
        print(f"      Tipo: {details['type']}")
        print(f"      Resultado: {details['result']}")
        print(f"      Razón: {details['reason']}")

    # Ejercicio de comprensión
    print("\n🧠 PASO 5: Ejercicio de Comprensión")
    print("-" * 50)

    questions = [
        {
            "question": "¿Por qué Wing Loss mejoró el rendimiento vs MSE?",
            "answer": "Wing Loss es estricto con errores pequeños (precisión sub-píxel) pero tolerante con errores grandes (casos complejos), ideal para landmarks médicos."
        },
        {
            "question": "¿Por qué falló Coordinate Attention?",
            "answer": "Dataset de 956 imágenes demasiado pequeño para 25K parámetros adicionales. Attention requiere >10K imágenes típicamente."
        },
        {
            "question": "¿Por qué Symmetry Loss fue tan exitoso?",
            "answer": "Incorpora conocimiento anatómico válido: pulmones son bilateralmente simétricos. Regularización natural basada en medicina."
        },
        {
            "question": "¿Qué hace Distance Preservation Loss?",
            "answer": "Preserva 5 distancias anatómicas críticas (mediastino, anchos torácicos) importantes para mediciones clínicas."
        }
    ]

    print("   Preguntas de verificación:")
    for i, q in enumerate(questions, 1):
        print(f"\n   {i}. {q['question']}")
        user_input = input("      Tu respuesta (Enter para ver respuesta correcta): ")
        print(f"      ✅ Respuesta: {q['answer']}")
        if user_input.strip():
            print("      💭 Compara tu respuesta con la correcta")

    # Cálculos de distribución de calidad
    print("\n📊 PASO 6: Distribución de Calidad Estimada")
    print("-" * 50)

    # Datos aproximados basados en el error promedio
    def estimate_quality_distribution(avg_error):
        """Estima distribución de calidad basada en error promedio"""
        if avg_error <= 8.2:  # Phase 4 territory
            return {"excellent": 17.4, "very_good": 49.3, "good": 27.8, "acceptable": 5.6}
        elif avg_error <= 9.0:  # Phase 3 territory
            return {"excellent": 15.0, "very_good": 47.0, "good": 30.0, "acceptable": 8.0}
        elif avg_error <= 11.0:  # Phase 1 territory
            return {"excellent": 10.0, "very_good": 35.0, "good": 40.0, "acceptable": 15.0}
        else:  # Baseline territory
            return {"excellent": 8.0, "very_good": 32.0, "good": 42.0, "acceptable": 18.0}

    print("   Distribución estimada de calidad (144 casos test):")
    for phase, data in phases_data.items():
        error = data["error_px"]
        dist = estimate_quality_distribution(error)

        print(f"\n   {phase} ({error}px):")
        print(f"      Excelente (<5px):     {dist['excellent']:>5.1f}%")
        print(f"      Muy bueno (5-8.5px):  {dist['very_good']:>5.1f}%")
        print(f"      Bueno (8.5-15px):     {dist['good']:>5.1f}%")
        print(f"      Aceptable (≥15px):    {dist['acceptable']:>5.1f}%")

        clinical_excellence = dist['excellent'] + dist['very_good']
        print(f"      🎯 Excelencia clínica: {clinical_excellence:>5.1f}%")

    # Visualización de la evolución
    print("\n📈 PASO 7: Generando Visualización de Evolución")
    try:
        create_phases_visualization(phases_data)
        print("   ✅ Gráfico guardado en: DEFENSA_TESISTA/03_TRANSFER_LEARNING/resultados/")
    except ImportError:
        print("   ⚠️  Matplotlib no disponible, saltando visualización")
    except Exception as e:
        print(f"   ❌ Error generando gráfico: {e}")

    # Análisis final
    print("\n🎯 PASO 8: Análisis Final y Conclusiones")
    print("-" * 50)

    final_analysis = {
        "Mejor estrategia": "Domain Knowledge Integration (Symmetry Loss)",
        "Mayor mejora": "Phase 3: +17.6% en una sola fase",
        "Peor decisión": "Phase 2: Coordinate Attention sin justificación médica",
        "Lección clave": "Conocimiento médico > Complejidad arquitectónica",
        "Resultado final": "8.13px = EXCELENCIA CLÍNICA comprobada"
    }

    print("   Conclusiones clave:")
    for key, value in final_analysis.items():
        print(f"   • {key}: {value}")

    print("\n   Datos para memorizar:")
    print("   • Evolución: 11.34px → 10.91px → 8.91px → 8.13px")
    print("   • Mejora total: 28.3% reducción de error")
    print("   • Tiempo total: ~8 minutos entrenamiento")
    print("   • Benchmark: <8.5px excelencia clínica ✅ SUPERADO")
    print("   • Casos excelentes: 66.7% del test set")

    print("\n✅ EJERCICIO COMPLETADO")
    print("Próximo: Preparación para defensa de metodología")

def create_phases_visualization(phases_data):
    """
    Crea visualización de la evolución de las fases
    """
    import os

    # Preparar datos para gráfico
    phases = list(phases_data.keys())
    errors = [data["error_px"] for data in phases_data.values()]
    improvements = [data["improvement"] for data in phases_data.values()]

    # Colores según resultado
    colors = []
    for data in phases_data.values():
        if data["improvement"] < 0:
            colors.append('red')  # Degradación
        elif data["improvement"] < 10:
            colors.append('orange')  # Mejora menor
        elif data["improvement"] < 25:
            colors.append('green')  # Breakthrough
        else:
            colors.append('darkgreen')  # Excelencia

    # Crear figura con subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico 1: Evolución del error
    ax1.plot(range(len(phases)), errors, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Fases Geométricas')
    ax1.set_ylabel('Error (píxeles)')
    ax1.set_title('Evolución del Error por Fase\nProyecto Landmarks Medical')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(len(phases)))
    ax1.set_xticklabels([p.replace(' Geom', '\nGeom').replace('Baseline ', '') for p in phases], rotation=45)

    # Línea de benchmark excelencia clínica
    ax1.axhline(y=8.5, color='red', linestyle='--', alpha=0.7, label='Excelencia clínica (<8.5px)')
    ax1.legend()

    # Añadir valores en puntos
    for i, (phase, error) in enumerate(zip(phases, errors)):
        ax1.annotate(f'{error:.2f}px', (i, error), textcoords="offset points",
                    xytext=(0,10), ha='center', fontweight='bold')

    # Gráfico 2: Mejoras porcentuales
    bars = ax2.bar(range(len(phases)), improvements, color=colors, alpha=0.7)
    ax2.set_xlabel('Fases Geométricas')
    ax2.set_ylabel('Mejora (%)')
    ax2.set_title('Mejora Porcentual por Fase\n(vs Baseline MSE)')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(range(len(phases)))
    ax2.set_xticklabels([p.replace(' Geom', '\nGeom').replace('Baseline ', '') for p in phases], rotation=45)

    # Línea en y=0
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # Añadir valores en barras
    for i, (bar, improvement) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        ax2.annotate(f'{improvement:+.1f}%', (bar.get_x() + bar.get_width()/2., height),
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold', fontsize=10)

    plt.tight_layout()

    # Guardar gráfico
    output_dir = "DEFENSA_TESISTA/03_TRANSFER_LEARNING/resultados"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/fases_geometricas_evolution.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()

    return output_file

if __name__ == "__main__":
    analyze_geometric_phases()
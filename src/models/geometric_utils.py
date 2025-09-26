"""
Utilidades geométricas para análisis de landmarks anatómicos pulmonares

Este módulo proporciona herramientas para:
- Análisis de simetría bilateral
- Cálculo de distancias anatómicas críticas
- Validación de constraints geométricos
- Métricas de coherencia estructural
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class GeometricAnalyzer:
    """
    Analizador geométrico para landmarks pulmonares

    Proporciona funciones para evaluar la coherencia anatómica
    de las predicciones de landmarks
    """

    def __init__(self, num_landmarks: int = 15):
        """
        Args:
            num_landmarks: Número de landmarks (15 para pulmones)
        """
        self.num_landmarks = num_landmarks

        # Definir pares simétricos anatómicos (índices de landmarks)
        # Basado en la estructura pulmonar bilateral identificada
        self.symmetric_pairs = [
            (2, 3),   # Ápices pulmonares izquierdo-derecho
            (4, 5),   # Hilios izquierdo-derecho
            (6, 7),   # Bases pulmonares izquierdo-derecho
            (11, 12), # Bordes costales superiores
            (13, 14)  # Senos costofrénicos
        ]

        # Landmarks del eje mediastinal central (anatomía estable)
        self.mediastinal_landmarks = [0, 1, 8, 9, 10]  # landmarks 1,2,9,10,11

        # Distancias anatómicas críticas (landmark_idx1, landmark_idx2, descripción)
        self.critical_distances = [
            (0, 1, "mediastino_superior_inferior"),
            (8, 9, "eje_central_medio"),
            (2, 3, "ancho_toracico_superior"),
            (4, 5, "ancho_toracico_medio"),
            (6, 7, "ancho_toracico_inferior"),
            (0, 8, "altura_mediastinal_superior"),
            (1, 9, "altura_mediastinal_inferior")
        ]

    def analyze_batch(self, landmarks_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Análisis geométrico completo de un batch de landmarks

        Args:
            landmarks_batch: Tensor de forma (batch_size, 30) con coordenadas [x1,y1,...,x15,y15]

        Returns:
            Diccionario con métricas geométricas
        """
        # Convertir a formato (batch_size, 15, 2)
        landmarks = landmarks_batch.view(-1, self.num_landmarks, 2)

        results = {}

        # 1. Análisis de simetría bilateral
        results.update(self._analyze_symmetry(landmarks))

        # 2. Análisis de distancias anatómicas
        results.update(self._analyze_distances(landmarks))

        # 3. Validación de constraints
        results.update(self._validate_constraints(landmarks))

        # 4. Métricas de forma
        results.update(self._analyze_shape(landmarks))

        return results

    def _analyze_symmetry(self, landmarks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Análisis de simetría bilateral"""
        batch_size = landmarks.size(0)
        symmetry_errors = []

        for left_idx, right_idx in self.symmetric_pairs:
            # Coordenadas x de landmarks simétricos
            left_x = landmarks[:, left_idx, 0]
            right_x = landmarks[:, right_idx, 0]

            # Calcular centro de masa en x
            center_x = torch.mean(landmarks[:, :, 0], dim=1)

            # Error de simetría: |distancia_izq - distancia_der| al centro
            left_dist = torch.abs(left_x - center_x)
            right_dist = torch.abs(right_x - center_x)

            symmetry_error = torch.abs(left_dist - right_dist)
            symmetry_errors.append(symmetry_error)

        # Métricas agregadas
        symmetry_errors = torch.stack(symmetry_errors, dim=1)  # (batch_size, num_pairs)

        return {
            'symmetry_errors': symmetry_errors,
            'symmetry_error_mean': torch.mean(symmetry_errors, dim=1),
            'symmetry_error_max': torch.max(symmetry_errors, dim=1)[0],
            'bilateral_consistency': 1.0 / (1.0 + torch.mean(symmetry_errors, dim=1))
        }

    def _analyze_distances(self, landmarks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Análisis de distancias anatómicas críticas"""
        batch_size = landmarks.size(0)
        distances = {}

        for idx1, idx2, name in self.critical_distances:
            # Calcular distancia euclidiana entre landmarks
            point1 = landmarks[:, idx1, :]  # (batch_size, 2)
            point2 = landmarks[:, idx2, :]  # (batch_size, 2)

            distance = torch.norm(point1 - point2, dim=1)  # (batch_size,)
            distances[f'distance_{name}'] = distance

        # Métricas agregadas
        all_distances = torch.stack(list(distances.values()), dim=1)

        return {
            **distances,
            'distance_mean': torch.mean(all_distances, dim=1),
            'distance_std': torch.std(all_distances, dim=1),
            'distance_variability': torch.std(all_distances, dim=1) / torch.mean(all_distances, dim=1)
        }

    def _validate_constraints(self, landmarks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Validación de constraints anatómicos"""
        batch_size = landmarks.size(0)

        # 1. Constraint de ordenamiento vertical (superior → inferior)
        # Landmarks superiores deben tener y menor que inferiores
        vertical_violations = 0

        # Ápices (2,3) deben estar arriba de bases (6,7)
        apex_y = torch.min(landmarks[:, [2, 3], 1], dim=1)[0]  # Y mínimo de ápices
        base_y = torch.max(landmarks[:, [6, 7], 1], dim=1)[0]  # Y máximo de bases

        vertical_violation = torch.relu(apex_y - base_y)  # Positivo si hay violación

        # 2. Constraint de landmarks centrales (deben estar cerca del eje central)
        center_x = torch.mean(landmarks[:, :, 0], dim=1, keepdim=True)  # (batch_size, 1)

        mediastinal_x = landmarks[:, self.mediastinal_landmarks, 0]  # (batch_size, 5)
        mediastinal_deviations = torch.abs(mediastinal_x - center_x)
        mediastinal_violation = torch.mean(mediastinal_deviations, dim=1)

        # 3. Constraint de simetría aproximada
        symmetry_tolerance = 0.15  # 15% de la imagen
        symmetry_violations = torch.zeros(batch_size, device=landmarks.device)

        for left_idx, right_idx in self.symmetric_pairs:
            left_x = landmarks[:, left_idx, 0]
            right_x = landmarks[:, right_idx, 0]
            center_x_val = center_x.squeeze(1)

            # Distancias al centro
            left_dist = torch.abs(left_x - center_x_val)
            right_dist = torch.abs(right_x - center_x_val)

            # Violación si la diferencia supera tolerancia
            diff = torch.abs(left_dist - right_dist)
            violation = torch.relu(diff - symmetry_tolerance)
            symmetry_violations += violation

        return {
            'vertical_violations': vertical_violation,
            'mediastinal_violations': mediastinal_violation,
            'symmetry_violations': symmetry_violations,
            'total_violations': vertical_violation + mediastinal_violation + symmetry_violations,
            'anatomical_validity': 1.0 / (1.0 + vertical_violation + mediastinal_violation + symmetry_violations)
        }

    def _analyze_shape(self, landmarks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Análisis de forma y proporciones anatómicas"""
        batch_size = landmarks.size(0)

        # 1. Área aproximada del tórax (convex hull simplificado)
        # Usar landmarks extremos para aproximar área
        min_x = torch.min(landmarks[:, :, 0], dim=1)[0]
        max_x = torch.max(landmarks[:, :, 0], dim=1)[0]
        min_y = torch.min(landmarks[:, :, 1], dim=1)[0]
        max_y = torch.max(landmarks[:, :, 1], dim=1)[0]

        thoracic_width = max_x - min_x
        thoracic_height = max_y - min_y
        approximate_area = thoracic_width * thoracic_height

        # 2. Ratio cardiotorácico aproximado (ancho medio / ancho superior)
        upper_width = torch.norm(landmarks[:, 2, :] - landmarks[:, 3, :], dim=1)
        middle_width = torch.norm(landmarks[:, 4, :] - landmarks[:, 5, :], dim=1)

        cardiothoracic_ratio = middle_width / (upper_width + 1e-8)

        # 3. Índice de compacidad
        perimeter = self._estimate_perimeter(landmarks)
        compactness = 4 * math.pi * approximate_area / (perimeter ** 2 + 1e-8)

        # 4. Centroide y dispersión
        centroid = torch.mean(landmarks, dim=1)  # (batch_size, 2)

        # Dispersión desde el centroide
        dispersions = torch.norm(landmarks - centroid.unsqueeze(1), dim=2)
        mean_dispersion = torch.mean(dispersions, dim=1)

        return {
            'thoracic_width': thoracic_width,
            'thoracic_height': thoracic_height,
            'thoracic_area': approximate_area,
            'cardiothoracic_ratio': cardiothoracic_ratio,
            'compactness': compactness,
            'centroid_x': centroid[:, 0],
            'centroid_y': centroid[:, 1],
            'mean_dispersion': mean_dispersion,
            'aspect_ratio': thoracic_width / (thoracic_height + 1e-8)
        }

    def _estimate_perimeter(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Estimación del perímetro usando landmarks extremos"""
        batch_size = landmarks.size(0)

        # Usar landmarks que forman el contorno aproximado
        contour_indices = [2, 4, 6, 14, 13, 7, 5, 3]  # Orden aproximado del contorno

        perimeter = torch.zeros(batch_size, device=landmarks.device)

        for i in range(len(contour_indices)):
            current_idx = contour_indices[i]
            next_idx = contour_indices[(i + 1) % len(contour_indices)]

            segment_length = torch.norm(
                landmarks[:, current_idx, :] - landmarks[:, next_idx, :],
                dim=1
            )
            perimeter += segment_length

        return perimeter

    def compute_geometric_loss(
        self,
        pred_landmarks: torch.Tensor,
        target_landmarks: torch.Tensor,
        weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Compute loss geométrico basado en constraints anatómicos

        Args:
            pred_landmarks: Predicciones (batch_size, 30)
            target_landmarks: Ground truth (batch_size, 30)
            weights: Pesos para diferentes componentes del loss

        Returns:
            Loss geométrico escalar
        """
        if weights is None:
            weights = {
                'symmetry': 0.3,
                'distance': 0.2,
                'constraint': 0.1
            }

        # Análisis de predicciones vs targets
        pred_analysis = self.analyze_batch(pred_landmarks)
        target_analysis = self.analyze_batch(target_landmarks)

        # 1. Loss de simetría
        symmetry_loss = torch.mean(pred_analysis['symmetry_error_mean'])

        # 2. Loss de preservación de distancias
        distance_loss = 0
        for name in ['distance_ancho_toracico_superior', 'distance_ancho_toracico_medio',
                     'distance_mediastino_superior_inferior']:
            if name in pred_analysis and name in target_analysis:
                pred_dist = pred_analysis[name]
                target_dist = target_analysis[name]
                distance_loss += torch.mean(torch.abs(pred_dist - target_dist))

        # 3. Loss de violaciones de constraints
        constraint_loss = torch.mean(pred_analysis['total_violations'])

        # Combinar losses
        total_geometric_loss = (
            weights['symmetry'] * symmetry_loss +
            weights['distance'] * distance_loss +
            weights['constraint'] * constraint_loss
        )

        return total_geometric_loss


def compute_anatomical_consistency_score(landmarks: torch.Tensor) -> torch.Tensor:
    """
    Compute score de consistencia anatómica para landmarks predichos

    Args:
        landmarks: Tensor de landmarks (batch_size, 30)

    Returns:
        Scores de consistencia por muestra (batch_size,) en rango [0,1]
    """
    analyzer = GeometricAnalyzer()
    analysis = analyzer.analyze_batch(landmarks)

    # Combinar métricas en score único
    consistency_score = (
        analysis['bilateral_consistency'] * 0.4 +
        analysis['anatomical_validity'] * 0.6
    )

    return consistency_score


def validate_anatomical_predictions(landmarks: torch.Tensor, threshold: float = 0.7) -> torch.Tensor:
    """
    Validar si las predicciones son anatómicamente plausibles

    Args:
        landmarks: Tensor de landmarks (batch_size, 30)
        threshold: Umbral de validez anatómica

    Returns:
        Máscara booleana indicando predicciones válidas (batch_size,)
    """
    consistency_scores = compute_anatomical_consistency_score(landmarks)
    return consistency_scores >= threshold
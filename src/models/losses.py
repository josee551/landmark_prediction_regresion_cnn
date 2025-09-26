"""
Funciones de pérdida especializadas para regresión de landmarks anatómicos

Este módulo implementa:
- Wing Loss para precisión sub-píxel
- Symmetry-Aware Loss para coherencia bilateral
- Distance Preservation Loss para relaciones anatómicas
- Loss function completo combinado
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
from .geometric_utils import GeometricAnalyzer


class WingLoss(nn.Module):
    """
    Wing Loss function para regresión de landmarks

    Paper: "Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks"
    Optimizado para precisión sub-píxel en landmarks anatómicos
    """

    def __init__(self, omega: float = 10.0, epsilon: float = 2.0):
        """
        Args:
            omega: Threshold parameter que separa régimen linear del logarítmico
            epsilon: Parameter que controla la curvatura en la región no-linear
        """
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

        # Pre-calcular constante para optimización
        self.C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calcular Wing Loss

        Args:
            prediction: Coordenadas predichas (batch_size, num_coords)
            target: Coordenadas verdaderas (batch_size, num_coords)

        Returns:
            Wing loss escalar
        """
        diff = torch.abs(prediction - target)

        # Condición para cambiar entre régimen logarítmico y linear
        condition = diff < self.omega

        # Wing loss formula
        wing_loss = torch.where(
            condition,
            self.omega * torch.log(1 + diff / self.epsilon),
            diff - self.C
        )

        return wing_loss.mean()


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss con pesos por landmark

    Versión mejorada que asigna pesos diferentes según importancia anatómica
    """

    def __init__(
        self,
        omega: float = 14.0,
        theta: float = 0.5,
        epsilon: float = 1.0,
        alpha: float = 2.1,
        landmark_weights: Optional[List[float]] = None
    ):
        """
        Args:
            omega: Threshold parameter
            theta: Threshold para cambio de régimen
            epsilon: Curvature parameter
            alpha: Exponente para el término adaptativo
            landmark_weights: Pesos por landmark [15] (si None, usa pesos uniformes)
        """
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

        # Pesos basados en importancia anatómica identificada
        if landmark_weights is None:
            # Pesos por defecto basados en análisis anatómico
            self.landmark_weights = torch.tensor([
                1.5,  # 0: Mediastino superior (crítico)
                1.5,  # 1: Mediastino inferior (crítico)
                1.2,  # 2: Ápice izquierdo
                1.2,  # 3: Ápice derecho
                1.3,  # 4: Hilio izquierdo (importante)
                1.3,  # 5: Hilio derecho (importante)
                1.1,  # 6: Base izquierda
                1.1,  # 7: Base derecha
                1.5,  # 8: Centro medio (crítico)
                1.4,  # 9: Centro inferior
                1.4,  # 10: Centro superior
                1.0,  # 11: Borde izquierdo
                1.0,  # 12: Borde derecho
                2.0,  # 13: Problema conocido (peso alto)
                2.0   # 14: Problema conocido (peso alto)
            ])
        else:
            self.landmark_weights = torch.tensor(landmark_weights)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calcular Adaptive Wing Loss

        Args:
            prediction: Coordenadas predichas (batch_size, 30)
            target: Coordenadas verdaderas (batch_size, 30)

        Returns:
            Adaptive wing loss escalar
        """
        # Reshape para procesar por landmark
        pred_reshaped = prediction.view(-1, 15, 2)  # (batch_size, 15, 2)
        target_reshaped = target.view(-1, 15, 2)

        # Mover pesos al dispositivo correcto
        weights = self.landmark_weights.to(prediction.device)

        total_loss = 0.0
        batch_size = pred_reshaped.size(0)

        for i in range(15):  # Para cada landmark
            pred_landmark = pred_reshaped[:, i, :]  # (batch_size, 2)
            target_landmark = target_reshaped[:, i, :]

            # Calcular distancia euclidiana
            diff = torch.norm(pred_landmark - target_landmark, dim=1)  # (batch_size,)

            # Adaptive Wing Loss formula
            condition1 = diff < self.theta
            condition2 = diff < self.omega

            # Término adaptativo
            A = self.omega * (1 / (1 + torch.pow(self.omega / self.epsilon, self.alpha - target_landmark)))
            C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.omega / self.epsilon, self.alpha - target_landmark))

            # Calcular loss por componente
            loss_small = A * torch.log(1 + torch.pow(diff / self.epsilon, self.alpha - target_landmark))
            loss_medium = A * torch.log(1 + torch.pow(self.omega / self.epsilon, self.alpha - target_landmark))
            loss_large = diff - C

            # Aplicar condiciones
            loss = torch.where(
                condition1,
                loss_small,
                torch.where(condition2, loss_medium, loss_large)
            )

            # Aplicar peso del landmark
            weighted_loss = loss * weights[i]
            total_loss += weighted_loss.mean()

        return total_loss / 15  # Promedio sobre landmarks


class SymmetryLoss(nn.Module):
    """
    Enhanced Symmetry Loss for enforcing bilateral anatomical constraints

    Implements sophisticated symmetry enforcement using mediastinal axis as reference
    and applying mirror transformation penalties for bilateral landmarks.
    """

    def __init__(self, symmetry_weight: float = 0.3, use_mediastinal_axis: bool = True):
        """
        Args:
            symmetry_weight: Weight for symmetry penalty term
            use_mediastinal_axis: Use mediastinal landmarks as symmetry axis
        """
        super(SymmetryLoss, self).__init__()
        self.symmetry_weight = symmetry_weight
        self.use_mediastinal_axis = use_mediastinal_axis

        # Bilateral symmetric pairs (anatomical knowledge)
        self.symmetric_pairs = [
            (2, 3),   # Ápices pulmonares (left-right apex)
            (4, 5),   # Hilios (left-right hilum)
            (6, 7),   # Bases pulmonares (left-right base)
            (11, 12), # Bordes superiores (left-right superior borders)
            (13, 14)  # Senos costofrénicos (left-right costophrenic angles)
        ]

        # Mediastinal landmarks for symmetry axis calculation
        self.mediastinal_landmarks = [0, 1, 8, 9, 10]  # Central anatomical structures

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Compute symmetry loss for bilateral anatomical constraints

        Args:
            landmarks: Predicted landmarks (batch_size, 30) in [x1,y1,...,x15,y15] format

        Returns:
            Symmetry loss scalar value
        """
        return self._compute_enhanced_symmetry_loss(landmarks)

    def _compute_enhanced_symmetry_loss(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Enhanced symmetry loss with mediastinal axis and mirror transformation"""
        # Reshape to landmark format: (batch_size, 15, 2)
        landmarks_2d = landmarks.view(-1, 15, 2)
        batch_size = landmarks_2d.size(0)

        total_symmetry_penalty = 0.0
        num_pairs = len(self.symmetric_pairs)

        # Calculate mediastinal axis for each sample
        if self.use_mediastinal_axis:
            mediastinal_center_x = self._calculate_mediastinal_axis(landmarks_2d)
        else:
            # Use simple center of mass as fallback
            mediastinal_center_x = torch.mean(landmarks_2d[:, :, 0], dim=1)  # (batch_size,)

        for left_idx, right_idx in self.symmetric_pairs:
            left_point = landmarks_2d[:, left_idx]    # (batch_size, 2)
            right_point = landmarks_2d[:, right_idx]  # (batch_size, 2)

            # Calculate expected symmetric position
            expected_right = self._mirror_point_across_axis(left_point, mediastinal_center_x)

            # Symmetry penalty: distance between actual and expected symmetric position
            symmetry_penalty = torch.norm(right_point - expected_right, dim=1)  # (batch_size,)

            # Add bidirectional penalty (mirror left from right as well)
            expected_left = self._mirror_point_across_axis(right_point, mediastinal_center_x)
            bidirectional_penalty = torch.norm(left_point - expected_left, dim=1)
            symmetry_penalty = symmetry_penalty + bidirectional_penalty

            total_symmetry_penalty += symmetry_penalty.mean()

        return total_symmetry_penalty / (num_pairs * 2)  # Normalize by pairs and bidirectional

    def _calculate_mediastinal_axis(self, landmarks_2d: torch.Tensor) -> torch.Tensor:
        """
        Calculate mediastinal axis using anatomically stable central landmarks

        Args:
            landmarks_2d: Landmarks in (batch_size, 15, 2) format

        Returns:
            X-coordinate of mediastinal axis for each sample (batch_size,)
        """
        # Use mediastinal landmarks: superior mediastinum, inferior mediastinum, central points
        mediastinal_points = landmarks_2d[:, self.mediastinal_landmarks, :]  # (batch_size, 5, 2)

        # Calculate weighted average of x-coordinates (y-coordinate importance weighting)
        weights = torch.tensor([1.2, 1.2, 1.5, 1.3, 1.3], device=landmarks_2d.device)  # More weight on central
        weights = weights / weights.sum()  # Normalize (creates new tensor)

        weighted_x = torch.sum(mediastinal_points[:, :, 0] * weights.unsqueeze(0), dim=1)
        return weighted_x

    def _mirror_point_across_axis(self, point: torch.Tensor, axis_x: torch.Tensor) -> torch.Tensor:
        """
        Mirror a point across vertical axis defined by axis_x

        Args:
            point: Point coordinates (batch_size, 2)
            axis_x: X-coordinate of symmetry axis (batch_size,)

        Returns:
            Mirrored point coordinates (batch_size, 2)
        """
        mirrored_point = point.clone()

        # Mirror x-coordinate: x_mirrored = 2 * axis_x - x_original
        mirrored_point[:, 0] = 2 * axis_x - point[:, 0]

        # Y-coordinate remains the same (vertical mirror)
        # mirrored_point[:, 1] = point[:, 1]  # Already copied from clone()

        return mirrored_point


class SymmetryAwareLoss(nn.Module):
    """
    Loss function que penaliza violaciones de simetría bilateral

    DEPRECATED: Use SymmetryLoss for new implementations
    """

    def __init__(self, base_loss: nn.Module, symmetry_weight: float = 0.3):
        """
        Args:
            base_loss: Loss function base (ej. WingLoss)
            symmetry_weight: Peso del término de simetría
        """
        super(SymmetryAwareLoss, self).__init__()
        self.base_loss = base_loss
        self.symmetry_weight = symmetry_weight

        # Pares simétricos anatómicos (índices base-0)
        self.symmetric_pairs = [
            (2, 3),   # Ápices pulmonares
            (4, 5),   # Hilios
            (6, 7),   # Bases pulmonares
            (11, 12), # Bordes superiores
            (13, 14)  # Senos costofrénicos
        ]

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calcular loss con penalización de simetría

        Args:
            prediction: Coordenadas predichas (batch_size, 30)
            target: Coordenadas verdaderas (batch_size, 30)

        Returns:
            Loss combinado
        """
        # Loss base
        base_loss = self.base_loss(prediction, target)

        # Loss de simetría
        symmetry_loss = self._compute_symmetry_loss(prediction)

        # Combinar
        total_loss = base_loss + self.symmetry_weight * symmetry_loss

        return total_loss

    def _compute_symmetry_loss(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Compute pérdida por violaciones de simetría"""
        # Reshape a formato landmark
        landmarks_reshaped = landmarks.view(-1, 15, 2)  # (batch_size, 15, 2)

        symmetry_loss = 0.0
        num_pairs = len(self.symmetric_pairs)

        for left_idx, right_idx in self.symmetric_pairs:
            # Coordenadas x de landmarks simétricos
            left_x = landmarks_reshaped[:, left_idx, 0]  # (batch_size,)
            right_x = landmarks_reshaped[:, right_idx, 0]

            # Calcular centro de masa en x para cada muestra
            center_x = torch.mean(landmarks_reshaped[:, :, 0], dim=1)  # (batch_size,)

            # Distancias al centro
            left_distance = torch.abs(left_x - center_x)
            right_distance = torch.abs(right_x - center_x)

            # Penalizar diferencia en distancias (violación de simetría)
            symmetry_error = torch.abs(left_distance - right_distance)
            symmetry_loss += symmetry_error.mean()

        return symmetry_loss / num_pairs


class DistancePreservationLoss(nn.Module):
    """
    Loss function que preserva distancias anatómicas críticas
    """

    def __init__(self, distance_weight: float = 0.2):
        """
        Args:
            distance_weight: Peso del término de preservación de distancias
        """
        super(DistancePreservationLoss, self).__init__()
        self.distance_weight = distance_weight

        # Distancias anatómicas críticas que deben preservarse
        self.critical_distances = [
            (0, 1),   # Mediastino superior-inferior
            (8, 9),   # Eje central medio
            (2, 3),   # Ancho torácico superior (ápices)
            (4, 5),   # Ancho torácico medio (hilios)
            (6, 7),   # Ancho torácico inferior (bases)
        ]

    def forward(
        self,
        pred_landmarks: torch.Tensor,
        target_landmarks: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcular pérdida de preservación de distancias

        Args:
            pred_landmarks: Landmarks predichos (batch_size, 30)
            target_landmarks: Landmarks verdaderos (batch_size, 30)

        Returns:
            Loss de preservación de distancias
        """
        # Reshape a formato landmark
        pred_reshaped = pred_landmarks.view(-1, 15, 2)
        target_reshaped = target_landmarks.view(-1, 15, 2)

        distance_loss = 0.0
        num_distances = len(self.critical_distances)

        for idx1, idx2 in self.critical_distances:
            # Calcular distancias predichas
            pred_dist = torch.norm(
                pred_reshaped[:, idx1, :] - pred_reshaped[:, idx2, :],
                dim=1
            )

            # Calcular distancias verdaderas
            target_dist = torch.norm(
                target_reshaped[:, idx1, :] - target_reshaped[:, idx2, :],
                dim=1
            )

            # Penalizar diferencia en distancias
            distance_error = torch.abs(pred_dist - target_dist)
            distance_loss += distance_error.mean()

        return self.distance_weight * distance_loss / num_distances


class CompleteLandmarkLoss(nn.Module):
    """
    Loss function completo que combina todos los componentes geométricos

    Integra Wing Loss, simetría y preservación de distancias para landmarks anatómicos
    """

    def __init__(
        self,
        wing_omega: float = 10.0,
        wing_epsilon: float = 2.0,
        symmetry_weight: float = 0.3,
        distance_weight: float = 0.2,
        use_adaptive_wing: bool = False,
        landmark_weights: Optional[List[float]] = None
    ):
        """
        Args:
            wing_omega: Parámetro omega para Wing Loss
            wing_epsilon: Parámetro epsilon para Wing Loss
            symmetry_weight: Peso del término de simetría
            distance_weight: Peso del término de distancias
            use_adaptive_wing: Si usar Adaptive Wing Loss en lugar de Wing Loss
            landmark_weights: Pesos por landmark (solo si use_adaptive_wing=True)
        """
        super(CompleteLandmarkLoss, self).__init__()

        # Loss base
        if use_adaptive_wing:
            self.base_loss = AdaptiveWingLoss(
                omega=wing_omega,
                epsilon=wing_epsilon,
                landmark_weights=landmark_weights
            )
        else:
            self.base_loss = WingLoss(omega=wing_omega, epsilon=wing_epsilon)

        # Componentes geométricos
        self.symmetry_loss = SymmetryAwareLoss(
            base_loss=nn.MSELoss(),  # Para cálculo interno
            symmetry_weight=1.0      # Peso se aplica externamente
        )

        self.distance_loss = DistancePreservationLoss(distance_weight=1.0)

        # Pesos de combinación
        self.symmetry_weight = symmetry_weight
        self.distance_weight = distance_weight

        # Analyzer geométrico para métricas adicionales
        self.geometric_analyzer = GeometricAnalyzer()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Calcular loss combinado

        Args:
            prediction: Coordenadas predichas (batch_size, 30)
            target: Coordenadas verdaderas (batch_size, 30)
            return_components: Si retornar componentes individuales

        Returns:
            Loss total (o tupla con componentes si return_components=True)
        """
        # 1. Wing Loss principal
        wing_loss = self.base_loss(prediction, target)

        # 2. Symmetry Loss
        symmetry_component = self.symmetry_loss._compute_symmetry_loss(prediction)

        # 3. Distance Preservation Loss
        distance_component = self.distance_loss(prediction, target)

        # 4. Combinar losses
        total_loss = (
            wing_loss +
            self.symmetry_weight * symmetry_component +
            self.distance_weight * distance_component
        )

        if return_components:
            return total_loss, {
                'wing_loss': wing_loss.item(),
                'symmetry_loss': symmetry_component.item(),
                'distance_loss': distance_component.item(),
                'total_loss': total_loss.item()
            }
        else:
            return total_loss

    def compute_metrics(self, prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Compute métricas geométricas detalladas para logging

        Args:
            prediction: Coordenadas predichas (batch_size, 30)
            target: Coordenadas verdaderas (batch_size, 30)

        Returns:
            Diccionario con métricas geométricas
        """
        with torch.no_grad():
            # Análisis geométrico
            pred_analysis = self.geometric_analyzer.analyze_batch(prediction)
            target_analysis = self.geometric_analyzer.analyze_batch(target)

            # Métricas de calidad
            consistency_scores = pred_analysis['bilateral_consistency']
            validity_scores = pred_analysis['anatomical_validity']

            metrics = {
                'geometric_consistency': consistency_scores.mean().item(),
                'anatomical_validity': validity_scores.mean().item(),
                'symmetry_error': pred_analysis['symmetry_error_mean'].mean().item(),
                'constraint_violations': pred_analysis['total_violations'].mean().item(),
                'cardiothoracic_ratio': pred_analysis['cardiothoracic_ratio'].mean().item(),
                'thoracic_aspect_ratio': pred_analysis['aspect_ratio'].mean().item()
            }

            return metrics


def create_loss_function(loss_config: Dict) -> nn.Module:
    """
    Factory function para crear función de pérdida según configuración

    Args:
        loss_config: Diccionario con configuración de loss

    Returns:
        Instancia de función de pérdida configurada
    """
    loss_type = loss_config.get('type', 'mse').lower()

    if loss_type == 'wing':
        return WingLoss(
            omega=loss_config.get('wing_omega', 10.0),
            epsilon=loss_config.get('wing_epsilon', 2.0)
        )

    elif loss_type == 'adaptive_wing':
        return AdaptiveWingLoss(
            omega=loss_config.get('wing_omega', 14.0),
            epsilon=loss_config.get('wing_epsilon', 1.0),
            landmark_weights=loss_config.get('landmark_weights', None)
        )

    elif loss_type == 'complete_landmark':
        return CompleteLandmarkLoss(
            wing_omega=loss_config.get('wing_omega', 10.0),
            wing_epsilon=loss_config.get('wing_epsilon', 2.0),
            symmetry_weight=loss_config.get('symmetry_weight', 0.3),
            distance_weight=loss_config.get('distance_weight', 0.2),
            use_adaptive_wing=loss_config.get('use_adaptive_wing', False),
            landmark_weights=loss_config.get('landmark_weights', None)
        )

    elif loss_type == 'mse':
        return nn.MSELoss()

    elif loss_type == 'l1':
        return nn.L1Loss()

    elif loss_type == 'smooth_l1':
        return nn.SmoothL1Loss()

    else:
        raise ValueError(f"Tipo de loss no soportado: {loss_type}")
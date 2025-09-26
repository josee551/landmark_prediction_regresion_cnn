"""
Modelo ResNet-18 modificado para regresión de landmarks
Utiliza transfer learning desde ImageNet
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetLandmarkRegressor(nn.Module):
    """
    ResNet-18 modificada para regresión de landmarks

    Esta clase:
    1. Carga ResNet-18 preentrenada en ImageNet
    2. Reemplaza la capa de clasificación por una capa de regresión
    3. Permite congelar/descongelar el backbone para transfer learning en fases
    4. Implementa métodos para guardar y cargar checkpoints
    """

    def __init__(
        self,
        num_landmarks: int = 15,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.5,
    ):
        """
        Args:
            num_landmarks: Número de landmarks (genera num_landmarks*2 coordenadas)
            pretrained: Si cargar pesos preentrenados de ImageNet
            freeze_backbone: Si congelar las capas convolucionales
            dropout_rate: Tasa de dropout para regularización
        """
        super(ResNetLandmarkRegressor, self).__init__()

        self.num_landmarks = num_landmarks
        self.num_coords = num_landmarks * 2  # x, y para cada landmark
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate

        # Cargar ResNet-18 preentrenada
        self.backbone = models.resnet18(pretrained=pretrained)

        # Obtener el número de features de la última capa convolucional
        # ResNet-18 tiene 512 features antes de la capa fc
        self.backbone_features = self.backbone.fc.in_features

        # Remover la capa de clasificación original
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Crear nueva cabeza de regresión
        self.regression_head = self._create_regression_head()

        # Congelar backbone si se especifica
        if freeze_backbone:
            self.freeze_backbone()

        print("Modelo creado:")
        print(f"  - Landmarks: {self.num_landmarks}")
        print(f"  - Coordenadas de salida: {self.num_coords}")
        print(f"  - Preentrenado: {self.pretrained}")
        print(f"  - Backbone congelado: {freeze_backbone}")
        print(f"  - Features del backbone: {self.backbone_features}")

    def _create_regression_head(self) -> nn.Module:
        """
        Crear cabeza de regresión personalizada

        Para regresión de landmarks, usamos:
        1. Global Average Pooling (ya incluido en ResNet)
        2. Dropout para regularización
        3. Capa linear principal
        4. Activación Sigmoid para normalizar salida entre [0,1]

        Returns:
            Módulo de cabeza de regresión
        """
        return nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate / 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate / 4),
            nn.Linear(256, self.num_coords),
            nn.Sigmoid(),  # Normalizar salida entre [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo

        Args:
            x: Tensor de entrada de forma (batch_size, 3, 224, 224)

        Returns:
            Tensor de coordenadas predichas de forma (batch_size, num_coords)
            Coordenadas normalizadas entre [0,1]
        """
        # Extraer features del backbone
        features = self.backbone(x)  # (batch_size, 512, 1, 1)

        # Flatten las features
        features = torch.flatten(features, 1)  # (batch_size, 512)

        # Aplicar cabeza de regresión
        landmarks = self.regression_head(features)  # (batch_size, num_coords)

        return landmarks

    def freeze_backbone(self):
        """
        Congelar parámetros del backbone para transfer learning Fase 1

        En esta fase solo entrenamos la cabeza de regresión
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        print("✓ Backbone congelado - Solo se entrenará la cabeza de regresión")

    def unfreeze_backbone(self):
        """
        Descongelar backbone para fine-tuning Fase 2

        En esta fase entrenamos toda la red con learning rate diferenciado
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

        print("✓ Backbone descongelado - Se entrenará toda la red")

    def get_backbone_parameters(self):
        """
        Obtener parámetros del backbone para optimización diferenciada

        Returns:
            Generador de parámetros del backbone
        """
        return self.backbone.parameters()

    def get_head_parameters(self):
        """
        Obtener parámetros de la cabeza para optimización diferenciada

        Returns:
            Generador de parámetros de la cabeza
        """
        return self.regression_head.parameters()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtener información del modelo para logging

        Returns:
            Diccionario con información del modelo
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.regression_head.parameters())

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "backbone_parameters": backbone_params,
            "head_parameters": head_params,
            "num_landmarks": self.num_landmarks,
            "num_coords": self.num_coords,
            "pretrained": self.pretrained,
        }

    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        optimizer_state: Optional[Dict] = None,
        loss: Optional[float] = None,
        metrics: Optional[Dict] = None,
    ):
        """
        Guardar checkpoint del modelo

        Args:
            filepath: Ruta donde guardar el checkpoint
            epoch: Época actual
            optimizer_state: Estado del optimizador
            loss: Pérdida actual
            metrics: Métricas adicionales
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "model_config": {
                "num_landmarks": self.num_landmarks,
                "pretrained": self.pretrained,
                "dropout_rate": self.dropout_rate,
            },
            "model_info": self.get_model_info(),
        }

        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state

        if loss is not None:
            checkpoint["loss"] = loss

        if metrics is not None:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, filepath)
        print(f"✓ Checkpoint guardado en: {filepath}")

    @classmethod
    def load_from_checkpoint(
        cls, filepath: str, map_location: Optional[str] = None
    ) -> "ResNetLandmarkRegressor":
        """
        Cargar modelo desde checkpoint

        Args:
            filepath: Ruta del checkpoint
            map_location: Dispositivo donde cargar el modelo

        Returns:
            Modelo cargado
        """
        checkpoint = torch.load(filepath, map_location=map_location)

        # Crear modelo con la configuración guardada
        model_config = checkpoint["model_config"]
        model = cls(
            num_landmarks=model_config["num_landmarks"],
            pretrained=False,  # No cargar pesos preentrenados
            dropout_rate=model_config.get("dropout_rate", 0.5),
        )

        # Cargar estado del modelo
        model.load_state_dict(checkpoint["model_state_dict"])

        print(f"✓ Modelo cargado desde: {filepath}")
        print(f"✓ Época: {checkpoint['epoch']}")

        return model, checkpoint


def create_model(
    num_landmarks: int = 15,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout_rate: float = 0.5,
) -> ResNetLandmarkRegressor:
    """
    Factory function para crear modelo ResNet de regresión

    Args:
        num_landmarks: Número de landmarks
        pretrained: Si usar pesos preentrenados
        freeze_backbone: Si congelar backbone
        dropout_rate: Tasa de dropout

    Returns:
        Modelo ResNet configurado
    """
    return ResNetLandmarkRegressor(
        num_landmarks=num_landmarks,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout_rate,
    )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Contar parámetros del modelo

    Args:
        model: Modelo PyTorch

    Returns:
        Diccionario con conteo de parámetros
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def print_model_summary(model):
    """
    Imprimir resumen detallado del modelo

    Args:
        model: Modelo ResNet
    """
    print("\n" + "=" * 60)
    print("RESUMEN DEL MODELO RESNET-18 LANDMARK REGRESSOR")
    print("=" * 60)

    model_info = model.get_model_info()
    param_counts = count_parameters(model)

    if hasattr(model, 'use_coordinate_attention') and model.use_coordinate_attention:
        print("Arquitectura: ResNet-18 + Coordinate Attention + Cabeza de Regresión")
    else:
        print("Arquitectura: ResNet-18 + Cabeza de Regresión")
    print(f"Preentrenado en ImageNet: {model_info['pretrained']}")
    print(f"Landmarks: {model_info['num_landmarks']}")
    print(f"Coordenadas de salida: {model_info['num_coords']}")

    print("\nParámetros:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  Entrenables: {param_counts['trainable']:,}")
    print(f"  Congelados: {param_counts['frozen']:,}")

    print("\nDistribución de parámetros:")
    print(f"  Backbone: {model_info['backbone_parameters']:,}")
    print(f"  Cabeza: {model_info['head_parameters']:,}")

    # Mostrar información de atención si está disponible
    if model_info.get('use_coordinate_attention', False):
        print(f"  Atención: {model_info.get('attention_parameters', 0):,}")
        print(f"  Reducción de atención: {model_info.get('attention_reduction', 'N/A')}")

    # Mostrar arquitectura de la cabeza
    print("\nArquitectura de cabeza de regresión:")
    for i, layer in enumerate(model.regression_head):
        print(f"  {i + 1}. {layer}")

    print("=" * 60)


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Module

    Implementa mecanismo de atención que considera tanto información espacial
    como de canal, especialmente útil para tareas de localización como landmarks.

    Paper: "Coordinate Attention for Efficient Mobile Network Design"
    """

    def __init__(self, inp: int, oup: int, reduction: int = 32):
        """
        Args:
            inp: Número de canales de entrada
            oup: Número de canales de salida (normalmente igual a inp)
            reduction: Factor de reducción para dimensionalidad interna
        """
        super(CoordinateAttention, self).__init__()

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()

        # Pooling en direcciones H y W
        x_h = self.pool_h(x)  # (n, c, h, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (n, c, w, 1)

        # Concatenar y procesar
        y = torch.cat([x_h, x_w], dim=2)  # (n, c, h+w, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Separar de nuevo
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (n, c, 1, w)

        # Generar mapas de atención
        a_h = self.conv_h(x_h).sigmoid()  # (n, c, h, 1)
        a_w = self.conv_w(x_w).sigmoid()  # (n, c, 1, w)

        # Aplicar atención
        out = identity * a_h * a_w

        return out


class ResNetWithCoordinateAttention(ResNetLandmarkRegressor):
    """
    ResNet-18 con Coordinate Attention integrado para regresión de landmarks

    Extiende ResNetLandmarkRegressor agregando mecanismo de atención coordinada
    entre el backbone y el global average pooling. Mantiene compatibilidad completa
    con la clase base y checkpoints existentes.

    Arquitectura:
    Input → ResNet Backbone → Coordinate Attention → Global Avg Pool → Regression Head → Output
    """

    def __init__(
        self,
        num_landmarks: int = 15,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.5,
        use_coordinate_attention: bool = True,
        attention_reduction: int = 32,
    ):
        """
        Args:
            num_landmarks: Número de landmarks (genera num_landmarks*2 coordenadas)
            pretrained: Si cargar pesos preentrenados de ImageNet
            freeze_backbone: Si congelar las capas convolucionales
            dropout_rate: Tasa de dropout para regularización
            use_coordinate_attention: Si usar módulo de atención
            attention_reduction: Factor de reducción para atención
        """
        super(ResNetLandmarkRegressor, self).__init__()

        self.num_landmarks = num_landmarks
        self.num_coords = num_landmarks * 2  # x, y para cada landmark
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.use_coordinate_attention = use_coordinate_attention
        self.attention_reduction = attention_reduction

        # Cargar ResNet-18 preentrenada
        resnet = models.resnet18(pretrained=pretrained)

        # Obtener el número de features de la última capa convolucional
        self.backbone_features = resnet.fc.in_features

        # Para usar coordinate attention, necesitamos separar las capas
        if self.use_coordinate_attention:
            # Crear backbone sin avgpool ni fc
            self.backbone_conv = nn.Sequential(*list(resnet.children())[:-2])  # Sin avgpool y fc

            # Agregar módulo de atención
            self.coordinate_attention = CoordinateAttention(
                inp=self.backbone_features,
                oup=self.backbone_features,
                reduction=attention_reduction
            )
        else:
            # Usar backbone original sin fc (incluye avgpool)
            self.backbone_conv = nn.Sequential(*list(resnet.children())[:-1])  # Sin fc, con avgpool
            self.coordinate_attention = None

        # Crear nueva cabeza de regresión
        self.regression_head = self._create_regression_head()

        # Configurar congelado
        if freeze_backbone:
            self.freeze_backbone()

        print("Modelo creado:")
        print(f"  - Landmarks: {self.num_landmarks}")
        print(f"  - Coordenadas de salida: {self.num_coords}")
        print(f"  - Preentrenado: {self.pretrained}")
        print(f"  - Backbone congelado: {freeze_backbone}")
        print(f"  - Features del backbone: {self.backbone_features}")
        print(f"  - Coordinate Attention: {self.use_coordinate_attention}")
        if self.use_coordinate_attention:
            print(f"  - Attention Reduction: {self.attention_reduction}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo con atención coordinada

        Args:
            x: Tensor de entrada de forma (batch_size, 3, 224, 224)

        Returns:
            Tensor de coordenadas predichas de forma (batch_size, num_coords)
            Coordenadas normalizadas entre [0,1]
        """
        # Extraer features del backbone
        features = self.backbone_conv(x)

        if self.use_coordinate_attention and self.coordinate_attention is not None:
            # Las features aquí tienen forma (batch_size, 512, 7, 7)
            # Aplicar atención coordinada
            features = self.coordinate_attention(features)

            # Aplicar global average pooling después de la atención
            features = F.adaptive_avg_pool2d(features, (1, 1))  # (batch_size, 512, 1, 1)
        else:
            # Si no hay atención, features ya están pooled (batch_size, 512, 1, 1)
            pass

        # Flatten las features
        features = torch.flatten(features, 1)  # (batch_size, 512)

        # Aplicar cabeza de regresión
        landmarks = self.regression_head(features)  # (batch_size, num_coords)

        return landmarks

    def freeze_backbone(self):
        """
        Congelar parámetros del backbone para transfer learning Fase 1
        """
        for param in self.backbone_conv.parameters():
            param.requires_grad = False
        print("✓ Backbone congelado - Solo se entrenará la cabeza de regresión")

    def unfreeze_backbone(self):
        """
        Descongelar backbone para fine-tuning Fase 2
        """
        for param in self.backbone_conv.parameters():
            param.requires_grad = True
        print("✓ Backbone descongelado - Se entrenará toda la red")

    def get_backbone_parameters(self):
        """
        Obtener parámetros del backbone para optimización diferenciada
        """
        return self.backbone_conv.parameters()

    def get_attention_parameters(self):
        """
        Obtener parámetros del módulo de atención para optimización diferenciada

        Returns:
            Generador de parámetros del módulo de atención
        """
        if self.use_coordinate_attention and self.coordinate_attention is not None:
            return self.coordinate_attention.parameters()
        else:
            return iter([])  # Generador vacío si no hay atención

    def get_head_parameters(self):
        """
        Obtener parámetros de la cabeza para optimización diferenciada
        """
        return self.regression_head.parameters()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtener información del modelo incluyendo atención

        Returns:
            Diccionario con información del modelo
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        backbone_params = sum(p.numel() for p in self.backbone_conv.parameters())
        head_params = sum(p.numel() for p in self.regression_head.parameters())

        # Agregar información de atención
        if self.use_coordinate_attention and self.coordinate_attention is not None:
            attention_params = sum(p.numel() for p in self.coordinate_attention.parameters())
        else:
            attention_params = 0

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "backbone_parameters": backbone_params,
            "head_parameters": head_params,
            "attention_parameters": attention_params,
            "num_landmarks": self.num_landmarks,
            "num_coords": self.num_coords,
            "pretrained": self.pretrained,
            "use_coordinate_attention": self.use_coordinate_attention,
            "attention_reduction": self.attention_reduction if self.use_coordinate_attention else None,
        }

    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        optimizer_state: Optional[Dict] = None,
        loss: Optional[float] = None,
        metrics: Optional[Dict] = None,
    ):
        """
        Guardar checkpoint del modelo con configuración de atención
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "model_config": {
                "num_landmarks": self.num_landmarks,
                "pretrained": self.pretrained,
                "dropout_rate": self.dropout_rate,
                "use_coordinate_attention": self.use_coordinate_attention,
                "attention_reduction": self.attention_reduction,
            },
            "model_info": self.get_model_info(),
        }

        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state

        if loss is not None:
            checkpoint["loss"] = loss

        if metrics is not None:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, filepath)
        print(f"✓ Checkpoint guardado en: {filepath}")

    @classmethod
    def load_from_checkpoint(
        cls, filepath: str, map_location: Optional[str] = None
    ) -> "ResNetWithCoordinateAttention":
        """
        Cargar modelo desde checkpoint con compatibilidad hacia atrás

        Maneja checkpoints tanto de ResNetLandmarkRegressor como de
        ResNetWithCoordinateAttention, cargando atención solo si está disponible.
        """
        checkpoint = torch.load(filepath, map_location=map_location)
        model_config = checkpoint["model_config"]

        # Compatibilidad hacia atrás: si no hay config de atención, deshabilitar
        use_attention = model_config.get("use_coordinate_attention", False)
        attention_reduction = model_config.get("attention_reduction", 32)

        # Crear modelo con configuración
        model = cls(
            num_landmarks=model_config["num_landmarks"],
            pretrained=False,  # No cargar pesos preentrenados
            dropout_rate=model_config.get("dropout_rate", 0.5),
            use_coordinate_attention=use_attention,
            attention_reduction=attention_reduction,
        )

        # Cargar estado del modelo con manejo de compatibilidad
        model_state = checkpoint["model_state_dict"]

        # Detectar si el checkpoint es del modelo base o con atención
        checkpoint_has_attention = any("coordinate_attention" in k for k in model_state.keys())
        checkpoint_has_backbone_conv = any("backbone_conv" in k for k in model_state.keys())

        if not checkpoint_has_backbone_conv:
            # Checkpoint del modelo base - necesitamos mapear 'backbone' -> 'backbone_conv'
            print("⚠️  Checkpoint del modelo base detectado - mapeando claves...")
            mapped_state = {}
            for k, v in model_state.items():
                if k.startswith("backbone."):
                    # Mapear backbone.X -> backbone_conv.X
                    new_key = k.replace("backbone.", "backbone_conv.")
                    mapped_state[new_key] = v
                else:
                    mapped_state[k] = v
            model_state = mapped_state

        # Manejar módulo de atención
        if use_attention and not checkpoint_has_attention:
            print(f"⚠️  Inicializando módulo de atención con pesos aleatorios")
            # Cargar solo las claves disponibles (backbone + head)
            model.load_state_dict(model_state, strict=False)
        elif not use_attention and checkpoint_has_attention:
            # Filtrar claves de atención del checkpoint
            filtered_state = {k: v for k, v in model_state.items()
                             if "coordinate_attention" not in k}
            model.load_state_dict(filtered_state, strict=False)
        else:
            # Cargar normalmente (ambos coinciden)
            model.load_state_dict(model_state, strict=False)

        print(f"✓ Modelo cargado desde: {filepath}")
        print(f"✓ Época: {checkpoint['epoch']}")
        print(f"✓ Coordinate Attention: {use_attention}")

        return model, checkpoint


def create_model_with_attention(
    num_landmarks: int = 15,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout_rate: float = 0.5,
    use_coordinate_attention: bool = True,
    attention_reduction: int = 32,
) -> ResNetWithCoordinateAttention:
    """
    Factory function para crear modelo ResNet con Coordinate Attention

    Args:
        num_landmarks: Número de landmarks
        pretrained: Si usar pesos preentrenados
        freeze_backbone: Si congelar backbone
        dropout_rate: Tasa de dropout
        use_coordinate_attention: Si usar atención coordinada
        attention_reduction: Factor de reducción para atención

    Returns:
        Modelo ResNet con atención configurado
    """
    return ResNetWithCoordinateAttention(
        num_landmarks=num_landmarks,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout_rate,
        use_coordinate_attention=use_coordinate_attention,
        attention_reduction=attention_reduction,
    )


def create_model_from_config(config: Dict[str, Any]) -> ResNetWithCoordinateAttention:
    """
    Crear modelo desde configuración YAML

    Args:
        config: Diccionario de configuración

    Returns:
        Modelo configurado
    """
    model_config = config.get("model", {})

    return create_model_with_attention(
        num_landmarks=model_config.get("num_landmarks", 15),
        pretrained=model_config.get("pretrained", True),
        freeze_backbone=model_config.get("freeze_backbone", True),
        dropout_rate=model_config.get("dropout_rate", 0.5),
        use_coordinate_attention=model_config.get("use_coordinate_attention", False),
        attention_reduction=model_config.get("attention_reduction", 32),
    )


if __name__ == "__main__":
    # Test del modelo
    print("Probando modelo ResNet-18 Landmark Regressor...")

    # Crear modelo base
    print("\n=== MODELO BASE ===")
    model = create_model(num_landmarks=15, freeze_backbone=True)
    print_model_summary(model)

    # Test forward pass modelo base
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    print("\nTest forward pass modelo base:")
    print(f"Input shape: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Verificar que la salida esté normalizada
    assert output.min() >= 0 and output.max() <= 1, "La salida debe estar entre [0,1]"
    print("✓ Salida correctamente normalizada entre [0,1]")

    # Crear modelo con atención
    print("\n=== MODELO CON COORDINATE ATTENTION ===")
    model_attention = create_model_with_attention(
        num_landmarks=15,
        freeze_backbone=True,
        use_coordinate_attention=True,
        attention_reduction=32
    )
    print_model_summary(model_attention)

    # Test forward pass modelo con atención
    print("\nTest forward pass modelo con atención:")
    print(f"Input shape: {dummy_input.shape}")

    with torch.no_grad():
        output_attention = model_attention(dummy_input)

    print(f"Output shape: {output_attention.shape}")
    print(f"Output range: [{output_attention.min():.3f}, {output_attention.max():.3f}]")

    # Verificar que la salida esté normalizada
    assert output_attention.min() >= 0 and output_attention.max() <= 1, "La salida debe estar entre [0,1]"
    print("✓ Salida correctamente normalizada entre [0,1]")

    # Comparar diferencias en outputs
    diff = torch.abs(output - output_attention).mean()
    print(f"\nDiferencia promedio entre modelos: {diff:.6f}")

    # Test de compatibilidad de carga
    print("\n=== TEST DE COMPATIBILIDAD ===")

    # Simular guardado y carga de modelo base
    temp_path = "/tmp/test_checkpoint.pt"
    model.save_checkpoint(temp_path, epoch=1)

    # Cargar con modelo de atención (debe funcionar sin atención)
    loaded_model, _ = ResNetWithCoordinateAttention.load_from_checkpoint(temp_path)
    print("✓ Compatibilidad hacia atrás funcionando")

    # Test parámetros diferenciados
    print("\n=== TEST DE PARÁMETROS DIFERENCIADOS ===")
    backbone_params = list(model_attention.get_backbone_parameters())
    attention_params = list(model_attention.get_attention_parameters())
    head_params = list(model_attention.get_head_parameters())

    print(f"Parámetros backbone: {len(backbone_params)}")
    print(f"Parámetros atención: {len(attention_params)}")
    print(f"Parámetros cabeza: {len(head_params)}")

    total_params_groups = len(backbone_params) + len(attention_params) + len(head_params)
    total_params_model = len(list(model_attention.parameters()))

    print(f"Total en grupos: {total_params_groups}")
    print(f"Total en modelo: {total_params_model}")

    # Cleanup
    import os
    if os.path.exists(temp_path):
        os.remove(temp_path)

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    print("\nTest forward pass:")
    print(f"Input shape: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Verificar que la salida esté normalizada
    assert output.min() >= 0 and output.max() <= 1, "La salida debe estar entre [0,1]"
    print("✓ Salida correctamente normalizada entre [0,1]")

    print("\n✓ Modelo funciona correctamente")

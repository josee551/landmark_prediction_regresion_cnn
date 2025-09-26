"""
Coordinate Attention Module for Landmark Regression

Implementation based on "Coordinate Attention for Efficient Mobile Network Design"
Optimized for ResNet-18 backbone and anatomical landmark regression tasks.

This module enhances spatial feature representation by encoding positional
information explicitly, crucial for precise coordinate prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Module for ResNet-18 Landmark Regression

    This attention mechanism decomposes channel attention into two 1D attention
    maps along spatial dimensions, preserving precise positional information
    essential for landmark regression tasks.

    Mathematical Formulation:
    -------------------------
    For input X ∈ R^(C×H×W):

    1. Coordinate Information Embedding:
       z_h^c(i) = 1/W * Σ(j=0 to W-1) x^c(i,j)  [Horizontal pooling]
       z_w^c(j) = 1/H * Σ(i=0 to H-1) x^c(i,j)  [Vertical pooling]

    2. Coordinate Attention Generation:
       f = δ(F1([z_h, z_w]))  [Joint encoding]
       g_h, g_w = split(f)    [Split to horizontal/vertical]
       g_h = σ(F_h(g_h))      [Horizontal attention]
       g_w = σ(F_w(g_w))      [Vertical attention]

    3. Feature Enhancement:
       y^c(i,j) = x^c(i,j) × g_h^c(i) × g_w^c(j)

    Where:
    - δ = ReLU activation
    - σ = Sigmoid activation
    - F1, F_h, F_w = 1D convolutions

    Architecture for ResNet-18:
    ---------------------------
    Input: (B, 512, 7, 7) from ResNet-18 backbone

    Step 1: Pooling
    - X_h: (B, 512, 7) via adaptive_avg_pool2d((7, 1))
    - X_w: (B, 512, 7) via adaptive_avg_pool2d((1, 7))

    Step 2: Concatenate & Encode
    - Concat: (B, 512, 14)
    - Conv1d: (B, 512, 14) → (B, 512//r, 14)
    - ReLU: Activation

    Step 3: Split & Generate Attention
    - Split: (B, 512//r, 7) + (B, 512//r, 7)
    - Conv1d_h: (B, 512//r, 7) → (B, 512, 7)
    - Conv1d_w: (B, 512//r, 7) → (B, 512, 7)
    - Sigmoid: Generate attention weights [0,1]

    Step 4: Apply Attention
    - Reshape & broadcast: (B, 512, 7, 7)
    - Element-wise multiplication with input

    Output: (B, 512, 7, 7) enhanced features
    """

    def __init__(self, in_channels: int = 512, reduction: int = 32):
        """
        Initialize Coordinate Attention Module

        Args:
            in_channels: Number of input channels (512 for ResNet-18)
            reduction: Channel reduction ratio for computational efficiency
                      - Smaller values = more parameters, higher precision
                      - 32 is optimal for landmark regression (balance precision/efficiency)

        Architecture:
            - Reduced channels: 512 // 32 = 16 (computational bottleneck)
            - Parameters: ~26K (negligible compared to ResNet-18's 11M)
            - Memory overhead: ~1% of total model memory
        """
        super(CoordinateAttention, self).__init__()

        self.in_channels = in_channels
        self.reduction = reduction
        self.reduced_channels = max(1, in_channels // reduction)

        # Coordinate information embedding
        # Uses 1D convolutions to capture spatial dependencies efficiently
        self.conv1 = nn.Conv1d(
            in_channels,
            self.reduced_channels,
            kernel_size=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(self.reduced_channels)

        # Coordinate attention generation
        # Separate branches for horizontal and vertical attention
        self.conv_h = nn.Conv1d(
            self.reduced_channels,
            in_channels,
            kernel_size=1,
            bias=False
        )
        self.conv_w = nn.Conv1d(
            self.reduced_channels,
            in_channels,
            kernel_size=1,
            bias=False
        )

        # Initialize weights for stable training
        self._initialize_weights()

        print(f"CoordinateAttention initialized:")
        print(f"  - Input channels: {in_channels}")
        print(f"  - Reduction ratio: {reduction}")
        print(f"  - Reduced channels: {self.reduced_channels}")
        print(f"  - Total parameters: {self._count_parameters():,}")

    def _initialize_weights(self):
        """
        Initialize weights for stable training

        Uses Xavier/Glorot initialization for convolutions
        and sets batch norm to identity transformation initially
        """
        # Xavier initialization for convolutions
        for m in [self.conv1, self.conv_h, self.conv_w]:
            nn.init.xavier_normal_(m.weight)

        # BatchNorm initialization
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

    def _count_parameters(self) -> int:
        """Count total parameters in the module"""
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Coordinate Attention

        Args:
            x: Input feature maps (batch_size, in_channels, height, width)
               Expected: (B, 512, 7, 7) from ResNet-18

        Returns:
            Enhanced feature maps with same shape as input

        Raises:
            AssertionError: If input dimensions don't match expected format
        """
        # Validate input dimensions
        assert len(x.shape) == 4, f"Expected 4D input, got {len(x.shape)}D"
        assert x.size(1) == self.in_channels, \
            f"Expected {self.in_channels} channels, got {x.size(1)}"

        batch_size, channels, height, width = x.size()

        # Step 1: Coordinate Information Embedding
        # Decompose 2D attention into two 1D attentions

        # Horizontal pooling: aggregate along width dimension
        # (B, C, H, W) → (B, C, H, 1) → (B, C, H)
        x_h = F.adaptive_avg_pool2d(x, (height, 1)).squeeze(-1)

        # Vertical pooling: aggregate along height dimension
        # (B, C, H, W) → (B, C, 1, W) → (B, C, W)
        x_w = F.adaptive_avg_pool2d(x, (1, width)).squeeze(-2)

        # Step 2: Joint Encoding
        # Concatenate horizontal and vertical features
        # (B, C, H) + (B, C, W) → (B, C, H+W)
        x_cat = torch.cat([x_h, x_w], dim=2)

        # Encode spatial information with 1D convolution
        # (B, C, H+W) → (B, C//r, H+W)
        x_cat = self.conv1(x_cat)
        x_cat = self.bn1(x_cat)
        x_cat = F.relu(x_cat, inplace=True)

        # Step 3: Split and Generate Attention Maps
        # Split back to horizontal and vertical components
        x_h_out = x_cat[:, :, :height]  # (B, C//r, H)
        x_w_out = x_cat[:, :, height:]  # (B, C//r, W)

        # Generate attention weights
        # (B, C//r, H) → (B, C, H)
        a_h = self.conv_h(x_h_out).sigmoid()
        # (B, C//r, W) → (B, C, W)
        a_w = self.conv_w(x_w_out).sigmoid()

        # Step 4: Apply Coordinate Attention
        # Reshape for broadcasting
        # (B, C, H) → (B, C, H, 1)
        a_h = a_h.unsqueeze(-1)
        # (B, C, W) → (B, C, 1, W)
        a_w = a_w.unsqueeze(-2)

        # Apply attention via element-wise multiplication
        # Broadcasting: (B, C, H, W) * (B, C, H, 1) * (B, C, 1, W)
        out = x * a_h * a_w

        return out

    def get_attention_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract attention maps for visualization and analysis

        Args:
            x: Input feature maps (batch_size, in_channels, height, width)

        Returns:
            Tuple of (horizontal_attention, vertical_attention)
            - horizontal_attention: (batch_size, in_channels, height)
            - vertical_attention: (batch_size, in_channels, width)

        Usage:
            ```python
            coord_attn = CoordinateAttention(512)
            features = backbone(images)  # (B, 512, 7, 7)
            enhanced_features = coord_attn(features)
            h_attn, v_attn = coord_attn.get_attention_maps(features)
            ```
        """
        with torch.no_grad():
            batch_size, channels, height, width = x.size()

            # Extract coordinate information
            x_h = F.adaptive_avg_pool2d(x, (height, 1)).squeeze(-1)
            x_w = F.adaptive_avg_pool2d(x, (1, width)).squeeze(-2)

            # Joint encoding
            x_cat = torch.cat([x_h, x_w], dim=2)
            x_cat = F.relu(self.bn1(self.conv1(x_cat)), inplace=True)

            # Generate attention maps
            x_h_out = x_cat[:, :, :height]
            x_w_out = x_cat[:, :, height:]

            a_h = self.conv_h(x_h_out).sigmoid()  # (B, C, H)
            a_w = self.conv_w(x_w_out).sigmoid()  # (B, C, W)

            return a_h, a_w

    def extra_repr(self) -> str:
        """String representation for debugging"""
        return (f'in_channels={self.in_channels}, '
                f'reduction={self.reduction}, '
                f'reduced_channels={self.reduced_channels}')


class CoordinateAttentionResNet(nn.Module):
    """
    ResNet-18 with integrated Coordinate Attention for Landmark Regression

    This class demonstrates how to integrate CoordinateAttention into the existing
    ResNet-18 architecture seamlessly, replacing the global average pooling
    with attention-enhanced feature extraction.

    Integration Strategy:
    --------------------
    Original ResNet-18 flow:
    conv → layer1-4 → avgpool → fc

    Modified flow with Coordinate Attention:
    conv → layer1-4 → CoordinateAttention → avgpool → regression_head

    The attention module is inserted BEFORE global pooling to preserve
    spatial information that would otherwise be lost.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_landmarks: int = 15,
        attention_reduction: int = 32,
        dropout_rate: float = 0.5
    ):
        """
        Initialize ResNet with Coordinate Attention

        Args:
            backbone: ResNet backbone (without final pooling and fc layers)
            num_landmarks: Number of landmarks to predict
            attention_reduction: Reduction ratio for attention module
            dropout_rate: Dropout rate for regression head
        """
        super(CoordinateAttentionResNet, self).__init__()

        self.backbone = backbone
        self.num_landmarks = num_landmarks
        self.num_coords = num_landmarks * 2

        # Coordinate attention module
        # Applied to 512-channel features from ResNet-18
        self.coord_attention = CoordinateAttention(
            in_channels=512,
            reduction=attention_reduction
        )

        # Global average pooling (after attention enhancement)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Regression head with enhanced features
        self.regression_head = self._create_regression_head(dropout_rate)

        print(f"CoordinateAttentionResNet created:")
        print(f"  - Landmarks: {num_landmarks}")
        print(f"  - Attention reduction: {attention_reduction}")
        print(f"  - Total parameters: {self._count_parameters():,}")

    def _create_regression_head(self, dropout_rate: float) -> nn.Module:
        """Create regression head for landmark prediction"""
        return nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 4),
            nn.Linear(256, self.num_coords),
            nn.Sigmoid(),  # Normalize to [0,1]
        )

    def _count_parameters(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with coordinate attention

        Args:
            x: Input images (batch_size, 3, 224, 224)

        Returns:
            Predicted landmark coordinates (batch_size, num_coords)
        """
        # Extract features from backbone
        features = self.backbone(x)  # (B, 512, 7, 7)

        # Apply coordinate attention
        attended_features = self.coord_attention(features)  # (B, 512, 7, 7)

        # Global pooling
        pooled_features = self.global_pool(attended_features)  # (B, 512, 1, 1)
        pooled_features = torch.flatten(pooled_features, 1)  # (B, 512)

        # Regression
        landmarks = self.regression_head(pooled_features)  # (B, 30)

        return landmarks


def integrate_coordinate_attention(
    resnet_model: nn.Module,
    attention_reduction: int = 32
) -> nn.Module:
    """
    Factory function to integrate CoordinateAttention into existing ResNet

    This function modifies an existing ResNetLandmarkRegressor to include
    coordinate attention without changing the training pipeline.

    Args:
        resnet_model: Existing ResNetLandmarkRegressor instance
        attention_reduction: Reduction ratio for attention module

    Returns:
        Modified model with coordinate attention

    Example Usage:
    --------------
    ```python
    # Load existing model
    from src.models.resnet_regressor import ResNetLandmarkRegressor
    from src.models.attention_modules import integrate_coordinate_attention

    model = ResNetLandmarkRegressor(num_landmarks=15)

    # Add coordinate attention
    enhanced_model = integrate_coordinate_attention(
        model,
        attention_reduction=32
    )

    # Use enhanced model in training
    optimizer = torch.optim.Adam(enhanced_model.parameters(), lr=0.001)
    ```

    Implementation Details:
    ----------------------
    1. Extracts backbone from original model
    2. Creates new model with coordinate attention
    3. Transfers regression head weights
    4. Maintains compatibility with existing training code
    """
    # Extract components from original model
    backbone = resnet_model.backbone
    num_landmarks = resnet_model.num_landmarks
    dropout_rate = resnet_model.dropout_rate

    # Create enhanced model
    enhanced_model = CoordinateAttentionResNet(
        backbone=backbone,
        num_landmarks=num_landmarks,
        attention_reduction=attention_reduction,
        dropout_rate=dropout_rate
    )

    # Transfer regression head weights if compatible
    try:
        enhanced_model.regression_head.load_state_dict(
            resnet_model.regression_head.state_dict()
        )
        print("✓ Transferred regression head weights")
    except Exception as e:
        print(f"⚠ Could not transfer weights: {e}")
        print("  Model will use random initialization")

    return enhanced_model


def visualize_attention_maps(
    model: CoordinateAttention,
    features: torch.Tensor,
    save_path: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Visualize coordinate attention maps for analysis

    Args:
        model: CoordinateAttention instance
        features: Input features (B, C, H, W)
        save_path: Optional path to save visualization

    Returns:
        Tuple of attention maps (horizontal, vertical)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Get attention maps
    h_attn, v_attn = model.get_attention_maps(features)

    # Average across batch and channels for visualization
    h_attn_avg = h_attn.mean(dim=(0, 1)).cpu().numpy()  # (H,)
    v_attn_avg = v_attn.mean(dim=(0, 1)).cpu().numpy()  # (W,)

    if save_path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(h_attn_avg)
        ax1.set_title('Horizontal Attention')
        ax1.set_xlabel('Spatial Position')
        ax1.set_ylabel('Attention Weight')
        ax1.grid(True)

        ax2.plot(v_attn_avg)
        ax2.set_title('Vertical Attention')
        ax2.set_xlabel('Spatial Position')
        ax2.set_ylabel('Attention Weight')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Attention visualization saved to: {save_path}")

    return h_attn, v_attn


# Performance benchmarking functions
def benchmark_coordinate_attention(
    input_size: Tuple[int, int, int, int] = (8, 512, 7, 7),
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    num_iterations: int = 100
) -> dict:
    """
    Benchmark CoordinateAttention performance

    Args:
        input_size: Input tensor size (B, C, H, W)
        device: Device for benchmarking
        num_iterations: Number of forward passes

    Returns:
        Performance metrics dictionary
    """
    import time

    # Create model and input
    model = CoordinateAttention(in_channels=input_size[1]).to(device)
    input_tensor = torch.randn(input_size).to(device)

    # Warm up
    for _ in range(10):
        _ = model(input_tensor)

    # Benchmark forward pass
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()

    for _ in range(num_iterations):
        output = model(input_tensor)

    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_iterations) * 1000

    # Memory usage
    if device == 'cuda':
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        memory_mb = 0

    # Parameter count
    params = sum(p.numel() for p in model.parameters())

    metrics = {
        'avg_forward_time_ms': avg_time_ms,
        'total_parameters': params,
        'memory_usage_mb': memory_mb,
        'throughput_fps': 1000 / avg_time_ms,
        'device': device
    }

    print(f"CoordinateAttention Benchmark Results:")
    print(f"  - Average forward time: {avg_time_ms:.2f} ms")
    print(f"  - Throughput: {metrics['throughput_fps']:.1f} FPS")
    print(f"  - Parameters: {params:,}")
    print(f"  - Memory usage: {memory_mb:.1f} MB")

    return metrics


if __name__ == "__main__":
    """
    Test and demonstration of CoordinateAttention module
    """
    print("=" * 60)
    print("COORDINATE ATTENTION MODULE TEST")
    print("=" * 60)

    # Test basic functionality
    print("\n1. Testing CoordinateAttention module...")

    # Create attention module
    coord_attn = CoordinateAttention(in_channels=512, reduction=32)

    # Test input (typical ResNet-18 features)
    batch_size = 8
    test_input = torch.randn(batch_size, 512, 7, 7)

    print(f"Input shape: {test_input.shape}")

    # Forward pass
    with torch.no_grad():
        output = coord_attn(test_input)
        print(f"Output shape: {output.shape}")

        # Verify output shape matches input
        assert output.shape == test_input.shape, "Output shape mismatch!"
        print("✓ Shape preservation verified")

        # Test attention maps
        h_attn, v_attn = coord_attn.get_attention_maps(test_input)
        print(f"Horizontal attention shape: {h_attn.shape}")
        print(f"Vertical attention shape: {v_attn.shape}")

        # Verify attention weights are in [0,1]
        assert h_attn.min() >= 0 and h_attn.max() <= 1, "Invalid attention values!"
        assert v_attn.min() >= 0 and v_attn.max() <= 1, "Invalid attention values!"
        print("✓ Attention weights in valid range [0,1]")

    # Test integration with ResNet
    print("\n2. Testing integration with ResNet...")

    # Mock ResNet backbone
    import torchvision.models as models
    resnet = models.resnet18(pretrained=False)
    backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc

    # Create integrated model
    integrated_model = CoordinateAttentionResNet(
        backbone=backbone,
        num_landmarks=15,
        attention_reduction=32
    )

    # Test with realistic input
    test_images = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        landmarks = integrated_model(test_images)
        print(f"Landmark predictions shape: {landmarks.shape}")
        print(f"Predictions range: [{landmarks.min():.3f}, {landmarks.max():.3f}]")

        # Verify output
        assert landmarks.shape == (batch_size, 30), "Incorrect landmark output shape!"
        assert landmarks.min() >= 0 and landmarks.max() <= 1, "Invalid landmark range!"
        print("✓ Landmark prediction format verified")

    # Performance benchmark
    print("\n3. Performance benchmark...")
    if torch.cuda.is_available():
        benchmark_coordinate_attention(device='cuda', num_iterations=50)
    else:
        benchmark_coordinate_attention(device='cpu', num_iterations=10)

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED - CoordinateAttention ready for integration!")
    print("=" * 60)

    # Usage example
    print("\nUsage Example:")
    print("""
    # Basic usage in existing training pipeline:
    from src.models.attention_modules import integrate_coordinate_attention
    from src.models.resnet_regressor import ResNetLandmarkRegressor

    # Load your existing model
    model = ResNetLandmarkRegressor(num_landmarks=15)

    # Add coordinate attention
    enhanced_model = integrate_coordinate_attention(model, attention_reduction=32)

    # Continue with normal training
    optimizer = torch.optim.Adam(enhanced_model.parameters(), lr=0.0002)
    """)
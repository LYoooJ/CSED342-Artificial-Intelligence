"""
CNN Mini Assignment - Student Submission (submission.py)

⚠️ Rules
- ONLY modify this file.
- You may ONLY use NumPy (no PyTorch, TensorFlow, etc.).
- Do NOT change function names, arguments, or return types.
- You MAY add helper functions if needed.
"""

import numpy as np
from typing import Dict


# ============================================================
# Problem 1: Basic CNN Building Blocks (NumPy)
# ============================================================

# ------------------------------------------------------------
# Problem 1a
# ------------------------------------------------------------
def problem1a(H: int, W: int, k_H: int, k_W: int, P: int, S: int) -> tuple:
    """
    Computes output spatial dimensions for a convolution layer.

    Args:
        H (int): Input height
        W (int): Input width
        k_H (int): Kernel height
        k_W (int): Kernel width
        P (int): Padding
        S (int): Stride

    Returns:
        (outH, outW): tuple of ints
    """
    outH = (H + 2 * P - k_H) // S + 1
    outW = (W + 2 * P - k_W) // S + 1
    return (outH, outW)

# ------------------------------------------------------------
# Problem 1b
# ------------------------------------------------------------
def conv2d(x: np.ndarray, w: np.ndarray, padding: int, stride: int) -> np.ndarray:
    """
    Problem 1b: Implement 2D convolution (no batch dimension).

    Args:
        x : np.ndarray
            Input feature map of shape (C_in, H, W)
        w : np.ndarray
            Convolution kernels of shape (C_out, C_in, kH, kW)
        padding : int
            Zero-padding (added to all sides of spatial dimensions)
        stride : int
            Stride for the convolution operation

    Returns:
        out : np.ndarray
            Output feature map of shape (C_out, outH, outW)

    Notes:
    - Bias is *NOT* added here. Bias will be added in the forward pass.
    - You must implement padding and stride manually.
    - Use for-loops (no broadcasting tricks).
    """
    _, H, W = x.shape
    C_out, C_in, kH, kW = w.shape

    outH = (H + 2 * padding - kH) // stride + 1
    outW = (W + 2 * padding - kW) // stride + 1

    x_padded = np.zeros((C_in, H + 2 * padding, W + 2 * padding))
    x_padded[:, padding : H + padding, padding : W + padding] = x
    out = np.zeros((C_out, outH, outW))
    
    for r in range(outH):
        for c in range(outW):
            x_ = x_padded[:, r * stride : r * stride + kH, c * stride : c * stride + kW]

            for channel in range(C_out):
                out[channel, r, c] = np.sum(x_ * w[channel])

    return out

# ------------------------------------------------------------
# Problem 2 — ReLU
# ------------------------------------------------------------
def relu(x: np.ndarray) -> np.ndarray:
    """
    Problem 2: Implement the ReLU activation function.

    Args:
        x : np.ndarray
            Any shape NumPy array

    Returns:
        out : np.ndarray
            Same shape as x, with negative values replaced by 0
    """
    return np.maximum(0, x)



# ------------------------------------------------------------
# Problem 3 — Max Pooling
# ------------------------------------------------------------
def maxpool2d(x: np.ndarray, pool_size: int, stride: int) -> np.ndarray:
    """
    Problem 3: 2D max pooling (no batch dimension).

    Args:
        x : np.ndarray
            Input feature map of shape (C, H, W)
        pool_size : int
            Size of pooling window (pool_size x pool_size)
        stride : int
            How far the window moves each step

    Returns:
        out : np.ndarray
            Max-pooled feature map of shape (C, outH, outW)
    """
    C, H, W = x.shape
    
    outH = (H - pool_size) // stride + 1
    outW = (W - pool_size) // stride + 1
    out = np.zeros((C, outH, outW))
    
    for r in range(outH):
        for c in range(outW):
            x_ = x[:, r * stride : r * stride + pool_size, c * stride : c * stride + pool_size]
            for channel in range(C):
                out[channel, r, c] = np.max(x_[channel])

    return out

# ------------------------------------------------------------
# Problem 4 — Flatten
# ------------------------------------------------------------
def flatten(x: np.ndarray) -> np.ndarray:
    """
    Problem 4: Flatten an input tensor.

    Args:
        x : np.ndarray
            Input array of any shape (typically (C, H, W))

    Returns:
        out : np.ndarray
            Flattened (1D) vector
    """
    return x.flatten()



# ------------------------------------------------------------
# Problem 5 — Fully Connected (FC) Layer
# ------------------------------------------------------------
def fc2d(f: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Problem 5: Fully Connected layer.

    Args:
        f : np.ndarray
            Flattened input vector of shape (N,)
        W : np.ndarray
            Weight matrix of shape (out_dim, N)
        b : np.ndarray
            Bias of shape (out_dim,)

    Returns:
        logits : np.ndarray
            Output scores of shape (out_dim,)
    """
    return W @ f + b



# ============================================================
# Problem 6: Full Tiny CNN Forward Pass
# ============================================================

def simple_cnn_forward(x: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Problem 6: Forward pass of the Tiny CNN.

    Architecture:
        x : (1, 28, 28)
            -> conv2d (C_out = 4, kernel 3×3, padding=1, stride=1)
            -> + conv_b
            -> ReLU
            -> MaxPool2d (2×2, stride 2)   -> (4, 14, 14)
            -> Flatten                    -> (4*14*14,)
            -> FC layer (10 outputs)

    Args:
        x : np.ndarray
            Input image, shape (1, 28, 28)

        params : Dict[str, np.ndarray]
            Dictionary containing pretrained parameters:
                "conv_w": (4, 1, 3, 3)
                "conv_b": (4,)
                "fc_W":   (10, 4*14*14)
                "fc_b":   (10,)

    Returns:
        logits : np.ndarray
            Class scores of shape (10,)
    """
    z = conv2d(x, params["conv_w"], 1, 1) + params["conv_b"].reshape(-1, 1, 1)
    a = relu(z)
    p = maxpool2d(a, 2, 2)
    f = flatten(p)
    logits = fc2d(f, params["fc_W"], params["fc_b"])

    return logits
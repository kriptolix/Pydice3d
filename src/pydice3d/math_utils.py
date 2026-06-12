"""
math_utils.py.
"""

from __future__ import annotations

import math
import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalizes an N-D vector. Returns a zero vector if the norm is less than epsilon."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


def quat_to_matrix(xyzw: np.ndarray) -> np.ndarray:
    """
    Quaternion [x, y, z, w] → 3×3 rotation matrix (float64).

    Input format: PyBullet / SciPy (scalar w at index 3).
    """
    x, y, z, w = xyzw
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def quat_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical interpolation (slerp) between two quaternions [x, y, z, w].    
    """
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = min(1.0, dot)

    if dot > 0.9995:
        result = a + t * (b - a)
        return result / np.linalg.norm(result)

    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_t = math.sin(theta)
    sin_t0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_t / sin_t0
    s1 = sin_t / sin_t0
    result = s0 * a + s1 * b
    return result / np.linalg.norm(result)

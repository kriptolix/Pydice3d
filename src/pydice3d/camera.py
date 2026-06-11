"""
camera.py – 3D Camera

Coordinate Convention
─────────────────────────
Y+ = top of the world
-Z = default viewing direction (camera looks at -Z in camera space)
NDC: X ∈ [-1,1], Y ∈ [-1,1], Z ∈ [-1,1] (OpenGL default)

Matrices returned as float32 column-major (compatible with glUniformMatrix4fv
with transpose=GL_FALSE, which is the OpenGL/GLSL default).
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field


# ────────────────────────────────────────────────────────────────────────────
# Helpers de álgebra linear
# ────────────────────────────────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)

    if norm > 1e-12:
        return v / norm

    return v.copy()


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Constructs the 4x4 view matrix.

    Parameters

    ---------
    eye: camera position in the world
    target: point the camera is pointing to
    up: "top" vector of the world (usually [0,1,0])

    Returns

    -------
    float32 (4, 4) — transforms world coordinates into camera space
    """
    f = _normalize(target - eye)          # forward (−Z câmera)
    r = _normalize(np.cross(f, _normalize(up)))   # right
    u = np.cross(r, f)                    # up real (reortogonalizado)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] =  r
    m[1, :3] =  u
    m[2, :3] = -f
    m[0, 3]  = -float(np.dot(r, eye))
    m[1, 3]  = -float(np.dot(u, eye))
    m[2, 3]  =  float(np.dot(f, eye))   # forward aponta para -Z, dot negativo é correto
    return m


def perspective(fov_y_rad: float, aspect: float,
                near: float, far: float) -> np.ndarray:
    """
    Constructs the 4×4 perspective projection matrix.

    Parameters

    ---------
    fov_y_rad : vertical field of view in radians
    aspect : viewport width / height
    near : near clipping plane (> 0)
    far : far clipping plane

    Returns

    -------
    float32 (4, 4) — perspective projection (standard OpenGL NDC)
    """
    f = 1.0 / math.tan(fov_y_rad / 2.0)
    rng = near - far

    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (near + far) / rng
    m[2, 3] = (2.0 * near * far) / rng
    m[3, 2] = -1.0
    return m
    

@dataclass
class Camera:
    """
    Orbital camera with ready-to-use default settings.

    By default, the camera is positioned above and behind the data table, facing the origin.

    The camera position is derived from spherical coordinates (azimuth, elevation, radius) around 
    a target.

    Optional controls:

    - orbit(): rotates around the target
    - zoom(): zooms in/out
    - pan(): moves the target    
    """

    target: np.ndarray = field(
        default_factory=lambda: np.array([0., 0., 0.], dtype=float)
    )

    # Aproximação equivalente ao antigo eye=[0, 8, 14]
    azimuth_deg: float = 90.0
    elevation_deg: float = 29.745
    radius: float = 16.125

    fov_y_deg: float = 45.0
    near: float = 0.1
    far: float = 200.0

    # Limites de elevação (evita singularidades nos polos)
    _ELEV_MIN: float = field(default=2.0, init=False, repr=False)
    _ELEV_MAX: float = field(default=88.0, init=False, repr=False)

    @classmethod
    def from_eye_target(
        cls,
        eye: np.ndarray,
        target: np.ndarray,
        *,
        fov_y_deg: float = 45.0,
        near: float = 0.1,
        far: float = 200.0,
    ) -> "Camera":
        """
        Creates a camera based on position (eye) and target.
        """

        eye = np.asarray(eye, dtype=float)
        target = np.asarray(target, dtype=float)

        offset = eye - target
        radius = float(np.linalg.norm(offset))

        if radius < 1e-8:
            raise ValueError("The eye and the target cannot coincide.")

        x, y, z = offset

        azimuth_deg = math.degrees(math.atan2(z, x))
        elevation_deg = math.degrees(math.asin(y / radius))

        return cls(
            target=target,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            radius=radius,
            fov_y_deg=fov_y_deg,
            near=near,
            far=far,
        )

    def _spherical_from_eye_target(self, eye, target):
        eye = np.asarray(eye, dtype=float)
        target = np.asarray(target, dtype=float)

        offset = eye - target
        radius = np.linalg.norm(offset)

        x, y, z = offset

        azimuth_deg = math.degrees(math.atan2(z, x))
        elevation_deg = math.degrees(math.asin(y / radius))

        return azimuth_deg, elevation_deg, radius
    
    def eye_position(self) -> np.ndarray:
        """
        Calculate the Cartesian position of the camera.
        """
        az = math.radians(self.azimuth_deg)
        el = math.radians(self.elevation_deg)

        cos_el = math.cos(el)

        x = self.radius * cos_el * math.cos(az)
        y = self.radius * math.sin(el)
        z = self.radius * cos_el * math.sin(az)

        return np.asarray(self.target, dtype=float) + np.array(
            [x, y, z],
            dtype=float,
        )

    def view_matrix(self) -> np.ndarray:
        eye = self.eye_position()

        return look_at(
            eye,
            np.asarray(self.target, dtype=float),
            np.array([0., 1., 0.], dtype=float),
        )

    def projection_matrix(self, width: int, height: int) -> np.ndarray:
        aspect = width / max(height, 1)

        return perspective(
            math.radians(self.fov_y_deg),
            aspect,
            self.near,
            self.far,
        )

    def view_projection(self, width: int, height: int) -> np.ndarray:
        return (
            self.projection_matrix(width, height)
            @ self.view_matrix()
        ).astype(np.float32)

    @property
    def position(self) -> np.ndarray:
        return self.eye_position().astype(np.float32)

    # ── Controles opcionais ──────────────────────────────────────────

    def orbit(self, delta_az: float, delta_el: float) -> None:
        """
        Rotaciona a câmera por deltas em graus.
        """
        self.azimuth_deg += delta_az

        self.elevation_deg = float(
            np.clip(
                self.elevation_deg + delta_el,
                self._ELEV_MIN,
                self._ELEV_MAX,
            )
        )

    def zoom(self, factor: float) -> None:
        """
        Multiplica o raio por `factor`.

        factor > 1  → afasta
        factor < 1  → aproxima
        """
        self.radius = float(
            np.clip(
                self.radius * factor,
                2.0,
                500.0,
            )
        )

    def pan(self, delta_x: float, delta_y: float) -> None:
        """
        Translada o alvo no plano da câmera.
        """
        V = self.view_matrix()

        right = V[0, :3].astype(float)
        up = V[1, :3].astype(float)

        self.target = (
            np.asarray(self.target, dtype=float)
            - right * delta_x
            + up * delta_y
        )
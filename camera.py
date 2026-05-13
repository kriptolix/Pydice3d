"""
camera.py – Câmera 3D (Fixa e Orbital)

Responsabilidade: calcular matrizes de view e projeção perspectiva.
Puro NumPy — sem dependência de OpenGL ou GTK.

Dois modos
──────────
FixedCamera    : posição e alvo fixos; útil para testes e câmera padrão.
OrbitalCamera  : gira ao redor de um alvo via ângulos azimute/elevação/raio.
                 Suporta zoom e pan; atualizada por input do usuário.

Convenção de coordenadas
─────────────────────────
    Y+  = cima do mundo
    -Z  = direção de visão padrão (câmera olha para -Z no espaço de câmera)
    NDC : X ∈ [-1,1], Y ∈ [-1,1], Z ∈ [-1,1]  (OpenGL padrão)

Matrizes retornadas como float32 column-major (compatível com glUniformMatrix4fv
com transpose=GL_FALSE, que é o padrão OpenGL/GLSL).
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field


# ────────────────────────────────────────────────────────────────────────────
# Helpers de álgebra linear
# ────────────────────────────────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v.copy()


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Constrói a matriz de view 4×4 (look-at).

    Parâmetros
    ----------
    eye    : posição da câmera no mundo
    target : ponto para o qual a câmera aponta
    up     : vetor "cima" do mundo (normalmente [0,1,0])

    Retorna
    -------
    float32 (4, 4) — transforma coordenadas do mundo para espaço de câmera
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
    Constrói a matriz de projeção perspectiva 4×4.

    Parâmetros
    ----------
    fov_y_rad : campo de visão vertical em radianos
    aspect    : largura / altura do viewport
    near      : plano de clipping próximo (> 0)
    far       : plano de clipping distante

    Retorna
    -------
    float32 (4, 4) — projeção perspectiva (NDC padrão OpenGL)
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


def orthographic(left: float, right: float, bottom: float, top: float,
                 near: float, far: float) -> np.ndarray:
    """
    Projeção ortográfica 4×4 (útil para debug / HUD).
    """
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] =  2.0 / (right - left)
    m[1, 1] =  2.0 / (top   - bottom)
    m[2, 2] = -2.0 / (far   - near)
    m[0, 3] = -(right + left)   / (right - left)
    m[1, 3] = -(top   + bottom) / (top   - bottom)
    m[2, 3] = -(far   + near)   / (far   - near)
    m[3, 3] =  1.0
    return m


# ────────────────────────────────────────────────────────────────────────────
# FixedCamera
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class FixedCamera:
    """
    Câmera com posição e alvo fixos.

    Parâmetros padrão posicionam a câmera acima e atrás da mesa de dados,
    olhando para a origem.

    Atributos
    ---------
    eye        : posição da câmera no mundo
    target     : ponto-alvo
    up         : vetor "cima"
    fov_y_deg  : campo de visão vertical em graus
    near, far  : planos de clipping
    """
    eye:       np.ndarray = field(default_factory=lambda: np.array([0., 8., 14.], dtype=float))
    target:    np.ndarray = field(default_factory=lambda: np.array([0., 0., 0.],  dtype=float))
    up:        np.ndarray = field(default_factory=lambda: np.array([0., 1., 0.],  dtype=float))
    fov_y_deg: float = 45.0
    near:      float = 0.1
    far:       float = 200.0

    def view_matrix(self) -> np.ndarray:
        """Retorna a matriz de view 4×4 float32."""
        return look_at(
            np.asarray(self.eye,    dtype=float),
            np.asarray(self.target, dtype=float),
            np.asarray(self.up,     dtype=float),
        )

    def projection_matrix(self, width: int, height: int) -> np.ndarray:
        """
        Retorna a matriz de projeção perspectiva 4×4 float32.

        Parâmetros
        ----------
        width, height : dimensões do viewport em pixels
        """
        aspect = width / max(height, 1)
        return perspective(
            math.radians(self.fov_y_deg),
            aspect, self.near, self.far,
        )

    def view_projection(self, width: int, height: int) -> np.ndarray:
        """VP = P × V (float32, 4×4)."""
        P = self.projection_matrix(width, height)
        V = self.view_matrix()
        return (P @ V).astype(np.float32)

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.eye, dtype=np.float32)


# ────────────────────────────────────────────────────────────────────────────
# OrbitalCamera (Not use right now)
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class OrbitalCamera:
    """
    Câmera orbital: gira ao redor de um alvo em coordenadas esféricas.

    Azimute e elevação são em graus para facilidade de uso.
    A posição cartesiana da câmera é derivada de (azimuth, elevation, radius).

    Controles sugeridos (implementados na Fase 6 via GTK):
        - Arrastar: azimuth ± elevation
        - Scroll:   zoom (radius)
        - Botão do meio: pan (target)

    Atributos
    ---------
    target      : ponto em torno do qual a câmera orbita
    azimuth_deg : ângulo horizontal em graus (0 = +X)
    elevation_deg: ângulo vertical em graus (0 = plano XZ, 90 = topo)
    radius      : distância ao alvo
    fov_y_deg   : campo de visão vertical em graus
    near, far   : planos de clipping
    """
    target:        np.ndarray = field(default_factory=lambda: np.array([0., 1., 0.], dtype=float))
    azimuth_deg:   float = 30.0
    elevation_deg: float = 35.0
    radius:        float = 18.0
    fov_y_deg:     float = 45.0
    near:          float = 0.1
    far:           float = 200.0

    # Limites de elevação (evita gimbal lock nos polos)
    _ELEV_MIN: float = field(default=2.0,  init=False, repr=False)
    _ELEV_MAX: float = field(default=88.0, init=False, repr=False)

    def eye_position(self) -> np.ndarray:
        """Calcula posição cartesiana da câmera a partir de coordenadas esféricas."""
        az  = math.radians(self.azimuth_deg)
        el  = math.radians(self.elevation_deg)
        cos_el = math.cos(el)
        x = self.radius * cos_el * math.cos(az)
        y = self.radius * math.sin(el)
        z = self.radius * cos_el * math.sin(az)
        return np.asarray(self.target, dtype=float) + np.array([x, y, z])

    def view_matrix(self) -> np.ndarray:
        eye = self.eye_position()
        return look_at(eye, np.asarray(self.target, dtype=float),
                       np.array([0., 1., 0.]))

    def projection_matrix(self, width: int, height: int) -> np.ndarray:
        aspect = width / max(height, 1)
        return perspective(math.radians(self.fov_y_deg), aspect, self.near, self.far)

    def view_projection(self, width: int, height: int) -> np.ndarray:
        return (self.projection_matrix(width, height) @ self.view_matrix()).astype(np.float32)

    @property
    def position(self) -> np.ndarray:
        return self.eye_position().astype(np.float32)

    # ── Controles ────────────────────────────────────────────────────

    def orbit(self, delta_az: float, delta_el: float) -> None:
        """Rotaciona a câmera por deltas em graus."""
        self.azimuth_deg   += delta_az
        self.elevation_deg  = float(np.clip(
            self.elevation_deg + delta_el,
            self._ELEV_MIN, self._ELEV_MAX,
        ))

    def zoom(self, factor: float) -> None:
        """Multiplica o raio por `factor` (> 1 = afasta, < 1 = aproxima)."""
        self.radius = float(np.clip(self.radius * factor, 2.0, 500.0))

    def pan(self, delta_x: float, delta_y: float) -> None:
        """
        Translada o alvo no plano da câmera por (delta_x, delta_y) unidades.
        """
        V = self.view_matrix()
        right = V[0, :3].astype(float)   # linha 0 do view = right
        up    = V[1, :3].astype(float)   # linha 1 do view = up
        self.target = np.asarray(self.target, dtype=float) \
                    - right * delta_x + up * delta_y
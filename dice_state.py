"""
dice_state.py – Estado e Ciclo de Vida do Dado (PyBullet)

Responsabilidade: manter o ciclo de vida de cada dado e capturar o resultado
quando o dado para. Orientação e velocidades vêm do PyBullet via
getBasePositionAndOrientation / getBaseVelocity.

Os helpers de quaternion foram mantidos apenas para quat_to_matrix (necessário
em top_face_value e rotation_matrix).

Ciclo de vida: SPAWNED → ROLLING → SETTLING → RESTING
"""

from __future__ import annotations

import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pybullet as p

from dice import Dice


# ────────────────────────────────────────────────────────────────────────────
# Quaternion helpers (apenas o necessário para quat_to_matrix)
# ────────────────────────────────────────────────────────────────────────────

def quat_to_matrix(xyzw: np.ndarray) -> np.ndarray:
    """
    Quaternion [x, y, z, w] (formato PyBullet) → matriz de rotação 3×3.
    """
    x, y, z, w = xyzw
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def quat_slerp_xyzw(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """
    slerp entre dois quaternions [x,y,z,w]. Mantido para interpolated_rotation_matrix
    (usado pelo renderer para suavizar frames).
    """
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b   = -b
        dot = -dot
    dot = min(1.0, dot)
    if dot > 0.9995:
        return (a + t * (b - a)) / np.linalg.norm(a + t * (b - a))
    theta_0 = math.acos(dot)
    theta   = theta_0 * t
    sin_t   = math.sin(theta)
    sin_t0  = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_t / sin_t0
    s1 = sin_t / sin_t0
    result = s0 * a + s1 * b
    return result / np.linalg.norm(result)


# ────────────────────────────────────────────────────────────────────────────
# Constantes de estabilização
# ────────────────────────────────────────────────────────────────────────────

# Limiar de velocidade linear (m/s) e angular (rad/s) para considerar "quase parado".
SETTLING_LINEAR_THRESHOLD:  float = 0.05   # m/s
SETTLING_ANGULAR_THRESHOLD: float = 0.10   # rad/s

SETTLING_FRAMES_REQUIRED: int = 20
RESTING_FRAMES_REQUIRED:  int = 30

# Altura máxima do centro do dado para aceitar RESTING (evita dado no ar).
MAX_RESTING_HEIGHT: float = 1.5


# ────────────────────────────────────────────────────────────────────────────
# Ciclo de vida
# ────────────────────────────────────────────────────────────────────────────

class DiceStatus(Enum):
    SPAWNED  = auto()
    ROLLING  = auto()
    SETTLING = auto()
    RESTING  = auto()


# ────────────────────────────────────────────────────────────────────────────
# DiceState
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class DiceState:
    """
    Estado mutável de ciclo de vida de um dado no mundo PyBullet.

    Orientação e velocidades são sempre lidas do PyBullet — não há estado
    angular local. O único estado mantido aqui são os contadores de
    estabilização e o resultado final.

    Para interpolação de frames pelo renderer, prev_orientation é salvo
    a cada update_status (quaternion [x,y,z,w] do PyBullet).
    """
    dice: Dice

    status: DiceStatus    = DiceStatus.SPAWNED
    result: Optional[int] = None

    # Quaternions [x,y,z,w] para interpolação de render
    prev_orientation: np.ndarray = field(
        default_factory=lambda: np.array([0., 0., 0., 1.])
    )

    _settling_frames: int = field(default=0, repr=False)
    _resting_frames:  int = field(default=0, repr=False)

    # ── construtor semântico ─────────────────────────────────────────

    @classmethod
    def create(cls, dice: Dice) -> "DiceState":
        """Cria estado inicial para um dado já registrado no PyBullet."""
        _, orn = p.getBasePositionAndOrientation(dice.body_id)
        return cls(
            dice=dice,
            prev_orientation=np.array(orn, dtype=np.float64),
        )

    # ── leitura de estado do PyBullet ────────────────────────────────

    @property
    def orientation_quat(self) -> np.ndarray:
        """Quaternion atual [x,y,z,w] lido do PyBullet."""
        _, orn = p.getBasePositionAndOrientation(self.dice.body_id)
        return np.array(orn, dtype=np.float64)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Matriz de rotação 3×3 atual."""
        return quat_to_matrix(self.orientation_quat)

    @property
    def linear_velocity(self) -> np.ndarray:
        lin, _ = p.getBaseVelocity(self.dice.body_id)
        return np.array(lin, dtype=np.float64)

    @property
    def angular_velocity(self) -> np.ndarray:
        _, ang = p.getBaseVelocity(self.dice.body_id)
        return np.array(ang, dtype=np.float64)

    # ── estabilização ────────────────────────────────────────────────

    def update_status(self) -> None:
        """
        Atualiza ciclo de vida lendo velocidades do PyBullet.Também 
        salva prev_orientation para interpolação do renderer.
        """
        if self.status == DiceStatus.RESTING:
            return

        # Salva orientação anterior para interpolação
        self.prev_orientation = self.orientation_quat

        lin = self.linear_velocity
        ang = self.angular_velocity

        lin_speed_xz = math.sqrt(float(lin[0])**2 + float(lin[2])**2)
        ang_speed    = float(np.linalg.norm(ang))
        moving       = (lin_speed_xz > SETTLING_LINEAR_THRESHOLD or
                        ang_speed     > SETTLING_ANGULAR_THRESHOLD)

        if self.status == DiceStatus.SPAWNED:
            self.status = DiceStatus.ROLLING
            return

        if moving:
            self.status           = DiceStatus.ROLLING
            self._settling_frames = 0
            self._resting_frames  = 0
            return

        self._settling_frames += 1

        if self.status == DiceStatus.ROLLING:
            if self._settling_frames >= SETTLING_FRAMES_REQUIRED:
                pos = self.dice.position
                if pos[1] <= MAX_RESTING_HEIGHT:
                    self.status          = DiceStatus.SETTLING
                    self._resting_frames = 0
            return

        if self.status == DiceStatus.SETTLING:
            self._resting_frames += 1
            if self._resting_frames >= RESTING_FRAMES_REQUIRED:
                pos = self.dice.position
                if pos[1] <= MAX_RESTING_HEIGHT:
                    self.status = DiceStatus.RESTING
                    self.result = self.top_face_value
                else:
                    self.status           = DiceStatus.ROLLING
                    self._settling_frames = 0
                    self._resting_frames  = 0

    # ── conveniências ────────────────────────────────────────────────

    @property
    def top_face_value(self) -> int:
        return self.dice.top_face_value(self.rotation_matrix)

    @property
    def top_face_index(self) -> int:
        return self.dice.top_face_index(self.rotation_matrix)

    @property
    def is_resting(self) -> bool:
        return self.status == DiceStatus.RESTING

    @property
    def world_vertices(self) -> np.ndarray:
        return self.dice.world_vertices(self.rotation_matrix)

    @property
    def world_face_normals(self) -> np.ndarray:
        return self.dice.world_face_normals(self.rotation_matrix)

    def interpolated_rotation_matrix(self, alpha: float) -> np.ndarray:
        """
        Interpola entre prev_orientation e a orientação atual via slerp.
        alpha=0 → frame anterior, alpha=1 → frame atual.
        Usado pelo renderer para suavizar entre steps de física.
        """
        q = quat_slerp_xyzw(self.prev_orientation, self.orientation_quat, alpha)
        return quat_to_matrix(q)

    def __repr__(self) -> str:
        return (f"DiceState({self.dice.dice_type}, "
                f"status={self.status.name}, result={self.result})")
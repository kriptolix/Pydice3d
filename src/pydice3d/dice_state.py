"""
dice_state.py – Data State and Lifecycle (PyBullet). Maintain the lifecycle of each 
dice (SPAWNED → ROLLING → SETTLING → RESTING) and expose guidance/speed via PyBullet.
"""

from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pybullet as pb
import math

from pydice3d.dice import Dice


from pydice3d.math_utils import quat_to_matrix, quat_slerp


SETTLING_LINEAR_THRESHOLD:  float = 0.05   # m/s
SETTLING_ANGULAR_THRESHOLD: float = 0.10   # rad/s

SETTLING_FRAMES_REQUIRED: int = 20
RESTING_FRAMES_REQUIRED:  int = 30

SETTLING_TIMEOUT_FRAMES: int = 180   # ~3 s a 60 Hz


class DiceStatus(Enum):
    SPAWNED = auto()
    ROLLING = auto()
    SETTLING = auto()
    RESTING = auto()


@dataclass
class DiceState:
    """
    Mutable lifecycle state of a given data point in the PyBullet world.    
    """
    dice: Dice

    status: DiceStatus = DiceStatus.SPAWNED

    prev_orientation: np.ndarray = field(
        default_factory=lambda: np.array([0., 0., 0., 1.])
    )

    _settling_frames: int = field(default=0, repr=False)
    _resting_frames:  int = field(default=0, repr=False)
    _settling_total:  int = field(default=0, repr=False)

    @classmethod
    def create(cls, dice: Dice) -> "DiceState":
        """Creates an initial state for data already registered in PyBullet."""
        _, orn = pb.getBasePositionAndOrientation(dice.body_id)
        return cls(
            dice=dice,
            prev_orientation=np.array(orn, dtype=np.float64),
        )

    @property
    def orientation_quat(self) -> np.ndarray:
        _, orn = pb.getBasePositionAndOrientation(self.dice.body_id)
        return np.array(orn, dtype=np.float64)

    @property
    def rotation_matrix(self) -> np.ndarray:
        return quat_to_matrix(self.orientation_quat)

    @property
    def linear_velocity(self) -> np.ndarray:
        lin, _ = pb.getBaseVelocity(self.dice.body_id)
        return np.array(lin, dtype=np.float64)

    @property
    def angular_velocity(self) -> np.ndarray:
        _, ang = pb.getBaseVelocity(self.dice.body_id)
        return np.array(ang, dtype=np.float64)

    def update_status(self) -> None:
        """
        Updates lifecycle by reading speeds from PyBullet. Also
        saves prev_orientation for renderer interpolation.        
        """
        if self.status == DiceStatus.RESTING:
            return

        self.prev_orientation = self.orientation_quat

        lin = self.linear_velocity
        ang = self.angular_velocity

        lin_speed_xz = math.sqrt(float(lin[0])**2 + float(lin[2])**2)
        ang_speed = float(np.linalg.norm(ang))
        moving = (lin_speed_xz > SETTLING_LINEAR_THRESHOLD or
                  ang_speed > SETTLING_ANGULAR_THRESHOLD)

        if self.status == DiceStatus.SPAWNED:
            self.status = DiceStatus.ROLLING
            return

        if moving:
            self.status = DiceStatus.ROLLING
            self._settling_frames = 0
            self._resting_frames = 0

            return

        self._settling_frames += 1
        self._settling_total += 1

        if self.status == DiceStatus.ROLLING:
            if self._settling_frames >= SETTLING_FRAMES_REQUIRED:
                self.status = DiceStatus.SETTLING
                self._resting_frames = 0
            return

        if self.status == DiceStatus.SETTLING:
            self._resting_frames += 1

            if self._resting_frames >= RESTING_FRAMES_REQUIRED:
                self.status = DiceStatus.RESTING
                return

            if self._settling_total >= SETTLING_TIMEOUT_FRAMES:
                self.status = DiceStatus.RESTING

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

        q = quat_slerp(self.prev_orientation, self.orientation_quat, alpha)
        return quat_to_matrix(q)

    def __repr__(self) -> str:
        return (f"DiceState({self.dice.dice_type}, "
                f"status={self.status.name})")

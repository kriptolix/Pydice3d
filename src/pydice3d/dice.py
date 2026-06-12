"""
dice.py – Domain Entity: Data (PyBullet). unites the physical body of the 
PyBullet (body_id) with the geometric mesh (DiceMesh).
"""

from __future__ import annotations
from pydice3d.math_utils import quat_to_matrix as _quat_to_matrix

import numpy as np
from dataclasses import dataclass

import pybullet as pb

from pydice3d.dice_mesh import DiceMesh, DiceType, get_mesh


DEFAULT_SCALE: float = 1.0


@dataclass
class Dice:

    body_id:   int
    mesh:      DiceMesh
    dice_type: str
    scale:     float = DEFAULT_SCALE

    @classmethod
    def create(
        cls,
        dice_type: DiceType,
        position:  tuple | list | np.ndarray,
        physics,                               # PhysicsWorld
        scale:     float = DEFAULT_SCALE,
        name:      str = "",
    ) -> "Dice":

        body_id = physics.create_dice_body(dice_type, position, scale)
        mesh = get_mesh(dice_type, scale=scale)
        return cls(body_id=body_id, mesh=mesh, dice_type=dice_type, scale=scale)

    @property
    def position(self) -> np.ndarray:
        pos, _ = pb.getBasePositionAndOrientation(self.body_id)
        return np.array(pos, dtype=np.float32)

    @property
    def orientation_quat(self) -> np.ndarray:

        _, orn = pb.getBasePositionAndOrientation(self.body_id)
        return np.array(orn, dtype=np.float64)

    @property
    def orientation_matrix(self) -> np.ndarray:

        xyzw = self.orientation_quat
        # Converte [x,y,z,w] → [w,x,y,z] para quat_to_matrix
        w, x, y, z = xyzw[3], xyzw[0], xyzw[1], xyzw[2]
        return _quat_wxyz_to_matrix(w, x, y, z)

    @property
    def num_faces(self) -> int:
        return self.mesh.num_faces

    @property
    def num_vertices(self) -> int:
        return self.mesh.num_vertices

    def __repr__(self) -> str:
        pos = self.position
        return (f"Dice({self.dice_type}, "
                f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
                f"scale={self.scale})")


def _quat_wxyz_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Compatibility shim: accepts separate components, delegates to math_utils."""
    return _quat_to_matrix(np.array([x, y, z, w], dtype=np.float64))

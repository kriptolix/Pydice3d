"""
dice.py – Entidade de Domínio: Dado (PyBullet)

Responsabilidade: unir o corpo físico do PyBullet (body_id) com a malha
geométrica (DiceMesh), representando um dado completo.

Posição e orientação são sempre consultadas via bullet.getBasePositionAndOrientation,
nunca armazenadas localmente — PyBullet é a fonte da verdade para estado físico.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

import pybullet as p

from dice_mesh import DiceMesh, DiceType, get_mesh


DEFAULT_SCALE: float = 1.0


@dataclass
class Dice:
    """
    Dado poliédrico: body_id PyBullet + geometria.

    Atributos
    ---------
    body_id   : identificador do corpo no mundo PyBullet
    mesh      : malha geométrica imutável (vértices, faces, normais)
    dice_type : string identificadora ("d6", "d20", etc.)
    scale     : escala visual aplicada
    """
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
        name:      str   = "",
    ) -> "Dice":
        """
        Cria um dado registrando seu corpo no PhysicsWorld fornecido.

        Parâmetros
        ----------
        dice_type : "d4", "d6", "d8", "d10", "d12" ou "d20"
        position  : posição inicial no mundo (x, y, z)
        physics   : instância de PhysicsWorld
        scale     : fator de escala visual
        """
        body_id = physics.create_dice_body(dice_type, position, scale)
        mesh    = get_mesh(dice_type, scale=scale)
        return cls(body_id=body_id, mesh=mesh, dice_type=dice_type, scale=scale)

    # ------------------------------------------------------------------
    # Estado físico (sempre via PyBullet)
    # ------------------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        return np.array(pos, dtype=np.float32)

    @property
    def orientation_quat(self) -> np.ndarray:
        """Quaternion [x, y, z, w] — formato nativo do PyBullet."""
        _, orn = p.getBasePositionAndOrientation(self.body_id)
        return np.array(orn, dtype=np.float64)

    @property
    def orientation_matrix(self) -> np.ndarray:
        """Matriz de rotação 3×3 derivada do quaternion do PyBullet."""
        xyzw = self.orientation_quat
        # Converte [x,y,z,w] → [w,x,y,z] para quat_to_matrix
        w, x, y, z = xyzw[3], xyzw[0], xyzw[1], xyzw[2]
        return _quat_wxyz_to_matrix(w, x, y, z)

    # ------------------------------------------------------------------
    # Geometria no mundo
    # ------------------------------------------------------------------

    def world_vertices(self, orientation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Vértices da malha no espaço do mundo.

        Se `orientation` não for fornecida, usa orientation_matrix do PyBullet.
        """
        R   = orientation if orientation is not None else self.orientation_matrix
        return (R @ self.mesh.vertices.T).T + self.position

    def world_face_normals(self, orientation: Optional[np.ndarray] = None) -> np.ndarray:
        """Normais das faces no espaço do mundo."""
        R = orientation if orientation is not None else self.orientation_matrix
        return (R @ self.mesh.normals.T).T

    def top_face_index(self, orientation: Optional[np.ndarray] = None) -> int:
        """Índice da face mais voltada para cima (+Y)."""
        up      = np.array([0.0, 1.0, 0.0])
        normals = self.world_face_normals(orientation)
        return int(np.argmax(normals @ up))

    def top_face_value(self, orientation: Optional[np.ndarray] = None) -> int:
        """Valor numérico da face superior."""
        return self.mesh.face_values[self.top_face_index(orientation)]

    # ------------------------------------------------------------------
    # Conveniências
    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helper interno — conversão quaternion → matriz
# ---------------------------------------------------------------------------

def _quat_wxyz_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Factory para múltiplos dados
# ---------------------------------------------------------------------------

def create_dice_set(
    spec:    dict[DiceType, int],
    physics,
    origin:  tuple = (0.0, 3.0, 0.0),
    spacing: float = 2.5,
    scale:   float = DEFAULT_SCALE,
) -> list[Dice]:
    """
    Cria um conjunto de dados a partir de um spec {tipo: quantidade}.

    Distribui os dados em linha a partir de `origin` com espaçamento `spacing`.

    Exemplo
    -------
    dice_set = create_dice_set({"d6": 2, "d20": 1}, physics)
    """
    dice_list: list[Dice] = []
    ox, oy, oz = origin

    for dtype, count in spec.items():
        for i in range(count):
            x = ox + len(dice_list) * spacing
            dice = Dice.create(
                dice_type=dtype,
                position=(x, oy, oz),
                physics=physics,
                scale=scale,
                name=f"{dtype}_{i+1}",
            )
            dice_list.append(dice)

    return dice_list
"""
dice_mesh.py – Define the 3D mesh (vertices + faces + normals)
for each supported data type.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal

DiceType = Literal["d4", "d6", "d8", "d10", "d12", "d20", "d100", "df"]
ALL_DICE: tuple[DiceType, ...] = (
    "d4", "d6", "d8", "d10", "d12", "d20", "d100", "df")


@dataclass(frozen=True)
class DiceMesh:
    """
    Immutable mesh of a given polyhedral.
    """
    dice_type: str
    vertices: np.ndarray      # shape (N, 3)
    faces: tuple              # tuple[tuple[int, ...], ...]
    normals: np.ndarray       # shape (F, 3)
    face_values: tuple        # tuple[int, ...]

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    def face_center(self, face_index: int) -> np.ndarray:
        
        idx = self.faces[face_index]
        return self.vertices[list(idx)].mean(axis=0)

    def scaled(self, scale: float) -> "DiceMesh":
        
        return DiceMesh(
            dice_type=self.dice_type,
            vertices=self.vertices * scale,
            faces=self.faces,
            normals=self.normals,          
            face_values=self.face_values,
        )

    def triangulated_faces(self) -> list[tuple[int, int, int]]:
        
        tris = []
        for face in self.faces:
            face = list(face)
            for i in range(1, len(face) - 1):
                tris.append((face[0], face[i], face[i + 1]))
        return tris


def _normalize(v: np.ndarray) -> np.ndarray:
    
    n = np.linalg.norm(v)
    
    if n <= 1e-12:
        return np.zeros(3)

    normalized = v / n

    return normalized


def _project_to_sphere(vertices: np.ndarray) -> np.ndarray:
    
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    
    return vertices / norms


def _face_normal(vertices: np.ndarray, face: tuple[int, ...]) -> np.ndarray:
    """
    Calculate the normal to a face using the equation of the mid-plane.    
    """
    verts = vertices[list(face)]
    n = len(verts)

    normal = np.zeros(3)
    for i in range(n):
        v0 = verts[i]
        v1 = verts[(i + 1) % n]
        normal += np.cross(v0, v1)

    return _normalize(normal)


def _compute_normals(vertices: np.ndarray, faces: list[list[int]]) -> np.ndarray:
    
    normals = []

    for face in faces:
        normal = _face_normal(vertices, tuple(face))
        normals.append(normal)

    return np.array(normals)


def _ensure_outward_normals(vertices: np.ndarray, faces: list[list[int]]) -> list[list[int]]:
    
    corrected = []

    for face in faces:
        center = vertices[face].mean(axis=0)
        normal = _face_normal(vertices, tuple(face))
        if np.dot(normal, center) < 0:
            corrected.append(list(reversed(face)))
        else:
            corrected.append(face)
    return corrected


def _build_d4() -> DiceMesh:
    """
    Regular tetrahedron — d4.
    4 vertices, 4 triangular faces.
    Values: 1–4 (face opposite the vertex).
    """

    # Tetrahedron inscribed in the unit sphere
    a = 1.0 / np.sqrt(3)
    raw = np.array([
        [a,  a,  a],
        [-a, -a,  a],
        [-a,  a, -a],
        [a, -a, -a],
    ])
    vertices = _project_to_sphere(raw)

    faces = [
        [1, 3, 2],  # opposite face to vertex 0 → value 1
        [0, 2, 3],  # opposite face to vertex 1 → value 2
        [0, 3, 1],  # opposite face to vertex 2 → value 3
        [0, 1, 2],  # opposite face to vertex 3 → value 4
    ]
    
    faces = _ensure_outward_normals(vertices, faces)
    normals = _compute_normals(vertices, faces)

    return DiceMesh(
        dice_type="d4",
        vertices=vertices,
        faces=tuple(tuple(f) for f in faces),
        normals=normals,
        face_values=(1, 2, 3, 4),
    )


def _build_d6(df=False) -> DiceMesh:
    """
    Cube — d6.
    8 vertices, 6 square faces.
    Values: 1–6, opposite faces sum to 7.
    """
    a = 1.0 / np.sqrt(3)
    raw = np.array([
        [-a, -a, -a], [a, -a, -a], [a,  a, -a], [-a,  a, -a],  # z=-a
        [-a, -a,  a], [a, -a,  a], [a,  a,  a], [-a,  a,  a],  # z=+a
    ])
    vertices = _project_to_sphere(raw)
    
    faces = [
        [4, 5, 6, 7],   # +Z → 1
        [3, 2, 1, 0],   # -Z → 6
        [0, 1, 5, 4],   # -Y → 2
        [2, 3, 7, 6],   # +Y → 5
        [1, 2, 6, 5],   # +X → 3
        [3, 0, 4, 7],   # -X → 4
    ]
    normals = _compute_normals(vertices, faces)

    if df:
        return DiceMesh(
            dice_type="df",
            vertices=vertices,
            faces=tuple(tuple(f) for f in faces),
            normals=normals,
            face_values=(+1, +1, -1, -1, 0, 0),
        )

    return DiceMesh(
        dice_type="d6",
        vertices=vertices,
        faces=tuple(tuple(f) for f in faces),
        normals=normals,
        face_values=(1, 6, 2, 5, 3, 4),
    )


def _build_d8() -> DiceMesh:
    """
    Regular octahedron — d8.
    6 vertices, 8 triangular faces.
    Values: 1–8, opposite faces sum to 9.
    """
    vertices = np.array([
        [1,  0,  0],
        [-1, 0,  0],
        [0,  1,  0],
        [0, -1,  0],
        [0,  0,  1],
        [0,  0, -1],
    ], dtype=float)

    vertices = _project_to_sphere(vertices)

    faces = [
        [0, 2, 4],  # 1
        [2, 1, 4],  # 2
        [1, 3, 4],  # 3
        [3, 0, 4],  # 4
        [0, 5, 2],  # 5  
        [2, 5, 1],  # 6
        [1, 5, 3],  # 7
        [3, 5, 0],  # 8
    ]

    normals = _compute_normals(vertices, faces)
    
    return DiceMesh(
        dice_type="d8",
        vertices=vertices,
        faces=tuple(tuple(f) for f in faces),
        normals=normals,
        face_values=(1, 2, 3, 4, 5, 6, 7, 8),
    )


def _build_d10(d100=False) -> DiceMesh:
    """
    Pentagonal trapezohedron — d10.
    Dual of the pentagonal antiprism: 12 vertices, 10 kite faces.
    V=12, F=10, E=20 → Euler=2. Perfect manifold.
    """   

    POLE_HEIGHT = 0.70

    vertices = np.array([
        [0.44721360,  0.10557281,  0.32491970],  # 0
        [0.17082039, -0.10557281,  0.52573111],  # 1
        [-0.17082039,  0.10557281,  0.52573111],  # 2
        [-0.44721360, -0.10557281,  0.32491970],  # 3
        [-0.55278640,  0.10557281,  0.00000000],  # 4
        [-0.44721360, -0.10557281, -0.32491970],  # 5
        [-0.17082039,  0.10557281, -0.52573111],  # 6
        [0.17082039, -0.10557281, -0.52573111],  # 7
        [0.44721360,  0.10557281, -0.32491970],  # 8
        [0.55278640, -0.10557281,  0.00000000],  # 9
        [0.00000000,  1.00000000,  0.00000000],  # 10 up pole
        [0.00000000, -1.00000000,  0.00000000],  # 11 dawn pole
    ], dtype=float)

    vertices[10] = (
        0.0,
        POLE_HEIGHT,
        0.0
    )

    vertices[11] = (
        0.0,
        -POLE_HEIGHT,
        0.0
    )

    # 10 kite faces — each with an acute vertex (36°) at the pole,
    # two 108° vertices on the sides and one 108° vertex at the base.
    # Winding CCW seen from the outside.
    faces_raw = [
        (10,  0,  9,  8),  # 1
        (2,  1,  0, 10),  # 2
        (4,  3,  2, 10),  # 3
        (6,  5,  4, 10),  # 4
        (8,  7,  6, 10),  # 5
        (9,  0,  1, 11),  # 6
        (11,  1,  2,  3),  # 7
        (11,  3,  4,  5),  # 8
        (11,  5,  6,  7),  # 9
        (11,  7,  8,  9),  # 0
    ]

    normals = _compute_normals(vertices, [list(f) for f in faces_raw])

    if d100:
        return DiceMesh(
            dice_type="d100",
            vertices=vertices,
            faces=tuple(faces_raw),
            normals=normals,
            face_values=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90),
        )

    return DiceMesh(
        dice_type="d10",
        vertices=vertices,
        faces=tuple(faces_raw),
        normals=normals,
        face_values=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    )


def _build_d12() -> DiceMesh:
    """
    Regular dodecahedron — d12.
    20 vertices, 12 regular pentagonal faces.
    V=20, F=12, E=30 → Euler=2.    
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv = 1.0 / phi
    
    raw = np.array([
        # Class 1: (±1, ±1, ±1)
        [1,  1,  1], [-1,  1,  1], [1, -1,  1], [-1, -1,  1],
        [1,  1, -1], [-1,  1, -1], [1, -1, -1], [-1, -1, -1],
        # Class 2: (0, ±1/φ, ±φ)
        [0,  inv,  phi], [0, -inv,  phi],
        [0,  inv, -phi], [0, -inv, -phi],
        # Class 3: (±1/φ, ±φ, 0)
        [inv,  phi, 0], [-inv,  phi, 0],
        [inv, -phi, 0], [-inv, -phi, 0],
        # Class 4: (±φ, 0, ±1/φ)
        [phi, 0,  inv], [phi, 0, -inv],
        [-phi, 0,  inv], [-phi, 0, -inv],
    ], dtype=np.float64)

    
    raw -= raw.mean(axis=0)
    raw /= np.max(np.linalg.norm(raw, axis=1))

    # 12 hardcoded pentagonal faces (winding CCW seen from the outside)
    # Obtained only once via ConvexHull and now fixed — the dodecahedron
    # is deterministic given these vertices in the same order.
    
    faces = [
        [0,  8,  9,  2, 16],
        [0, 12, 13,  1,  8],
        [0, 16, 17,  4, 12],
        [1, 13,  5, 19, 18],
        [1, 18,  3,  9,  8],
        [2,  9,  3, 15, 14],
        [2, 14,  6, 17, 16],
        [3, 18, 19,  7, 15],
        [4, 10,  5, 13, 12],
        [4, 17,  6, 11, 10],
        [5, 10, 11,  7, 19],
        [6, 14, 15,  7, 11],
    ]

    faces = _ensure_outward_normals(raw, faces)
    normals = _compute_normals(raw, faces)

    return DiceMesh(
        dice_type="d12",
        vertices=raw,
        faces=tuple(tuple(f) for f in faces),
        normals=normals,
        face_values=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    )


def _build_d20() -> DiceMesh:
    """
    Regular icosahedron — d20.
    12 vertices, 20 equilateral triangular faces.
    V=12, F=20, E=30 → Euler=2.   
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    # 12 vertices: three orthogonal golden rectangles
    raw = np.array([
        [0,  1,  phi], [0, -1,  phi], [0,  1, -phi], [0, -1, -phi],
        [1,  phi,  0], [-1,  phi,  0], [1, -phi,  0], [-1, -phi,  0],
        [phi,  0,  1], [phi,  0, -1], [-phi,  0,  1], [-phi,  0, -1],
    ], dtype=np.float64)

    raw -= raw.mean(axis=0)
    raw /= np.max(np.linalg.norm(raw, axis=1))
     
    faces = [
        [0,  1,  8], [0,  8,  4], [0,  4,  5], [0,  5, 10], [0, 10,  1],
        [3,  2,  9], [3,  9,  6], [3,  6,  7], [3,  7, 11], [3, 11,  2],
        [1,  6,  8], [8,  6,  9], [8,  9,  4], [4,  9,  2], [4,  2,  5],
        [5,  2, 11], [5, 11, 10], [10, 11,  7], [10,  7,  1], [1,  7,  6],
    ]

    faces = _ensure_outward_normals(raw, faces)
    normals = _compute_normals(raw, faces)

    return DiceMesh(
        dice_type="d20",
        vertices=raw,
        faces=tuple(tuple(f) for f in faces),
        normals=normals,
        face_values=tuple(range(1, 21)),
    )


_BUILDERS = {
    "d4": lambda: _build_d4(),
    "d6": lambda: _build_d6(),
    "d8": lambda: _build_d8(),
    "d10": lambda: _build_d10(),
    "d12": lambda: _build_d12(),
    "d20": lambda: _build_d20(),
    "d100": lambda: _build_d10(d100=True),
    "df": lambda: _build_d6(df=True),
}

_MESH_CACHE: dict[str, DiceMesh] = {}


def get_mesh(dice_type: DiceType, scale: float = 1.0) -> DiceMesh:
    """
    Returns the DiceMesh for the requested type, with scaling applied.    
    """

    if dice_type not in _BUILDERS:
        raise ValueError(f"Unknown dice type: '{dice_type}'. "
                         f"Supported: {list(_BUILDERS.keys())}")

    if dice_type not in _MESH_CACHE:

        _MESH_CACHE[dice_type] = _BUILDERS[dice_type]()

    mesh = _MESH_CACHE[dice_type]
    return mesh if scale == 1.0 else mesh.scaled(scale)

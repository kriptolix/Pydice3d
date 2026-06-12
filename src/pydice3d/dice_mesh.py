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
        """Centróide de uma face (média dos seus vértices)."""
        idx = self.faces[face_index]
        return self.vertices[list(idx)].mean(axis=0)

    def scaled(self, scale: float) -> "DiceMesh":
        """Retorna nova DiceMesh com vértices escalados uniformemente."""
        return DiceMesh(
            dice_type=self.dice_type,
            vertices=self.vertices * scale,
            faces=self.faces,
            normals=self.normals,          # normais não mudam com escala uniforme
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
    """Normaliza vetor 3D. Retorna vetor zero se norma < epsilon."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros(3)


def _project_to_sphere(vertices: np.ndarray) -> np.ndarray:
    """Projeta todos os vértices na esfera unitária (normalização por linha)."""
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return vertices / norms


def _face_normal(vertices: np.ndarray, face: tuple[int, ...]) -> np.ndarray:
    """
    Calcula a normal de uma face pela equação do plano médio
    (soma de cross products para robustez em polígonos não-planares).
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
    """Computa normais para todas as faces."""
    return np.array([_face_normal(vertices, tuple(f)) for f in faces])


# ---------------------------------------------------------------------------
# Builders por tipo de dado
# ---------------------------------------------------------------------------

def _ensure_outward_normals(vertices: np.ndarray, faces: list[list[int]]) -> list[list[int]]:
    """
    Garante que as normais de todas as faces apontem para fora da esfera.
    Se dot(normal, centróide_da_face) < 0, inverte o winding da face.
    """
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
    Tetraedro regular — d4.
    4 vértices, 4 faces triangulares.
    Valores: 1–4 (face oposta ao vértice).
    """
    # Tetraedro inscrito na esfera unitária
    a = 1.0 / np.sqrt(3)
    raw = np.array([
        [a,  a,  a],
        [-a, -a,  a],
        [-a,  a, -a],
        [a, -a, -a],
    ])
    vertices = _project_to_sphere(raw)

    faces = [
        [1, 3, 2],  # face oposta ao vértice 0 → valor 1
        [0, 2, 3],  # face oposta ao vértice 1 → valor 2
        [0, 3, 1],  # face oposta ao vértice 2 → valor 3
        [0, 1, 2],  # face oposta ao vértice 3 → valor 4
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
    Cubo — d6.
    8 vértices, 6 faces quadradas.
    Valores: 1–6, faces opostas somam 7.
    """
    a = 1.0 / np.sqrt(3)
    raw = np.array([
        [-a, -a, -a], [a, -a, -a], [a,  a, -a], [-a,  a, -a],  # z=-a
        [-a, -a,  a], [a, -a,  a], [a,  a,  a], [-a,  a,  a],  # z=+a
    ])
    vertices = _project_to_sphere(raw)

    # Faces com orientação CCW vista de fora
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
    Octaedro regular — d8.
    6 vértices, 8 faces triangulares.
    Valores: 1–8, faces opostas somam 9.
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
        [0, 5, 2],  # 5  (oposta a 4 → 9-4=5 — par com face 3)
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
    Trapezoedro pentagonal — d10.
    Dual do antiprisma pentagonal: 12 vértices, 10 faces kite.
    V=12, F=10, E=20 → Euler=2. Manifold perfeito.

    Cada face é um kite com ângulos internos 36°, 108°, 108°, 108°
    (conforme a geometria canônica do dado de 10 faces).

    Construção: dual do antiprisma pentagonal regular (h=0.5, r=1.0),
    calculado pelo método do polo recíproco e normalizado para raio unitário.
    Vértices e faces hardcoded para eliminar dependência de scipy em runtime.
    """
    # Vértices pré-calculados (normalizados, raio máximo = 1.0)
    # Índices 0-9: anel equatorial (alternado +y/-y)
    # Índice 10: polo superior (+Y), índice 11: polo inferior (-Y)

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
        [0.00000000,  1.00000000,  0.00000000],  # 10 polo sup
        [0.00000000, -1.00000000,  0.00000000],  # 11 polo inf
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

    # 10 faces kite — cada uma com vértice agudo (36°) no polo,
    # dois vértices de 108° nos lados e um vértice de 108° na base.
    # Winding CCW visto de fora.
    faces_raw = [
        (10,  0,  9,  8),  # kite  1
        (2,  1,  0, 10),  # kite  2
        (4,  3,  2, 10),  # kite  3
        (6,  5,  4, 10),  # kite  4
        (8,  7,  6, 10),  # kite  5
        (9,  0,  1, 11),  # kite  6
        (11,  1,  2,  3),  # kite  7
        (11,  3,  4,  5),  # kite  8
        (11,  5,  6,  7),  # kite  9
        (11,  7,  8,  9),  # kite 10
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
    Dodecaedro regular — d12.
    20 vértices, 12 faces pentagonais regulares.
    V=20, F=12, E=30 → Euler=2.

    Vértices e faces hardcoded — elimina dependência de scipy em runtime.
    Derivados das três classes de simetria do dodecaedro (±1,±1,±1),
    (0, ±1/φ, ±φ) e permutações cíclicas, normalizados para raio unitário.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv = 1.0 / phi

    # 20 vértices em três classes de simetria
    raw = np.array([
        # Classe 1: (±1, ±1, ±1)
        [1,  1,  1], [-1,  1,  1], [1, -1,  1], [-1, -1,  1],
        [1,  1, -1], [-1,  1, -1], [1, -1, -1], [-1, -1, -1],
        # Classe 2: (0, ±1/φ, ±φ)
        [0,  inv,  phi], [0, -inv,  phi],
        [0,  inv, -phi], [0, -inv, -phi],
        # Classe 3: (±1/φ, ±φ, 0)
        [inv,  phi, 0], [-inv,  phi, 0],
        [inv, -phi, 0], [-inv, -phi, 0],
        # Classe 4: (±φ, 0, ±1/φ)
        [phi, 0,  inv], [phi, 0, -inv],
        [-phi, 0,  inv], [-phi, 0, -inv],
    ], dtype=np.float64)

    # Normaliza para raio unitário
    raw -= raw.mean(axis=0)
    raw /= np.max(np.linalg.norm(raw, axis=1))

    # 12 faces pentagonais hardcoded (winding CCW visto de fora)
    # Obtidas uma única vez via ConvexHull e agora fixas — o dodecaedro
    # é determinístico dado estes vértices na mesma ordem.
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
    Icosaedro regular — d20.
    12 vértices, 20 faces triangulares equiláteras.
    V=12, F=20, E=30 → Euler=2.

    Vértices e faces hardcoded — elimina dependência de scipy em runtime.
    Os 12 vértices são os cantos de três retângulos áureos ortogonais
    (proporção φ = golden ratio), normalizados para raio unitário.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    # 12 vértices: três retângulos áureos ortogonais
    raw = np.array([
        [0,  1,  phi], [0, -1,  phi], [0,  1, -phi], [0, -1, -phi],
        [1,  phi,  0], [-1,  phi,  0], [1, -phi,  0], [-1, -phi,  0],
        [phi,  0,  1], [phi,  0, -1], [-phi,  0,  1], [-phi,  0, -1],
    ], dtype=np.float64)

    raw -= raw.mean(axis=0)
    raw /= np.max(np.linalg.norm(raw, axis=1))

    # 20 faces triangulares hardcoded (winding CCW visto de fora)
    # Derivadas da topologia canônica do icosaedro.
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

# Cache de malhas (singleton por tipo — malhas são imutáveis)
_MESH_CACHE: dict[str, DiceMesh] = {}


def get_mesh(dice_type: DiceType, scale: float = 1.0) -> DiceMesh:
    """
    Retorna a DiceMesh para o tipo solicitado, com escala aplicada.
    Malhas base são cacheadas; escala cria nova instância leve.
    """

    if dice_type not in _BUILDERS:
        raise ValueError(f"Tipo de dado desconhecido: '{dice_type}'. "
                         f"Suportados: {list(_BUILDERS.keys())}")

    if dice_type not in _MESH_CACHE:

        _MESH_CACHE[dice_type] = _BUILDERS[dice_type]()

    mesh = _MESH_CACHE[dice_type]
    return mesh if scale == 1.0 else mesh.scaled(scale)

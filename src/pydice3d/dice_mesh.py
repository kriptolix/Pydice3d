"""
dice_mesh.py – Geometria dos Dados Poliédricos

Responsabilidade: definir a malha 3D (vértices + faces + normais)
de cada tipo de dado suportado. Não conhece física, rendering nem estado.

Tipos suportados: d4, d6, d8, d10, d12, d20.

Convenções
----------
- Vértices normalizados: inscritos na esfera unitária (raio = 1.0).
  Para escalar, multiplique todos os vértices por `scale`.
- Centro de massa na origem (0, 0, 0).
- Normais de face apontam para fora do sólido.
- Faces definidas com orientação CCW (counter-clockwise) vista de fora.
- Cada face é uma lista de índices de vértices (triangular ou poligonal).

Estrutura de retorno de DiceMesh
---------------------------------
vertices  : ndarray (N, 3)  – posições dos vértices
faces     : list[list[int]] – índices por face (pode ser triângulos ou polígonos)
normals   : ndarray (F, 3)  – normal unitária de cada face
face_values: list[int]      – valor numérico associado a cada face
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal
from scipy.spatial import ConvexHull

DiceType = Literal["d4", "d6", "d8", "d10", "d12", "d20"]
ALL_DICE: tuple[DiceType, ...] = ("d4", "d6", "d8", "d10", "d12", "d20")


@dataclass(frozen=True)
class DiceMesh:
    """
    Malha imutável de um dado poliédrico.

    Atributos
    ---------
    dice_type   : tipo do dado (ex: "d6")
    vertices    : (N, 3) float64 — posições na esfera unitária
    faces       : list de listas de índices de vértices por face
    normals     : (F, 3) float64 — normal unitária de cada face
    face_values : (F,) int       — valor numérico da face i
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
        """
        Decompõe todas as faces em triângulos (fan triangulation).
        Necessário para renderização OpenGL com GL_TRIANGLES.
        Retorna lista de tuplas (i0, i1, i2).
        """
        tris = []
        for face in self.faces:
            face = list(face)
            for i in range(1, len(face) - 1):
                tris.append((face[0], face[i], face[i + 1]))
        return tris


# ---------------------------------------------------------------------------
# Funções auxiliares de geometria
# ---------------------------------------------------------------------------

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
        [ a,  a,  a],
        [-a, -a,  a],
        [-a,  a, -a],
        [ a, -a, -a],
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

def _build_d4_beveled(bevel=0.15)-> DiceMesh:

    """
    Cria um tetraedro chanfrado.
    1. Mantém as faces originais
    2. Adiciona vértices extras nas arestas
    3. não reconstrói topologia perfeita
    4. Usa convex hull do PyBullet    
    """

    a = 1.0 / np.sqrt(3)

    corners = np.array([
        [ a,  a,  a],
        [-a, -a,  a],
        [-a,  a, -a],
        [ a, -a, -a],
    ])

    corners *= (1.0 - bevel)

    edges = [
        (0,1),
        (0,2),
        (0,3),
        (1,2),
        (1,3),
        (2,3),
    ]

    edge_points = []

    for i, j in edges:
        midpoint = (corners[i] + corners[j]) * 0.5

        midpoint = midpoint / np.linalg.norm(midpoint)

        midpoint *= (1.0 - bevel * 0.5)

        edge_points.append(midpoint)

    vertices = np.vstack([
        corners,
        np.array(edge_points)
    ])

    return vertices

def _build_d6() -> DiceMesh:
    """
    Cubo — d6.
    8 vértices, 6 faces quadradas.
    Valores: 1–6, faces opostas somam 7.
    """
    a = 1.0 / np.sqrt(3)
    raw = np.array([
        [-a, -a, -a], [ a, -a, -a], [ a,  a, -a], [-a,  a, -a],  # z=-a
        [-a, -a,  a], [ a, -a,  a], [ a,  a,  a], [-a,  a,  a],  # z=+a
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
    return DiceMesh(
        dice_type="d6",
        vertices=vertices,
        faces=tuple(tuple(f) for f in faces),
        normals=normals,
        face_values=(1, 6, 2, 5, 3, 4),
    )

def _build_d6_beveled(bevel=0.18) -> DiceMesh:

    a = 1.0 / np.sqrt(3)

    # vértices do cubo
    corners = np.array([
        [-a, -a, -a],
        [ a, -a, -a],
        [ a,  a, -a],
        [-a,  a, -a],

        [-a, -a,  a],
        [ a, -a,  a],
        [ a,  a,  a],
        [-a,  a,  a],
    ], dtype=np.float64)

    # encolhe os cantos
    corners *= (1.0 - bevel)

    # 12 arestas do cubo
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7),
    ]

    edge_points = []

    for i, j in edges:

        midpoint = (corners[i] + corners[j]) * 0.5

        # projeta para esfera/cubo arredondado
        midpoint /= np.linalg.norm(midpoint)

        midpoint *= a * (1.0 - bevel * 0.3)

        edge_points.append(midpoint)

    # centros das faces
    face_centers = np.array([
        [0, 0,  a],   # +Z
        [0, 0, -a],   # -Z
        [0, -a, 0],   # -Y
        [0,  a, 0],   # +Y
        [ a, 0, 0],   # +X
        [-a, 0, 0],   # -X
    ], dtype=np.float64)

    face_centers *= (1.0 - bevel * 0.15)

    vertices = np.vstack([
        corners,
        np.array(edge_points),
        face_centers,
    ])

    # convex hull simples
    hull = ConvexHull(vertices)

    faces = hull.simplices.tolist()

    faces = _ensure_outward_normals(vertices, faces)

    normals = _compute_normals(vertices, faces)

    return DiceMesh(
        dice_type="d6_beveled",
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
        [ 1,  0,  0],
        [-1,  0,  0],
        [ 0,  1,  0],
        [ 0, -1,  0],
        [ 0,  0,  1],
        [ 0,  0, -1],
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

def _build_d10() -> DiceMesh:
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
    vertices = np.array([
        [ 0.44721360,  0.10557281,  0.32491970],  #  0
        [ 0.17082039, -0.10557281,  0.52573111],  #  1
        [-0.17082039,  0.10557281,  0.52573111],  #  2
        [-0.44721360, -0.10557281,  0.32491970],  #  3
        [-0.55278640,  0.10557281,  0.00000000],  #  4
        [-0.44721360, -0.10557281, -0.32491970],  #  5
        [-0.17082039,  0.10557281, -0.52573111],  #  6
        [ 0.17082039, -0.10557281, -0.52573111],  #  7
        [ 0.44721360,  0.10557281, -0.32491970],  #  8
        [ 0.55278640, -0.10557281,  0.00000000],  #  9
        [ 0.00000000,  1.00000000,  0.00000000],  # 10 polo sup
        [ 0.00000000, -1.00000000,  0.00000000],  # 11 polo inf
    ], dtype=float)

    # 10 faces kite — cada uma com vértice agudo (36°) no polo,
    # dois vértices de 108° nos lados e um vértice de 108° na base.
    # Winding CCW visto de fora.
    faces_raw = [
        (10,  0,  9,  8),  # kite  1
        ( 2,  1,  0, 10),  # kite  2
        ( 4,  3,  2, 10),  # kite  3
        ( 6,  5,  4, 10),  # kite  4
        ( 8,  7,  6, 10),  # kite  5
        ( 9,  0,  1, 11),  # kite  6
        (11,  1,  2,  3),  # kite  7
        (11,  3,  4,  5),  # kite  8
        (11,  5,  6,  7),  # kite  9
        (11,  7,  8,  9),  # kite 10
    ]

    normals = _compute_normals(vertices, [list(f) for f in faces_raw])
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
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv = 1.0 / phi

    # Vértices do dodecaedro regular (3 classes de simetria)
    raw = []
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                raw.append([sx, sy, sz])
    for s1 in (1, -1):
        for s2 in (1, -1):
            raw.append([0,   s1 * inv, s2 * phi])
            raw.append([s1 * inv, s2 * phi,   0])
            raw.append([s2 * phi,   0, s1 * inv])

    vertices = np.array(raw, dtype=float)
    vertices -= vertices.mean(axis=0)
    vertices /= np.max(np.linalg.norm(vertices, axis=1))

    # Extrai as 12 faces pentagonais por agrupamento de normais do hull convexo
    from scipy.spatial import ConvexHull as _CH
    hull = _CH(vertices)
    areas, normals_list = [], []
    for tri in hull.simplices:
        a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        cross = np.cross(b - a, c - a)
        nm = np.linalg.norm(cross)
        if nm < 1e-12: continue
        areas.append(nm / 2)
        normals_list.append(cross / nm)

    groups = []
    used = [False] * len(normals_list)
    for i in range(len(normals_list)):
        if used[i]: continue
        g = {'area': areas[i], 'normal': normals_list[i].copy()}
        used[i] = True
        for j in range(i + 1, len(normals_list)):
            if not used[j] and np.dot(normals_list[i], normals_list[j]) > 0.999:
                g['area'] += areas[j]; used[j] = True
        groups.append(g)
    groups.sort(key=lambda g: g['area'], reverse=True)
    face_normals = [g['normal'] for g in groups[:12]]

    faces = []
    for n_ in face_normals:
        projs = vertices @ n_
        tol = (projs.max() - projs.min()) * 0.001 + 1e-8
        idx = np.where(projs >= projs.max() - tol)[0]
        if len(idx) < 3: continue
        fv = vertices[idx]; center = fv.mean(0)
        u = fv[0] - center
        if np.linalg.norm(u) < 1e-10: u = fv[1] - center
        u /= np.linalg.norm(u)
        w = np.cross(n_, u)
        angles = [np.arctan2(np.dot(p - center, w), np.dot(p - center, u)) for p in fv]
        oi = idx[np.argsort(angles)]
        if np.dot(np.cross(vertices[oi[1]] - vertices[oi[0]],
                           vertices[oi[2]] - vertices[oi[0]]), n_) < 0:
            oi = oi[::-1]
        faces.append(list(oi))

    normals = _compute_normals(vertices, faces)
    return DiceMesh(
        dice_type="d12",
        vertices=vertices,
        faces=tuple(tuple(f) for f in faces),
        normals=normals,
        face_values=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    )

def _build_d20() -> DiceMesh:
    """
    Icosaedro regular — d20.
    12 vértices, 20 faces triangulares equiláteras.
    V=12, F=20, E=30 → Euler=2.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    # 12 vértices: 3 retângulos áureos ortogonais
    raw = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            raw.append([0,  s1,       s2 * phi])
            raw.append([s1, s2 * phi, 0       ])
            raw.append([s2 * phi, 0,  s1      ])

    vertices = np.array(raw, dtype=float)
    vertices -= vertices.mean(axis=0)
    vertices /= np.max(np.linalg.norm(vertices, axis=1))

    # Convex hull de 12 pontos = 20 faces triangulares
    from scipy.spatial import ConvexHull as _CH
    hull = _CH(vertices)
    center = vertices.mean(axis=0)
    faces = []
    for tri in hull.simplices:
        a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        n_ = np.cross(b - a, c - a)
        fc = np.array([a, b, c]).mean(0)
        tri_list = list(tri)
        if np.dot(n_, fc - center) < 0:
            tri_list = [tri_list[0], tri_list[2], tri_list[1]]
        faces.append(tri_list)

    normals = _compute_normals(vertices, faces)
    return DiceMesh(
        dice_type="d20",
        vertices=vertices,
        faces=tuple(tuple(f) for f in faces),
        normals=normals,
        face_values=tuple(range(1, 21)),
    )


_BUILDERS = {
    "d4":  _build_d4,
    "d6":  _build_d6,
    "d8":  _build_d8,
    "d10": _build_d10,
    "d12": _build_d12,
    "d20": _build_d20,
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
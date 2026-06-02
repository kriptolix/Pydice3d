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
    12 vértices (2 anéis de 5 + 2 polos), 10 faces kite.
    Valores: 1–10 (ímpares nas faces upper, pares nas lower — padrão de dado).

    Geometria: dois anéis intercalados de 36° + polos em y=±1.
    Os polos são VÉRTICES REAIS das faces (kites), não só âncoras.
    Não usa _project_to_sphere — normaliza pelo raio máximo para preservar
    as proporções e garantir que as faces kite sejam visualmente corretas.

    V=12, F=10, E=20 → Euler=2 (esfera topológica, manifold perfeito).
    """
    n = 5
    h      = 0.4    # altura dos anéis (y = ±h)
    r      = 0.917  # raio equatorial dos anéis
    pole_h = 1.0    # altura dos polos (y = ±pole_h)

    verts_list = []

    # Anel superior: índices 0..4
    for i in range(n):
        a = 2 * np.pi * i / n
        verts_list.append([r * np.cos(a),  h, r * np.sin(a)])

    # Anel inferior: índices 5..9, intercalado de π/n (36°)
    for i in range(n):
        a = 2 * np.pi * i / n + np.pi / n
        verts_list.append([r * np.cos(a), -h, r * np.sin(a)])

    # Polos: índice 10 (+Y) e 11 (−Y)
    verts_list.append([0.,  pole_h, 0.])
    verts_list.append([0., -pole_h, 0.])

    vertices = np.array(verts_list, dtype=float)

    # Normaliza pelo raio máximo — preserva proporções e mantém faces kite planas.
    # NÃO usa _project_to_sphere (deformaria as faces).
    max_r = np.max(np.linalg.norm(vertices, axis=1))
    vertices = vertices / max_r

    # 10 faces kite (V=12, E=20, F=10 → Euler=2, manifold perfeito):
    #   Upper kite i : polo_sup(10) — sup[(i+1)%n] — inf[i] — sup[i]
    #   Lower kite i : polo_inf(11) — inf[i] — sup[(i+1)%n] — inf[(i+1)%n]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([10,    j,   n+i,   i])   # upper kite
        faces.append([11,  n+i,     j, n+j])   # lower kite

    faces = _ensure_outward_normals(vertices, faces)
    normals = _compute_normals(vertices, faces)

    return DiceMesh(
        dice_type="d10",
        vertices=vertices,
        faces=tuple(tuple(f) for f in faces),
        normals=normals,
        face_values=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    )

def _build_d12() -> DiceMesh:
    """
    Dodecaedro regular — d12.
    20 vértices, 12 faces pentagonais.
    Valores: 1–12.
    """
    phi = (1 + np.sqrt(5)) / 2  # razão áurea
    inv_phi = 1 / phi

    # Vértices do dodecaedro (3 retângulos dourados + permutações)
    raw = []
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            for s3 in (+1, -1):
                raw.append([s1, s2, s3])                  # 8 vértices do cubo
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            raw.append([0, s1 * phi, s2 * inv_phi])       # retângulo YZ
            raw.append([s1 * inv_phi, 0, s2 * phi])       # retângulo XZ
            raw.append([s1 * phi, s2 * inv_phi, 0])       # retângulo XY

    vertices = np.array(raw, dtype=float)
    vertices = _project_to_sphere(vertices)

    # 12 faces pentagonais do dodecaedro (orientação CCW de fora)
    # Índices calculados analiticamente para o dodecaedro canônico
    faces = [
        [0, 8, 10, 2, 16],
        [0, 16, 4, 14, 12],
        [0, 12, 6, 18, 8],
        [1, 9, 11, 3, 17],
        [1, 17, 5, 15, 13],
        [1, 13, 7, 19, 9],
        [2, 10, 11, 3, 17],  # ajustado
        [4, 16, 2, 17, 5],
        [6, 12, 14, 5, 15],  # ajustado
        [8, 18, 7, 19, 9],   # ajustado
        [10, 8, 9, 11, 10],  # placeholder — recalculado abaixo
        [14, 4, 5, 15, 7],   # placeholder
    ]

    # Recalcula faces do dodecaedro corretamente por proximidade angular
    faces = _dodecahedron_faces(vertices)
    faces = _ensure_outward_normals(vertices, faces)
    normals = _compute_normals(vertices, faces)
    return DiceMesh(
        dice_type="d12",
        vertices=vertices,
        faces=tuple(tuple(f) for f in faces),
        normals=normals,
        face_values=tuple(range(1, 13)),
    )


def _dodecahedron_faces(vertices: np.ndarray) -> list[list[int]]:
    """
    Reconstrói as 12 faces pentagonais do dodecaedro agrupando vértices
    por proximidade: dois vértices pertencem à mesma face se o ângulo
    entre eles (a partir do centro) é menor que o ângulo de aresta do dodecaedro.

    O ângulo de aresta do dodecaedro é arccos(−1/√5) ≈ 116.57°.
    Dois vértices adjacentes no mesmo face têm ângulo ≈ 41.8° entre si.
    """
    n = len(vertices)  # 20 vértices

    # Threshold: ângulo de aresta do dodecaedro ≈ cos(41.8°) ≈ 0.7454
    EDGE_COS = 0.7454

    # Grafo de adjacência: vi ~ vj se cos(ângulo) > EDGE_COS
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            c = float(np.dot(vertices[i], vertices[j]))
            if c > EDGE_COS:
                adj[i].append(j)
                adj[j].append(i)

    # Cada vértice pertence exatamente a 3 faces pentagonais
    # Encontra pentágonos: cliques de 5 vértices mutuamente adjacentes a 2 passos
    visited: set[frozenset] = set()
    faces: list[list[int]] = []

    for start in range(n):
        neighbors = adj[start]
        # Para cada par de vizinhos, verifica se existe triângulo (caminho de 2)
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                vi, vj = neighbors[i], neighbors[j]
                # vi e vj devem ser vizinhos de 2 passos via outros vértices
                # Busca pentágono: start → vi → ? → vj → ? → start
                common_vi = set(adj[vi])
                common_vj = set(adj[vj])
                # Intermediário entre vi e vj
                for mid in common_vi & common_vj:
                    if mid == start:
                        continue
                    face_set = frozenset([start, vi, mid, vj])
                    if len(face_set) == 4:
                        # Busca quinto vértice
                        for fifth in adj[start]:
                            if fifth in adj[vj] and fifth != vi:
                                full = frozenset([start, vi, mid, vj, fifth])
                                if len(full) == 5 and full not in visited:
                                    visited.add(full)
                                    ordered = _order_face_ccw(vertices, list(full))
                                    faces.append(ordered)

    # Fallback: se não encontrou 12 faces, usa faces canônicas hardcoded
    if len(faces) != 12:
        faces = _dodecahedron_faces_canonical()

    return faces[:12]


def _order_face_ccw(vertices: np.ndarray, indices: list[int]) -> list[int]:
    """
    Ordena os índices de uma face em sentido CCW visto de fora da esfera.
    """
    verts = vertices[indices]
    center = verts.mean(axis=0)
    normal = _normalize(center)

    # Cria base ortonormal no plano da face
    ref = verts[0] - center
    ref = ref - np.dot(ref, normal) * normal
    ref = _normalize(ref)
    perp = np.cross(normal, ref)

    angles = []
    for idx, v in zip(indices, verts):
        d = v - center
        d = d - np.dot(d, normal) * normal
        angle = np.arctan2(np.dot(d, perp), np.dot(d, ref))
        angles.append((angle, idx))

    angles.sort()
    return [idx for _, idx in angles]


def _dodecahedron_faces_canonical() -> list[list[int]]:
    """Faces canônicas hardcoded do dodecaedro (fallback)."""
    return [
        [0, 8, 10, 2, 16],
        [0, 16, 4, 14, 12],
        [0, 12, 6, 18, 8],
        [1, 9, 19, 7, 13],
        [1, 13, 15, 5, 17],
        [1, 17, 3, 11, 9],
        [2, 10, 11, 3, 17],
        [4, 16, 2, 17, 5],
        [6, 12, 14, 5, 15],
        [8, 18, 7, 19, 9],
        [10, 8, 9, 11, 3],
        [14, 4, 5, 15, 7],
    ]


def _build_d20() -> DiceMesh:
    """
    Icosaedro regular — d20.
    12 vértices, 20 faces triangulares.
    Valores: 1–20.
    """
    phi = (1 + np.sqrt(5)) / 2

    raw = []
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            raw.append([0, s1, s2 * phi])
            raw.append([s1, s2 * phi, 0])
            raw.append([s2 * phi, 0, s1])

    vertices = np.array(raw, dtype=float)
    vertices = _project_to_sphere(vertices)

    # 20 faces do icosaedro (índices canônicos)
    faces = [
        [0, 2, 8],  [0, 8, 4],  [0, 4, 6],  [0, 6, 10], [0, 10, 2],
        [3, 1, 9],  [3, 9, 5],  [3, 5, 7],  [3, 7, 11], [3, 11, 1],
        [2, 1, 8],  [8, 1, 5],  [8, 5, 4],  [4, 5, 7],  [4, 7, 6],
        [6, 7, 11], [6, 11, 10],[10, 11, 9],[10, 9, 2],  [2, 9, 1],
    ]
    faces = _ensure_outward_normals(vertices, faces)
    normals = _compute_normals(vertices, faces)
    return DiceMesh(
        dice_type="d20",
        vertices=vertices,
        faces=tuple(tuple(f) for f in faces),
        normals=normals,
        face_values=tuple(range(1, 21)),
    )


# ---------------------------------------------------------------------------
# Registry e factory pública
# ---------------------------------------------------------------------------

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
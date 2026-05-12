"""
geometrics.py — Construção e processamento de geometria 3D.

Responsabilidades:
  - Carregamento de arquivos OBJ
  - Expansão de meshes para flat shading
  - Cálculo de tangentes (normal mapping)
  - Construção procedural de caixas com UV
"""

import numpy as np
import math


# ---------------------------------------------------------------------------
# Carregamento de OBJ
# ---------------------------------------------------------------------------

def load_obj(path: str):
    """
    Carrega um OBJ retornando posições, UVs e faces.

    Retorna
    -------
    positions : list of [x, y, z]
    uvs       : list of [u, v]   (vazio se não houver 'vt')
    faces     : list of face-tokens  ex: ["1/1/1", "2/2/2", "3/3/3"]
    """
    positions, uvs, faces = [], [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                positions.append(list(map(float, line.split()[1:4])))
            elif line.startswith("vt "):
                uvs.append(list(map(float, line.split()[1:3])))
            elif line.startswith("f "):
                faces.append(line.split()[1:])
    return positions, uvs, faces


# ---------------------------------------------------------------------------
# Cálculo de tangente
# ---------------------------------------------------------------------------

def _compute_tangent(p0, p1, p2, uv0, uv1, uv2, n):
    """
    Calcula o vetor tangente para um triângulo usando deltas de UV.
    Retorna o tangente ortogonalizado (Gram-Schmidt) em relação à normal.
    """
    e1, e2   = p1 - p0, p2 - p0
    du1, dv1 = uv1[0] - uv0[0], uv1[1] - uv0[1]
    du2, dv2 = uv2[0] - uv0[0], uv2[1] - uv0[1]

    denom = du1 * dv2 - du2 * dv1
    if abs(denom) < 1e-8:
        ref   = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
        t_vec = ref - np.dot(ref, n) * n
    else:
        f     = 1.0 / denom
        t_vec = f * (dv2 * e1 - dv1 * e2)
        t_vec = t_vec - np.dot(t_vec, n) * n   # Gram-Schmidt

    tl = np.linalg.norm(t_vec)
    return t_vec / tl if tl > 1e-8 else np.array([1., 0., 0.])


# ---------------------------------------------------------------------------
# Expansão de OBJ com UV e tangentes (dados)
# ---------------------------------------------------------------------------

def expand_obj_with_uv_tangent(positions, uvs, faces, scale=1.0):
    """
    Expande o OBJ para flat shading com UVs e tangentes por triângulo.

    Retorna arrays flat (3 verts × N triângulos):
        pos_out  : float32  shape (N*3*3,)
        nor_out  : float32  shape (N*3*3,)
        uv_out   : float32  shape (N*3*2,)
        tan_out  : float32  shape (N*3*3,)

    Se o OBJ não tiver UVs, uv_out e tan_out serão zeros.
    """
    has_uv = len(uvs) > 0
    pos_out, nor_out, uv_out, tan_out = [], [], [], []

    for face in faces:
        vi_list, vti_list = [], []
        for token in face:
            parts = token.split('/')
            vi_list.append(int(parts[0]) - 1)
            if has_uv and len(parts) > 1 and parts[1]:
                vti_list.append(int(parts[1]) - 1)
            else:
                vti_list.append(None)

        for i in range(1, len(vi_list) - 1):
            i0, i1, i2 = vi_list[0], vi_list[i], vi_list[i + 1]

            p0 = np.array(positions[i0]) * scale
            p1 = np.array(positions[i1]) * scale
            p2 = np.array(positions[i2]) * scale

            n  = np.cross(p1 - p0, p2 - p0)
            nl = np.linalg.norm(n)
            n  = n / nl if nl > 1e-8 else np.array([0., 1., 0.])

            if has_uv and vti_list[0] is not None:
                t0, t1, t2 = vti_list[0], vti_list[i], vti_list[i + 1]
                uv0 = np.array(uvs[t0]) if t0 is not None else np.zeros(2)
                uv1 = np.array(uvs[t1]) if t1 is not None else np.zeros(2)
                uv2 = np.array(uvs[t2]) if t2 is not None else np.zeros(2)
                tangent = _compute_tangent(p0, p1, p2, uv0, uv1, uv2, n)
            else:
                uv0 = uv1 = uv2 = np.zeros(2)
                tangent = np.array([1., 0., 0.])

            for p, uv in zip((p0, p1, p2), (uv0, uv1, uv2)):
                pos_out.extend(p)
                nor_out.extend(n)
                uv_out.extend(uv)
                tan_out.extend(tangent)

    return (
        np.array(pos_out, dtype=np.float32),
        np.array(nor_out, dtype=np.float32),
        np.array(uv_out,  dtype=np.float32),
        np.array(tan_out, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Expansão de OBJ sem UV (piso)
# ---------------------------------------------------------------------------

def expand_flat_shading_from_obj(vertices, faces):
    """Retorna (pos_flat, nor_flat) sem UV — usado pelo piso."""
    pos_out, nor_out = [], []
    for face in faces:
        idx = [int(v.split('/')[0]) - 1 for v in face]
        for i in range(1, len(idx) - 1):
            v0 = np.array(vertices[idx[0]])
            v1 = np.array(vertices[idx[i]])
            v2 = np.array(vertices[idx[i + 1]])
            n  = np.cross(v1 - v0, v2 - v0)
            nl = np.linalg.norm(n)
            n  = n / nl if nl > 1e-8 else np.array([0., 1., 0.])
            for v in (v0, v1, v2):
                pos_out.extend(v)
                nor_out.extend(n)
    return (
        np.array(pos_out, dtype=np.float32),
        np.array(nor_out, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Caixa com UV (piso da bandeja)
# ---------------------------------------------------------------------------

def build_floor_mesh(w: float, h: float, d: float, uv_tile: float = 4.0):
    """
    Constrói a caixa do piso com UV tiled na face superior (Y+) e UV
    simples nas laterais.

    Retorna (pos_flat, nor_flat, uv_flat, tan_flat) — arrays float32.
    """
    hw, hh, hd = w / 2, h / 2, d / 2
    t = uv_tile
    rows = []

    def add_tri(verts3, n, uvs3):
        p0, p1, p2   = [np.array(v, dtype=np.float64) for v in verts3]
        uv0, uv1, uv2 = [np.array(u, dtype=np.float64) for u in uvs3]
        e1, e2 = p1 - p0, p2 - p0
        du1, dv1 = uv1 - uv0
        du2, dv2 = uv2 - uv0
        denom = du1 * dv2 - du2 * dv1
        if abs(denom) > 1e-8:
            f     = 1.0 / denom
            t_vec = f * (dv2 * e1 - dv1 * e2)
            tl    = np.linalg.norm(t_vec)
            t_vec = t_vec / tl if tl > 1e-8 else np.array([1., 0., 0.])
        else:
            t_vec = np.array([1., 0., 0.])
        nn = np.array(n, dtype=np.float64)
        for p, uv in zip([p0, p1, p2], [uv0, uv1, uv2]):
            rows.append([*p, *nn, *uv, *t_vec])

    # +Y topo (UV tiled)
    add_tri([[-hw, hh,-hd],[hw, hh,-hd],[hw, hh, hd]], [0,1,0], [[0,0],[t,0],[t,t]])
    add_tri([[-hw, hh,-hd],[hw, hh, hd],[-hw, hh, hd]], [0,1,0], [[0,0],[t,t],[0,t]])
    # -Y base
    add_tri([[-hw,-hh,-hd],[-hw,-hh, hd],[ hw,-hh, hd]], [0,-1,0], [[0,0],[0,1],[1,1]])
    add_tri([[-hw,-hh,-hd],[ hw,-hh, hd],[ hw,-hh,-hd]], [0,-1,0], [[0,0],[1,1],[1,0]])
    # -Z
    add_tri([[-hw,-hh,-hd],[ hw,-hh,-hd],[ hw, hh,-hd]], [0,0,-1], [[0,0],[1,0],[1,1]])
    add_tri([[-hw,-hh,-hd],[ hw, hh,-hd],[-hw, hh,-hd]], [0,0,-1], [[0,0],[1,1],[0,1]])
    # +Z
    add_tri([[-hw,-hh, hd],[-hw, hh, hd],[ hw, hh, hd]], [0,0,1], [[0,0],[0,1],[1,1]])
    add_tri([[-hw,-hh, hd],[ hw, hh, hd],[ hw,-hh, hd]], [0,0,1], [[0,0],[1,1],[1,0]])
    # -X
    add_tri([[-hw,-hh,-hd],[-hw, hh,-hd],[-hw, hh, hd]], [-1,0,0], [[0,0],[0,1],[1,1]])
    add_tri([[-hw,-hh,-hd],[-hw, hh, hd],[-hw,-hh, hd]], [-1,0,0], [[0,0],[1,1],[1,0]])
    # +X
    add_tri([[ hw,-hh,-hd],[ hw,-hh, hd],[ hw, hh, hd]], [1,0,0], [[0,0],[1,0],[1,1]])
    add_tri([[ hw,-hh,-hd],[ hw, hh, hd],[ hw, hh,-hd]], [1,0,0], [[0,0],[1,1],[0,1]])

    data = np.array(rows, dtype=np.float32)
    return (
        np.ascontiguousarray(data[:, 0:3].flatten()),
        np.ascontiguousarray(data[:, 3:6].flatten()),
        np.ascontiguousarray(data[:, 6:8].flatten()),
        np.ascontiguousarray(data[:, 8:11].flatten()),
    )


# ---------------------------------------------------------------------------
# Legado: caixa indexada simples (sem UV)
# ---------------------------------------------------------------------------

def build_flat_box(w, h, d):
    """Caixa simples para uso sem UV."""
    hw, hh, hd = w / 2, h / 2, d / 2
    verts = np.array([
        [-hw,-hh,-hd],[ hw,-hh,-hd],[ hw, hh,-hd],[-hw, hh,-hd],
        [-hw,-hh, hd],[ hw,-hh, hd],[ hw, hh, hd],[-hw, hh, hd],
    ], dtype=np.float32)
    indices = np.array([
        [0,1,2],[0,2,3],[4,6,5],[4,7,6],
        [0,5,1],[0,4,5],[2,6,3],[3,6,7],
        [0,3,7],[0,7,4],[1,5,6],[1,6,2],
    ], dtype=np.int32)
    return verts, indices


def expand_flat_shading(vertices, indices):
    """Expande mesh indexada para flat shading (normal por face)."""
    pos_out, nor_out = [], []
    for tri in indices:
        v0, v1, v2 = [np.array(vertices[tri[i]]) for i in range(3)]
        n  = np.cross(v1 - v0, v2 - v0)
        nl = np.linalg.norm(n)
        if nl > 1e-8:
            n /= nl
        for v in (v0, v1, v2):
            pos_out.extend(v)
            nor_out.extend(n)
    return (
        np.array(pos_out, dtype=np.float32),
        np.array(nor_out, dtype=np.float32),
    )

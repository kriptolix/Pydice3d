import numpy as np
import math

def build_flat_box(w, h, d):
    """Caixa simples para a superfície visível."""
    hw, hh, hd = w / 2, h / 2, d / 2
    verts = np.array([
        [-hw, -hh, -hd], [ hw, -hh, -hd], [ hw,  hh, -hd], [-hw,  hh, -hd],
        [-hw, -hh,  hd], [ hw, -hh,  hd], [ hw,  hh,  hd], [-hw,  hh,  hd],
    ], dtype=np.float32)
    indices = np.array([
        [0,1,2],[0,2,3],
        [4,6,5],[4,7,6],
        [0,5,1],[0,4,5],
        [2,6,3],[3,6,7],
        [0,3,7],[0,7,4],
        [1,5,6],[1,6,2],
    ], dtype=np.int32)
    return verts, indices


def expand_flat_shading(vertices, indices):
    """Expande indexed mesh para flat shading (normal por face)."""
    pos_out, nor_out = [], []
    for tri in indices:
        v0 = np.array(vertices[tri[0]])
        v1 = np.array(vertices[tri[1]])
        v2 = np.array(vertices[tri[2]])
        n = np.cross(v1 - v0, v2 - v0)
        nl = np.linalg.norm(n)
        if nl > 1e-8:
            n /= nl
        for v in [v0, v1, v2]:
            pos_out.extend(v)
            nor_out.extend(n)
    return (np.array(pos_out, dtype=np.float32),
            np.array(nor_out, dtype=np.float32))


def load_obj(path):
    """
    Carrega um OBJ retornando posições, UVs e faces.

    Retorna:
        positions : list of [x, y, z]
        uvs       : list of [u, v]   (lista vazia se o OBJ não tiver 'vt')
        faces     : list of face-tokens  ex: ["1/1/1", "2/2/2", "3/3/3"]
    """
    positions = []
    uvs       = []
    faces     = []

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


def _compute_tangent(p0, p1, p2, uv0, uv1, uv2, n):
    """
    Calcula o vetor tangente para um triângulo usando as deltas de UV.
    Retorna o tangente ortogonalizado (Gram-Schmidt) em relação à normal.
    """
    e1  = p1 - p0
    e2  = p2 - p0
    du1 = uv1[0] - uv0[0]
    dv1 = uv1[1] - uv0[1]
    du2 = uv2[0] - uv0[0]
    dv2 = uv2[1] - uv0[1]

    denom = du1 * dv2 - du2 * dv1
    if abs(denom) < 1e-8:
        # UV degenerada — escolhe tangente arbitrária perpendicular à normal
        ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        t = ref - np.dot(ref, n) * n
    else:
        f = 1.0 / denom
        t = f * (dv2 * e1 - dv1 * e2)
        # Gram-Schmidt: ortogonaliza t em relação à normal
        t = t - np.dot(t, n) * n

    tl = np.linalg.norm(t)
    return t / tl if tl > 1e-8 else np.array([1.0, 0.0, 0.0])


def expand_obj_with_uv_tangent(positions, uvs, faces, scale=1.0):
    """
    Expande o OBJ para flat shading com UVs e tangentes por triângulo.

    Retorna arrays flat (3 verts × N triângulos):
        pos_out  : float32  shape (N*3*3,)   — posições XYZ
        nor_out  : float32  shape (N*3*3,)   — normais de face
        uv_out   : float32  shape (N*3*2,)   — coordenadas UV
        tan_out  : float32  shape (N*3*3,)   — tangentes

    Se o OBJ não tiver UVs (uvs vazio), uv_out e tan_out serão zeros.
    """
    has_uv = len(uvs) > 0

    pos_out = []
    nor_out = []
    uv_out  = []
    tan_out = []

    for face in faces:
        # Parseia tokens "vi/vti/vni" — aceita vi, vi/vti, vi//vni, vi/vti/vni
        vi_list  = []
        vti_list = []
        for token in face:
            parts = token.split('/')
            vi_list.append(int(parts[0]) - 1)
            if has_uv and len(parts) > 1 and parts[1]:
                vti_list.append(int(parts[1]) - 1)
            else:
                vti_list.append(None)

        # Triangulação em fan
        for i in range(1, len(vi_list) - 1):
            i0, i1, i2 = vi_list[0], vi_list[i], vi_list[i + 1]

            p0 = np.array(positions[i0]) * scale
            p1 = np.array(positions[i1]) * scale
            p2 = np.array(positions[i2]) * scale

            # Normal de face
            n = np.cross(p1 - p0, p2 - p0)
            nl = np.linalg.norm(n)
            n = n / nl if nl > 1e-8 else np.array([0.0, 1.0, 0.0])

            # UVs
            if has_uv and vti_list[0] is not None:
                t0_idx, t1_idx, t2_idx = vti_list[0], vti_list[i], vti_list[i + 1]
                uv0 = np.array(uvs[t0_idx]) if t0_idx is not None else np.zeros(2)
                uv1 = np.array(uvs[t1_idx]) if t1_idx is not None else np.zeros(2)
                uv2 = np.array(uvs[t2_idx]) if t2_idx is not None else np.zeros(2)
                tangent = _compute_tangent(p0, p1, p2, uv0, uv1, uv2, n)
            else:
                uv0 = uv1 = uv2 = np.zeros(2)
                tangent = np.array([1.0, 0.0, 0.0])

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
# Funções legadas mantidas para compatibilidade com o piso (sem UV)
# ---------------------------------------------------------------------------

def expand_flat_shading_from_obj(vertices, faces):
    """Compatibilidade: retorna só pos+nor sem UV (usado pelo piso)."""
    pos_out = []
    nor_out = []

    for face in faces:
        idx = []
        for v in face:
            vi = int(v.split('/')[0]) - 1
            idx.append(vi)

        for i in range(1, len(idx) - 1):
            i0, i1, i2 = idx[0], idx[i], idx[i + 1]
            v0 = np.array(vertices[i0])
            v1 = np.array(vertices[i1])
            v2 = np.array(vertices[i2])

            n = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(n)
            n = n / norm if norm > 1e-8 else np.array([0.0, 1.0, 0.0])

            for v in (v0, v1, v2):
                pos_out.extend(v)
                nor_out.extend(n)

    return (
        np.array(pos_out, dtype=np.float32),
        np.array(nor_out, dtype=np.float32),
    )
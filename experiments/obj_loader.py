"""
obj_loader.py – Loading .obj meshes into DiceMesh

>> This file is not currently in use; it anticipates a feature that will be added in the future.

Loads Wavefront OBJ files preserving texture UVs (vt) and smooth normals (vn) per vertex — essential for PBR textures with bevels.

Scale Normalization
──────────────────────
Vertices are scaled UNIFORMLY so that the solid fits within the unit sphere (division by the maximum of the norms). This is different from projecting each vertex onto the sphere — the original shape and angles of the OBJ are preserved. Chamfers and rounded edges remain correct.

When the .obj file contains UVs (vt) and normals (vn):

- DiceMesh.has_obj_data = True

- DiceMesh.tri_* contains the complete vertex buffer ready for GPU

- PBR pipeline uses the UVs from the OBJ; glyph atlas disabled

When UVs/vn are missing or an error occurs:

- Returns None → fallback to procedural builder + SDF pipeline

Group naming convention for face values:

g face_1 → value 1
g face1 → value 1
# face: 6 → value 6

If not specified: faces are assigned 1..N in order of appearance.
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Localização dos assets
# ---------------------------------------------------------------------------

def _default_assets_root() -> Path:
    env = os.environ.get("PYDICE3D_ASSETS")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    try:
        from importlib.resources import files
        pkg_path = Path(str(files("pydice3d").joinpath("assets")))
        if pkg_path.is_dir():
            return pkg_path
    except Exception:
        pass
    return Path("assets")


ASSETS_ROOT: Path = _default_assets_root()


def obj_path_for(dice_type: str, assets_root: Optional[Path] = None) -> Path:
    root = assets_root or ASSETS_ROOT
    return root / dice_type / f"{dice_type}.obj"


# ---------------------------------------------------------------------------
# Rotações de reorientação (Blender Z-up → Y-up)
# ---------------------------------------------------------------------------

def _rot_x(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=np.float64)


# Rotação aplicada após carregamento para converter Z-up (Blender) → Y-up.
# Ajuste ou remova entradas se o seu exportador usar convenção diferente.
_REORIENT: dict[str, np.ndarray] = {
    "d4":  _rot_x(-90.0),
    "d6":  _rot_x(-90.0),
    "d8":  _rot_x(-90.0),
    "d10": _rot_x(-90.0),
    "d12": _rot_x(-90.0),
    "d20": _rot_x(-90.0),
}


# ---------------------------------------------------------------------------
# Parser OBJ completo (v, vt, vn, f, g/o, comentários de valor)
# ---------------------------------------------------------------------------

_RE_FACE_COMMENT = re.compile(r"#\s*face[:\s]+(\d+)", re.IGNORECASE)
_RE_FACE_IN_NAME = re.compile(r"(\d+)")


def _parse_face_value_from_name(name: str) -> Optional[int]:
    m = _RE_FACE_IN_NAME.search(name)
    return int(m.group(1)) if m else None


def _parse_obj_full(
    text: str,
) -> tuple[list, list, list, list]:
    """
    Parseia OBJ e retorna estruturas brutas:
      positions     : list[(x, y, z)]
      tex_coords    : list[(u, v)]
      vert_normals  : list[(nx, ny, nz)]
      raw_tris      : list[( corner0, corner1, corner2, face_value )]
                      corner = (vi, ti, ni) — índices 0-based, -1 se ausente

    Faces poligonais são triangulizadas com fan triangulation.
    """
    positions:    list = []
    tex_coords:   list = []
    vert_normals: list = []
    raw_tris:     list = []

    current_val: Optional[int] = None

    for raw in text.splitlines():
        line = raw.strip()

        # Comentário com valor de face explícito
        m = _RE_FACE_COMMENT.match(line)
        if m:
            current_val = int(m.group(1))
            continue

        if not line or line.startswith("#"):
            continue

        tok = line.split()
        if not tok:
            continue
        cmd = tok[0]

        if cmd == "v":
            try:
                positions.append(tuple(float(x) for x in tok[1:4]))
            except ValueError:
                pass

        elif cmd == "vt":
            try:
                u = float(tok[1])
                v = float(tok[2]) if len(tok) > 2 else 0.0
                tex_coords.append((u, v))
            except ValueError:
                pass

        elif cmd == "vn":
            try:
                vert_normals.append(tuple(float(x) for x in tok[1:4]))
            except ValueError:
                pass

        elif cmd in ("g", "o"):
            name = tok[1] if len(tok) > 1 else ""
            current_val = _parse_face_value_from_name(name)

        elif cmd == "f":
            corners = []
            for t in tok[1:]:
                parts = t.split("/")
                try:
                    vi = int(parts[0])
                    ti = int(parts[1]) if len(parts) > 1 and parts[1] else 0
                    ni = int(parts[2]) if len(parts) > 2 and parts[2] else 0
                except ValueError:
                    continue
                # OBJ 1-based → 0-based; negativo = relativo ao fim
                vi = (vi - 1) if vi > 0 else (len(positions)    + vi)
                ti = (ti - 1) if ti > 0 else (len(tex_coords)   + ti) if ti != 0 else -1
                ni = (ni - 1) if ni > 0 else (len(vert_normals) + ni) if ni != 0 else -1
                corners.append((vi, ti, ni))

            if len(corners) < 3:
                continue
            # Fan triangulation
            for i in range(1, len(corners) - 1):
                raw_tris.append((corners[0], corners[i], corners[i + 1], current_val))

    return positions, tex_coords, vert_normals, raw_tris


# ---------------------------------------------------------------------------
# Escalonamento uniforme (preserva forma)
# ---------------------------------------------------------------------------

def _scale_uniform(pos_arr: np.ndarray) -> np.ndarray:
    """
    Escala UNIFORMEMENTE todos os vértices para que o sólido caiba na
    esfera unitária (raio máximo = 1).

    *** Diferente de normalizar cada vértice individualmente ***
    Normalização individual projeta tudo na esfera → destrói chanfros.
    Escala uniforme preserva ângulos, proporções e chanfros.
    """
    max_r = np.linalg.norm(pos_arr, axis=1).max()
    if max_r < 1e-12:
        return pos_arr
    return pos_arr / max_r


# ---------------------------------------------------------------------------
# Cálculo de face_idx e valores lógicos por triângulo
# ---------------------------------------------------------------------------

def _compute_face_idx_per_tri(
    raw_tris: list,
) -> tuple[np.ndarray, list[int]]:
    """
    Agrupa triângulos por face_value (ordem de aparição) e atribui
    índice contíguo 0..F-1.

    Retorna:
      face_idx_per_tri : (T,) int32
      face_values      : list[int]
    """
    seen: dict[Optional[int], int] = {}
    for _, _, _, fval in raw_tris:
        if fval not in seen:
            seen[fval] = len(seen)

    face_values = [
        fval if fval is not None else (i + 1)
        for fval, i in sorted(seen.items(), key=lambda x: x[1])
    ]
    val_to_idx = {fval: i for i, (fval, i) in
                  zip(range(len(seen)), sorted(seen.items(), key=lambda x: x[1]))}

    face_idx_per_tri = np.array(
        [val_to_idx[fval] for _, _, _, fval in raw_tris],
        dtype=np.int32,
    )
    return face_idx_per_tri, face_values


# ---------------------------------------------------------------------------
# Faces lógicas para física (convex hull PyBullet)
# ---------------------------------------------------------------------------

def _build_logical_faces(
    pos_arr:          np.ndarray,
    raw_tris:         list,
    face_idx_per_tri: np.ndarray,
    n_logical:        int,
) -> list[list[int]]:
    """Agrupa índices de posição únicos por face lógica."""
    face_verts: list[set] = [set() for _ in range(n_logical)]
    for ti, (c0, c1, c2, _) in enumerate(raw_tris):
        fi = int(face_idx_per_tri[ti])
        for vi, _, _ in (c0, c1, c2):
            face_verts[fi].add(vi)
    return [sorted(s) for s in face_verts]


# ---------------------------------------------------------------------------
# Montagem do vertex buffer GPU
# ---------------------------------------------------------------------------

def _build_gpu_buffers(
    pos_arr:          np.ndarray,         # (V, 3) float64 já rotacionado + escalado
    tex_coords:       list,               # list[(u, v)]
    vert_normals_rot: np.ndarray,         # (VN, 3) float64 já rotacionado
    raw_tris:         list,               # triangulos brutos
    face_idx_per_tri: np.ndarray,         # (T,) int32
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Monta tri_positions, tri_normals, tri_uvs, tri_face_idx para upload GPU.
    Layout: (T*3, _).  V flipado para convenção OpenGL.
    """
    n      = len(raw_tris)
    n_vert = n * 3
    tri_pos = np.zeros((n_vert, 3), dtype=np.float32)
    tri_nrm = np.zeros((n_vert, 3), dtype=np.float32)
    tri_uv  = np.zeros((n_vert, 2), dtype=np.float32)

    has_uv = len(tex_coords) > 0
    has_vn = len(vert_normals_rot) > 0

    tc_arr = np.array(tex_coords, dtype=np.float32) if has_uv else None

    for tri_idx, (c0, c1, c2, _) in enumerate(raw_tris):
        for k, (vi, ti, ni) in enumerate((c0, c1, c2)):
            row = tri_idx * 3 + k

            # Posição (já escalada e rotacionada)
            if 0 <= vi < len(pos_arr):
                tri_pos[row] = pos_arr[vi].astype(np.float32)

            # Normal suave por vértice
            if has_vn and 0 <= ni < len(vert_normals_rot):
                nv = vert_normals_rot[ni]
                nl = np.linalg.norm(nv)
                tri_nrm[row] = (nv / nl).astype(np.float32) if nl > 1e-9 else np.float32([0, 1, 0])
            else:
                tri_nrm[row] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

            # UV — flip V (OBJ: V=0 topo; OpenGL: V=0 base)
            if has_uv and tc_arr is not None and 0 <= ti < len(tc_arr):
                tri_uv[row, 0] = tc_arr[ti, 0]
                tri_uv[row, 1] = 1.0 - tc_arr[ti, 1]

    tri_fi = np.repeat(face_idx_per_tri, 3).astype(np.float32)
    return tri_pos, tri_nrm, tri_uv, tri_fi


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def load_obj_mesh(
    dice_type:   str,
    assets_root: Optional[Path] = None,
) -> "Optional[DiceMesh]":
    """
    Carrega malha 3D de um .obj com UV e normais por vértice.

    Retorna DiceMesh com has_obj_data=True se sucesso, ou None em falha.
    None → chamador usa builder procedural + pipeline SDF.

    Escala: uniforme (preserva chanfros), não projeção na esfera.
    """
    from pydice3d.dice_mesh import DiceMesh, _compute_normals, _ensure_outward_normals

    path = obj_path_for(dice_type, assets_root)
    if not path.exists():
        logger.debug("OBJ não encontrado: %s", path)
        return None

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Não foi possível ler %s: %s", path, exc)
        return None

    try:
        positions, tex_coords, vert_normals, raw_tris = _parse_obj_full(text)
    except Exception as exc:
        logger.warning("Falha ao parsear %s: %s", path, exc)
        return None

    if not positions or not raw_tris:
        logger.warning("%s: OBJ vazio (pos=%d tris=%d)", path, len(positions), len(raw_tris))
        return None

    has_uv = len(tex_coords) > 0
    has_vn = len(vert_normals) > 0

    # ── 1. Rotação de reorientação (Blender Z-up → Y-up) ────────────────────
    R       = _REORIENT.get(dice_type)
    pos_arr = np.array(positions, dtype=np.float64)
    if R is not None:
        pos_arr = (R @ pos_arr.T).T

    vn_arr = np.array(vert_normals, dtype=np.float64) if has_vn else np.zeros((0, 3))
    if R is not None and has_vn:
        vn_arr = (R @ vn_arr.T).T

    # ── 2. Escala uniforme — preserva forma e chanfros ───────────────────────
    #
    # NÃO usamos normalização por vértice (_project_to_sphere) aqui.
    # Dividir cada vértice pela sua própria norma projeta todos na esfera
    # unitária, destruindo chanfros e arestas arredondadas.
    # Dividimos pelo MÁXIMO das normas: o sólido fica inscrito na esfera
    # unitária com sua geometria original preservada.
    pos_arr = _scale_uniform(pos_arr)

    # ── 3. Faces lógicas para física / detecção de face superior ────────────
    face_idx_per_tri, face_values_list = _compute_face_idx_per_tri(raw_tris)
    n_logical    = len(face_values_list)
    logical_faces = _build_logical_faces(pos_arr, raw_tris, face_idx_per_tri, n_logical)

    try:
        logical_faces = _ensure_outward_normals(pos_arr, logical_faces)
    except Exception:
        pass
    face_normals = _compute_normals(pos_arr, logical_faces)

    # ── 4. Vertex buffer para GPU ────────────────────────────────────────────
    tri_pos, tri_nrm, tri_uv, tri_fi = _build_gpu_buffers(
        pos_arr, tex_coords, vn_arr, raw_tris, face_idx_per_tri,
    )

    logger.info(
        "OBJ carregado: %s  verts=%d  tris=%d  faces=%d  uv=%s  vn=%s",
        dice_type, len(pos_arr), len(raw_tris), n_logical, has_uv, has_vn,
    )

    return DiceMesh(
        dice_type=dice_type,
        vertices=pos_arr,
        faces=tuple(tuple(f) for f in logical_faces),
        normals=face_normals,
        face_values=tuple(face_values_list),
        has_obj_data=(has_uv and has_vn),
        tri_positions=tri_pos,
        tri_normals=tri_nrm,
        tri_uvs=tri_uv if has_uv else None,
        tri_face_idx=tri_fi,
    )
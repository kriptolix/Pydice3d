"""
render_data.py – Dados de Render CPU
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydice3d.shaders import (
    GLYPH_NONE, GLYPH_PLUS, GLYPH_MINUS, GLYPH_BLANK, MAX_FACES,
)

if TYPE_CHECKING:
    from pydice3d.dice_state import DiceState
from typing import NamedTuple


class DiceTheme(NamedTuple):
    dice_color:  tuple[float, float, float]
    glyph_color: tuple[float, float, float]


DICE_THEMES: dict[str, DiceTheme] = {
    "dark":  DiceTheme(dice_color=(0.15, 0.15, 0.15), glyph_color=(0.95, 0.95, 0.95)),
    "light": DiceTheme(dice_color=(0.95, 0.95, 0.95), glyph_color=(0.15, 0.15, 0.15)),
}

DEFAULT_DICE_COLOR: tuple[float, float, float] = (0.7, 0.7, 0.7)

FUDGE_GLYPHS: tuple[int, ...] = (GLYPH_PLUS, GLYPH_PLUS,
                                 GLYPH_MINUS, GLYPH_MINUS,
                                 GLYPH_BLANK, GLYPH_BLANK)


def glyph_d100(tens: int) -> int:
    return 21 + ((tens % 100) // 10)


def build_face_glyphs(dice_type: str, face_values: list[int]) -> list[int]:

    if dice_type == "df":
        return list(FUDGE_GLYPHS)

    glyphs = []

    for v in face_values:
        if dice_type == "d100":
            glyphs.append(glyph_d100(v))

        elif dice_type == "d10":
            glyphs.append(0 if int(v) == 10 else int(v))

        else:
            glyphs.append(int(v) if 0 <= int(v) <= 20 else GLYPH_NONE)

    return glyphs


# Larger = smaller glyph on the face
GLYPH_UV_SCALE: dict[str, float] = {
    "d4":     0.00,    # see in _D4_GLYPH_SCALE
    "d6":     1.20,
    "d8":     2.00,
    "d10":    2.50,
    "d12":    2.00,
    "d20":    2.50,
    "d100":   2.50,
    "df":     1.00,
}
_DEFAULT_UV_SCALE = 1.0

_USE_LONG_AXIS = {"d10", "d100"}


def _face_uvs(vertices: np.ndarray, face: list[int],
              normal: np.ndarray,
              uv_scale: float = 1.0,
              use_long_axis: bool = False) -> np.ndarray:
    """
    Calculates local UVs for the vertices of a polygonal face.    
    """
    pts = vertices[face].astype(np.float64)
    centroid = pts.mean(axis=0)
    local = pts - centroid
    n = np.asarray(normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-15)

    if use_long_axis:

        ref = np.array([1.0, 0.0, 0.0])

        if abs(np.dot(ref, n)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])

        e1 = ref - np.dot(ref, n) * n
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(n, e1)

        coords2d = np.stack([local @ e1, local @ e2], axis=1)  # (N, 2)
        cov = coords2d.T @ coords2d

        eigvals, eigvecs = np.linalg.eigh(cov)
        long_2d = eigvecs[:, np.argmax(eigvals)]
        u_axis = long_2d[0] * e1 + long_2d[1] * e2
        u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-15)

        if u_axis[1] < 0:
            u_axis = -u_axis
    else:
        world_up = np.array([0.0, 1.0, 0.0])
        u_axis = world_up - np.dot(world_up, n) * n
        u_len = np.linalg.norm(u_axis)

        if u_len < 0.1:
            world_up = np.array([0.0, 0.0, 1.0])
            u_axis = world_up - np.dot(world_up, n) * n
            u_len = np.linalg.norm(u_axis)

        if u_len < 1e-9:
            u_axis = np.array([1.0, 0.0, 0.0])

        else:
            u_axis = u_axis / u_len

    v_axis = np.cross(n, u_axis)
    v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-15)

    u_coords = local @ u_axis
    v_coords = local @ v_axis
    max_r = max(np.max(np.abs(u_coords)), np.max(np.abs(v_coords)), 1e-9)
    u_coords = u_coords / max_r * uv_scale
    v_coords = v_coords / max_r * uv_scale

    return np.stack([u_coords, v_coords], axis=1).astype(np.float32)


# ── d4: Sub-triangles by edge ───────────────────────────────────────────
# In a real d4, each face displays the same number on each of its 3 edges,
# oriented so that the number is legible when viewed "from bottom to top" from that
# edge (i.e., perpendicular to the edge, pointing towards the interior of the face).
#
# Convention: Subtriangle k corresponds to the EDGE opposite vertex k.
# Edge k = [v_{k+1}, v_{k+2}] (the two vertices that are NOT v_k)
# Sub k = [v_{k+1}, v_{k+2}, centroid]
#
# The glyph is centered at the midpoint of the edge, with:
# U-axis: along the edge (from v_{k+1} to v_{k+2}), so that the number
# is parallel to the edge and "upright" relative to the interior.
# V-axis: perpendicular to U in the face plane, pointing inward
# (from mid_edge → centroid).
#
# The number displayed in the subtriangle k of face fi is the value of the face opposite
# to the global vertex face[k] — calculated in _from_state_d4.

_D4_GLYPH_SCALE = 6.5    # glyph scale in relation to edge length


def _d4_subtri_data(vertices: np.ndarray, face: list[int],
                    normal: np.ndarray
                    ) -> list[tuple[np.ndarray, np.ndarray]]:

    assert len(face) == 3

    pts = vertices[face].astype(np.float64)   # (3, 3)
    centroid = pts.mean(axis=0)                    # (3,)
    n = np.asarray(normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-15)

    result = []

    for k in range(3):
        va = pts[(k + 1) % 3]
        vb = pts[(k + 2) % 3]
        mid_edge = (va + vb) * 0.5

        sub_pts = np.array([va, vb, centroid])     # (3, 3)

        u_axis = vb - va
        u_axis = u_axis - np.dot(u_axis, n) * n
        u_len = np.linalg.norm(u_axis)

        if u_len < 1e-9:
            u_axis = np.array([1.0, 0.0, 0.0])
        else:
            u_axis = u_axis / u_len

        v_axis = np.cross(n, u_axis)
        v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-15)

        if np.dot(v_axis, centroid - mid_edge) < 0:
            v_axis = -v_axis

        edge_len = np.linalg.norm(vb - va)

        if edge_len < 1e-9:
            edge_len = 1.0

        half_glyph = edge_len / _D4_GLYPH_SCALE
        inward_safe = half_glyph * 0.90

        inward_max = np.linalg.norm(centroid - mid_edge) * 0.65
        inward = min(inward_safe, inward_max)
        draw_center = mid_edge + v_axis * inward

        local = sub_pts - draw_center
        u_coords = (local @ u_axis) / edge_len * _D4_GLYPH_SCALE
        v_coords = (local @ v_axis) / edge_len * _D4_GLYPH_SCALE

        uvs = np.stack([u_coords, v_coords], axis=1).astype(np.float32)
        result.append((sub_pts.astype(np.float32), uvs))

    return result


@dataclass
class DiceRenderData:
    vertex_buffer: np.ndarray
    index_buffer:  np.ndarray
    n_indices:     int
    face_glyphs:   list[int]
    glyph_color:   tuple = (1.0, 1.0, 1.0)
    model_mat:     np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float32))
    is_resting:    bool = False

    @classmethod
    def from_state(cls, state: "DiceState", theme: str = "light") -> "DiceRenderData":

        dice_type = state.dice.dice_type

        if dice_type == "d4":
            return cls._from_state_d4(state, theme)

        uv_scale = GLYPH_UV_SCALE.get(dice_type, _DEFAULT_UV_SCALE)
        use_long_axis = dice_type in _USE_LONG_AXIS

        return cls._from_state_generic(state, uv_scale, use_long_axis, theme)

    @classmethod
    def _from_state_generic(cls, state: "DiceState",
                            uv_scale: float,
                            use_long_axis: bool,
                            theme: str = "light") -> "DiceRenderData":

        mesh = state.dice.mesh
        dice_type = state.dice.dice_type

        tris: list[tuple[int, int, int]] = mesh.triangulated_faces()

        face_of_tri: list[int] = []

        for fi, face in enumerate(mesh.faces):
            n_tris = len(list(face)) - 2
            face_of_tri.extend([fi] * n_tris)

        face_vert_uv: list[np.ndarray] = []

        for fi, face in enumerate(mesh.faces):
            uvs = _face_uvs(mesh.vertices, list(face), mesh.normals[fi],
                            uv_scale, use_long_axis)
            face_vert_uv.append(uvs)

        face_vert_local: list[dict[int, int]] = []

        for face in mesh.faces:
            face_list = list(face)
            vertex_to_local_index = {}

            for index, vertex in enumerate(face_list):
                vertex_to_local_index[vertex] = index

            face_vert_local.append(vertex_to_local_index)

        n_tris = len(tris)
        vb = np.zeros((n_tris * 3, 9), dtype=np.float32)
        ib = np.arange(n_tris * 3, dtype=np.uint32)

        for ti, (i0, i1, i2) in enumerate(tris):
            fi = face_of_tri[ti]
            normal = mesh.normals[fi].astype(np.float32)
            local = face_vert_local[fi]

            for k, vi in enumerate((i0, i1, i2)):
                row = ti * 3 + k
                vb[row, :3] = mesh.vertices[vi].astype(np.float32)
                vb[row, 3:6] = normal
                li = local.get(vi, 0)
                vb[row, 6:8] = face_vert_uv[fi][li]
                vb[row, 8] = float(fi)

        face_glyphs = build_face_glyphs(dice_type, list(mesh.face_values))
        glyph_color = _choose_glyph_color(dice_type, theme)

        return cls(vertex_buffer=vb, index_buffer=ib, n_indices=len(ib),
                   face_glyphs=face_glyphs, glyph_color=glyph_color)

    @classmethod
    def _from_state_d4(cls, state: "DiceState", theme: str = "light") -> "DiceRenderData":

        mesh = state.dice.mesh
        dice_type = state.dice.dice_type

        base_glyphs = build_face_glyphs(dice_type, list(mesh.face_values))
        glyph_color = _choose_glyph_color(dice_type, theme)

        # n_faces = len(mesh.faces)
        vert_to_opposite_face: dict[int, int] = {}
        all_verts = set(range(mesh.num_vertices))

        for fi, face in enumerate(mesh.faces):
            face_set = set(face)
            missing = all_verts - face_set

            for v in missing:
                vert_to_opposite_face[v] = fi

        expanded_glyphs: list[int] = [GLYPH_NONE] * MAX_FACES

        for fi, face in enumerate(mesh.faces):
            for k in range(3):
                global_vert = face[k]
                opp_fi = vert_to_opposite_face.get(global_vert, fi)

                if opp_fi < len(base_glyphs):
                    glyph = base_glyphs[opp_fi]
                else:
                    glyph = GLYPH_NONE

                slot = fi * 3 + k

                if slot < MAX_FACES:
                    expanded_glyphs[slot] = glyph

        vb = np.zeros((36, 9), dtype=np.float32)
        ib = np.arange(36, dtype=np.uint32)

        row = 0

        for fi, face in enumerate(mesh.faces):
            face_list = list(face)
            normal_f32 = mesh.normals[fi].astype(np.float32)
            subtris = _d4_subtri_data(
                mesh.vertices, face_list, mesh.normals[fi])

            for k, (sub_pos, sub_uvs) in enumerate(subtris):
                face_slot = float(fi * 3 + k)

                for j in range(3):
                    vb[row, :3] = sub_pos[j]
                    vb[row, 3:6] = normal_f32
                    vb[row, 6:8] = sub_uvs[j]
                    vb[row, 8] = face_slot
                    row += 1

        return cls(vertex_buffer=vb, index_buffer=ib, n_indices=36,
                   face_glyphs=expanded_glyphs, glyph_color=glyph_color)


def _choose_glyph_color(dice_type: str, theme: str = "light") -> tuple:
    t = DICE_THEMES.get(theme)
    if t is not None:
        return t.glyph_color


class RenderScene:
    def __init__(self, dice_renders: list[DiceRenderData]) -> None:
        self.dice_renders = dice_renders

    @classmethod
    def from_states(cls, states: list["DiceState"], theme: str = "light") -> "RenderScene":
        render_data = []

        for state in states:
            data = DiceRenderData.from_state(state, theme)
            render_data.append(data)

        return cls(render_data)

    def update(self, states: list["DiceState"], alpha: float = 1.0) -> None:
        for rd, state in zip(self.dice_renders, states):
            R = state.interpolated_rotation_matrix(alpha)
            pos = state.dice.position
            M = np.eye(4, dtype=np.float32)
            M[:3, :3] = R.astype(np.float32)
            M[:3,  3] = pos
            rd.model_mat = M
            rd.is_resting = state.is_resting

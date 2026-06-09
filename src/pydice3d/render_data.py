"""
render_data.py – Dados de Render CPU
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydice3d.shaders import GLYPH_NONE, MAX_FACES

if TYPE_CHECKING:
    from pydice3d.dice_state import DiceState

from pydice3d.shaders import GLYPH_PLUS, GLYPH_MINUS, GLYPH_BLANK, glyph_d100

FUDGE_GLYPHS = [GLYPH_PLUS, GLYPH_PLUS,
                GLYPH_MINUS, GLYPH_MINUS,
                GLYPH_BLANK, GLYPH_BLANK]


def build_face_glyphs(dice_type: str, face_values: list[int]) -> list[int]:
    if dice_type == "df":
        return list(FUDGE_GLYPHS)
    glyphs = []
    for v in face_values:
        if dice_type == "d100":
            glyphs.append(glyph_d100(v))
        elif dice_type == "d10":
            # d10 real vai de 0-9; face_values tem 1-10 onde 10 representa o 0
            glyphs.append(0 if int(v) == 10 else int(v))
        else:
            glyphs.append(int(v) if 0 <= int(v) <= 20 else GLYPH_NONE)
    return glyphs


# ── Escala UV por tipo de dado ───────────────────────────────────────────────
# Maior = glifo menor na face; menor = glifo maior.
GLYPH_UV_SCALE: dict[str, float] = {
    "d4":     2.50,    # d4 tem lógica própria
    "d6":     1.20,
    "d8":     2.00,
    "d10":    2.50,   # kite: orientação corrigida via eixo próprio
    "d12":    2.00,
    "d20":    2.50,
    "d100":   2.50,   # mesma geometria kite que o d10
    "df":     1.00,
}
_DEFAULT_UV_SCALE = 1.0


# ── Tipos que usam eixo longo da face como U (não world-up) ─────────────────
# Para o d10 (kite), o eixo longo vai do polo até o vértice de base,
# passando pelo centróide — é o eixo de simetria da kite.
_USE_LONG_AXIS = {"d10", "d100"}


def _face_uvs(vertices: np.ndarray, face: list[int],
              normal: np.ndarray,
              uv_scale: float = 1.0,
              use_long_axis: bool = False) -> np.ndarray:
    """
    Calcula UVs locais para os vértices de uma face poligonal.

    use_long_axis=False (padrão): eixo U = projeção de world-up no plano,
      orientação consistente para dados com faces simétricas (d6, d8, d20…).

    use_long_axis=True (d10/d100 kite): eixo U = direção que maximiza a
      variância dos vértices projetados, i.e. o eixo longo da face.
      Para a kite, isso alinha U com polo→base (eixo de simetria da face).
    """
    pts      = vertices[face].astype(np.float64)
    centroid = pts.mean(axis=0)
    local    = pts - centroid
    n        = np.asarray(normal, dtype=np.float64)
    n        = n / (np.linalg.norm(n) + 1e-15)

    if use_long_axis:
        # Projeta todos os vértices no plano da face e faz PCA manual (2D)
        # para encontrar o eixo de maior variância.
        # Qualquer base ortogonal no plano serve de partida.
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, n)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        e1 = ref - np.dot(ref, n) * n
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(n, e1)

        coords2d = np.stack([local @ e1, local @ e2], axis=1)  # (N, 2)
        cov = coords2d.T @ coords2d
        # Eigenvector do maior eigenvalue → eixo longo
        eigvals, eigvecs = np.linalg.eigh(cov)
        long_2d = eigvecs[:, np.argmax(eigvals)]
        u_axis  = long_2d[0] * e1 + long_2d[1] * e2
        u_axis  = u_axis / (np.linalg.norm(u_axis) + 1e-15)

        # Garante que U aponta "para cima" (componente Y positiva no mundo)
        if u_axis[1] < 0:
            u_axis = -u_axis
    else:
        # Projeção de world-up no plano da face
        world_up = np.array([0.0, 1.0, 0.0])
        u_axis   = world_up - np.dot(world_up, n) * n
        u_len    = np.linalg.norm(u_axis)
        if u_len < 0.1:
            world_up = np.array([0.0, 0.0, 1.0])
            u_axis   = world_up - np.dot(world_up, n) * n
            u_len    = np.linalg.norm(u_axis)
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


# ── d4: sub-triângulos por aresta ────────────────────────────────────────────
# No d4 real, cada face exibe o mesmo número em cada uma de suas 3 arestas,
# orientado de modo que o número fique legível quando visto "de baixo para
# cima" a partir dessa aresta (i.e. perpendicular à aresta, apontando para
# o interior da face).
#
# Convenção: sub-triângulo k corresponde à ARESTA oposta ao vértice k.
#
#   aresta k = [v_{k+1}, v_{k+2}]   (os dois vértices que NÃO são v_k)
#   sub k    = [v_{k+1}, v_{k+2}, centróide]
#
# O glifo é centralizado no midpoint da aresta, com:
#   U-axis: ao longo da aresta (de v_{k+1} para v_{k+2}), para que o número
#           fique paralelo à aresta e "de pé" em relação ao interior.
#   V-axis: perpendicular a U no plano da face, apontando para o interior
#           (de mid_aresta → centróide).
#
# O número exibido no sub-triângulo k da face fi é o valor da face oposta
# ao vértice global face[k] — calculado em _from_state_d4.

_D4_GLYPH_SCALE      = 6.5    # escala do glifo em relação ao comprimento da aresta


def _d4_subtri_data(vertices: np.ndarray, face: list[int],
                    normal: np.ndarray
                    ) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Retorna lista de 3 tuplas (positions (3,3), uvs (3,2)) — uma por sub-triângulo.

    Sub-triângulo k cobre a aresta oposta ao vértice k:
        vértices do sub = [v_{k+1}, v_{k+2}, centróide]
    UVs centrados no draw_center sobre a aresta, com U ao longo da aresta.
    """
    assert len(face) == 3
    pts      = vertices[face].astype(np.float64)   # (3, 3)
    centroid = pts.mean(axis=0)                    # (3,)
    n        = np.asarray(normal, dtype=np.float64)
    n        = n / (np.linalg.norm(n) + 1e-15)

    result = []
    for k in range(3):
        # Aresta oposta ao vértice k
        va = pts[(k + 1) % 3]
        vb = pts[(k + 2) % 3]
        mid_edge = (va + vb) * 0.5

        # Sub-triângulo cobre a região da aresta: [va, vb, centróide]
        sub_pts = np.array([va, vb, centroid])     # (3, 3)

        # Eixo U: ao longo da aresta (va → vb), projetado no plano da face
        u_axis = vb - va
        u_axis = u_axis - np.dot(u_axis, n) * n
        u_len  = np.linalg.norm(u_axis)
        if u_len < 1e-9:
            u_axis = np.array([1.0, 0.0, 0.0])
        else:
            u_axis = u_axis / u_len

        # Eixo V: perpendicular a U no plano, apontando para o interior da face
        v_axis = np.cross(n, u_axis)
        v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-15)
        # Garante que V aponta de mid_edge → centróide (interior)
        if np.dot(v_axis, centroid - mid_edge) < 0:
            v_axis = -v_axis

        edge_len = np.linalg.norm(vb - va)
        if edge_len < 1e-9:
            edge_len = 1.0

        # O shader desenha o glifo na faixa v_uv.y ∈ [-1, +1].
        # Para que a metade 'inferior' do glifo (v < 0, além da aresta) não
        # seja clipada pelo limite do sub-triângulo, o draw_center precisa
        # estar pelo menos half_glyph_world para dentro da aresta:
        #   half_glyph_world = edge_len / _D4_GLYPH_SCALE
        half_glyph  = edge_len / _D4_GLYPH_SCALE  # metade do glifo no mundo
        inward_safe = half_glyph * 0.90            # 15 % de margem extra
        # Não ultrapassa 65 % do caminho até o centróide
        inward_max  = np.linalg.norm(centroid - mid_edge) * 0.65
        inward      = min(inward_safe, inward_max)
        draw_center = mid_edge + v_axis * inward

        # UVs: normalizados pelo comprimento da aresta para escala consistente
        local    = sub_pts - draw_center
        u_coords = (local @ u_axis) / edge_len * _D4_GLYPH_SCALE
        v_coords = (local @ v_axis) / edge_len * _D4_GLYPH_SCALE

        uvs = np.stack([u_coords, v_coords], axis=1).astype(np.float32)
        result.append((sub_pts.astype(np.float32), uvs))

    return result


# ── DiceRenderData ───────────────────────────────────────────────────────────

@dataclass
class DiceRenderData:
    vertex_buffer: np.ndarray
    index_buffer:  np.ndarray
    n_indices:     int
    face_glyphs:   list[int]
    glyph_color:   tuple = (1.0, 1.0, 1.0)
    model_mat:     np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    is_resting:    bool = False

    @classmethod
    def from_state(cls, state: "DiceState") -> "DiceRenderData":
        mesh      = state.dice.mesh
        dice_type = state.dice.dice_type
        if dice_type == "d4":
            return cls._from_state_d4(state)
        uv_scale     = GLYPH_UV_SCALE.get(dice_type, _DEFAULT_UV_SCALE)
        use_long_axis = dice_type in _USE_LONG_AXIS
        return cls._from_state_generic(state, uv_scale, use_long_axis)

    @classmethod
    def _from_state_generic(cls, state: "DiceState",
                             uv_scale: float,
                             use_long_axis: bool) -> "DiceRenderData":
        mesh      = state.dice.mesh
        dice_type = state.dice.dice_type

        tris: list[tuple[int,int,int]] = mesh.triangulated_faces()

        face_of_tri: list[int] = []
        for fi, face in enumerate(mesh.faces):
            n_tris = len(list(face)) - 2
            face_of_tri.extend([fi] * n_tris)

        face_vert_uv: list[np.ndarray] = []
        for fi, face in enumerate(mesh.faces):
            uvs = _face_uvs(mesh.vertices, list(face), mesh.normals[fi],
                            uv_scale, use_long_axis)
            face_vert_uv.append(uvs)

        face_vert_local: list[dict[int,int]] = []
        for face in mesh.faces:
            face_list = list(face)
            face_vert_local.append({v: i for i, v in enumerate(face_list)})

        n_tris = len(tris)
        vb = np.zeros((n_tris * 3, 9), dtype=np.float32)
        ib = np.arange(n_tris * 3, dtype=np.uint32)

        for ti, (i0, i1, i2) in enumerate(tris):
            fi     = face_of_tri[ti]
            normal = mesh.normals[fi].astype(np.float32)
            local  = face_vert_local[fi]
            for k, vi in enumerate((i0, i1, i2)):
                row = ti * 3 + k
                vb[row, :3]  = mesh.vertices[vi].astype(np.float32)
                vb[row, 3:6] = normal
                li = local.get(vi, 0)
                vb[row, 6:8] = face_vert_uv[fi][li]
                vb[row, 8]   = float(fi)

        face_glyphs = build_face_glyphs(dice_type, list(mesh.face_values))
        glyph_color = _choose_glyph_color(dice_type)
        return cls(vertex_buffer=vb, index_buffer=ib, n_indices=len(ib),
                   face_glyphs=face_glyphs, glyph_color=glyph_color)

    @classmethod
    def _from_state_d4(cls, state: "DiceState") -> "DiceRenderData":
        """
        Cada face triangular é dividida em 3 sub-triângulos (via centróide).
        Sub-triângulo k cobre a ARESTA oposta ao vértice local k:
            sub k = [v_{k+1}, v_{k+2}, centróide]
        O glifo exibido na aresta k de fi é o valor da face oposta ao
        vértice global face[k] — ou seja, a face cujo índice é face[k]
        (pois no tetraedro a face fi é oposta ao vértice global fi).
        face_idx = fi*3 + k  →  face_glyphs expandido para 12 entradas.
        """
        mesh      = state.dice.mesh
        dice_type = state.dice.dice_type

        base_glyphs = build_face_glyphs(dice_type, list(mesh.face_values))
        glyph_color = _choose_glyph_color(dice_type)

        # Monta tabela: vértice global v → índice da face oposta a v
        # No tetraedro padrão: face fi é oposta ao vértice fi.
        # Isso é verdade pela construção de _build_d4 (face i não contém vértice i).
        # Verificamos explicitamente para robustez.
        n_faces = len(mesh.faces)
        vert_to_opposite_face: dict[int, int] = {}
        all_verts = set(range(mesh.num_vertices))
        for fi, face in enumerate(mesh.faces):
            face_set = set(face)
            missing = all_verts - face_set
            for v in missing:
                vert_to_opposite_face[v] = fi

        # 12 slots: fi*3+k para fi=0..3, k=0..2
        # Aresta k da face fi é oposta ao vértice local k → vértice global face[k].
        # O glifo dessa aresta = valor da face oposta ao vértice global face[k].
        expanded_glyphs: list[int] = [GLYPH_NONE] * MAX_FACES
        for fi, face in enumerate(mesh.faces):
            for k in range(3):
                global_vert = face[k]
                opp_fi = vert_to_opposite_face.get(global_vert, fi)
                glyph  = base_glyphs[opp_fi] if opp_fi < len(base_glyphs) else GLYPH_NONE
                slot   = fi * 3 + k
                if slot < MAX_FACES:
                    expanded_glyphs[slot] = glyph

        # 4 faces × 3 sub-triângulos × 3 vértices = 36 vértices
        vb = np.zeros((36, 9), dtype=np.float32)
        ib = np.arange(36, dtype=np.uint32)

        row = 0
        for fi, face in enumerate(mesh.faces):
            face_list  = list(face)
            normal_f32 = mesh.normals[fi].astype(np.float32)
            subtris    = _d4_subtri_data(mesh.vertices, face_list, mesh.normals[fi])

            for k, (sub_pos, sub_uvs) in enumerate(subtris):
                face_slot = float(fi * 3 + k)
                for j in range(3):
                    vb[row, :3]  = sub_pos[j]
                    vb[row, 3:6] = normal_f32
                    vb[row, 6:8] = sub_uvs[j]
                    vb[row, 8]   = face_slot
                    row += 1

        return cls(vertex_buffer=vb, index_buffer=ib, n_indices=36,
                   face_glyphs=expanded_glyphs, glyph_color=glyph_color)


def _choose_glyph_color(dice_type: str) -> tuple:
    from pydice3d.renderer import DICE_COLORS, DEFAULT_DICE_COLOR, DICE_THEMES
    # r, g, b = DICE_COLORS.get(dice_type, DEFAULT_DICE_COLOR)
    r, g, b = DICE_THEMES.get("light", DEFAULT_DICE_COLOR)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if luminance > 0.5:
        return (0.08, 0.08, 0.10)
    return (0.95, 0.95, 0.95)


# ── RenderScene ──────────────────────────────────────────────────────────────

class RenderScene:
    def __init__(self, dice_renders: list[DiceRenderData]) -> None:
        self.dice_renders = dice_renders

    @classmethod
    def from_states(cls, states: list["DiceState"]) -> "RenderScene":
        return cls([DiceRenderData.from_state(s) for s in states])

    def update(self, states: list["DiceState"], alpha: float = 1.0) -> None:
        for rd, state in zip(self.dice_renders, states):
            R   = state.interpolated_rotation_matrix(alpha)
            pos = state.dice.position
            M = np.eye(4, dtype=np.float32)
            M[:3, :3] = R.astype(np.float32)
            M[:3,  3] = pos
            rd.model_mat  = M
            rd.is_resting = state.is_resting
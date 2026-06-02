"""
render_data.py – Dados de Render CPU

Responsabilidade: preparar os buffers NumPy e matrizes de modelo que o
Renderer consumirá a cada frame. Não toca em OpenGL.

Estruturas
──────────
DiceRenderData  : buffers de vértice/índice + estado por-dado (model_mat, is_resting)
RenderScene     : coleção de DiceRenderData + utilitários de atualização

Fluxo
─────
    # Na inicialização (após contexto OpenGL)
    scene = RenderScene.from_states(states)
    renderer = Renderer(scene, dice_types)

    # A cada frame
    scene.update(states, alpha)   # atualiza model_mat via slerp
    renderer.draw(scene, camera, w, h)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dice_state import DiceState


# ────────────────────────────────────────────────────────────────────────────
# DiceRenderData
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class DiceRenderData:
    """
    Dados de render de um único dado.

    Atributos imutáveis (criados uma vez)
    ──────────────────────────────────────
    vertex_buffer : float32 (N, 6) — posição(3) + normal(3) por vértice
    index_buffer  : uint32  (M,)   — índices para GL_TRIANGLES
    n_indices     : número de índices

    Atributos mutáveis (atualizados por frame)
    ──────────────────────────────────────────
    model_mat  : float32 (4,4) — matriz de modelo TRS (só T+R, escala = 1)
    is_resting : bool — True quando dado parou (ativa destaque dourado)
    """
    vertex_buffer: np.ndarray    # float32 (N, 6)
    index_buffer:  np.ndarray    # uint32  (M,)
    n_indices:     int
    model_mat:     np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    is_resting:    bool = False

    @classmethod
    def from_state(cls, state: "DiceState") -> "DiceRenderData":
        """
        Constrói DiceRenderData a partir de um DiceState.
        Triangula a malha e monta o vertex_buffer intercalado.
        """
        mesh = state.dice.mesh

        # Triangula faces (fan triangulation para polígonos)
        tris = mesh.triangulated_faces()  # list[(i0,i1,i2)]

        # Constrói mapeamento vértice→normal-de-face para flat shading
        # Cada triângulo recebe a normal da face original
        # face_of_tri[t] = índice da face a que pertence o triângulo t
        face_of_tri: list[int] = []
        for fi, face in enumerate(mesh.faces):
            face_list = list(face)
            n_tris = len(face_list) - 2
            face_of_tri.extend([fi] * n_tris)

        n_tris = len(tris)
        vb = np.zeros((n_tris * 3, 6), dtype=np.float32)
        ib = np.arange(n_tris * 3, dtype=np.uint32)

        for ti, (i0, i1, i2) in enumerate(tris):
            normal = mesh.normals[face_of_tri[ti]].astype(np.float32)
            for k, vi in enumerate((i0, i1, i2)):
                row = ti * 3 + k
                vb[row, :3] = mesh.vertices[vi].astype(np.float32)
                vb[row, 3:] = normal

        return cls(
            vertex_buffer=vb,
            index_buffer=ib,
            n_indices=len(ib),
        )


# ────────────────────────────────────────────────────────────────────────────
# RenderScene
# ────────────────────────────────────────────────────────────────────────────

class RenderScene:
    """
    Coleção de DiceRenderData para todos os dados da cena.

    Uso
    ───
    scene = RenderScene.from_states(states)
    # a cada frame:
    scene.update(states, alpha)
    """

    def __init__(self, dice_renders: list[DiceRenderData]) -> None:
        self.dice_renders = dice_renders

    @classmethod
    def from_states(cls, states: list["DiceState"]) -> "RenderScene":
        """Cria a cena a partir dos estados iniciais."""
        renders = [DiceRenderData.from_state(s) for s in states]
        return cls(renders)

    def update(self, states: list["DiceState"], alpha: float = 1.0) -> None:
        """
        Atualiza model_mat e is_resting para cada dado.

        alpha : fator de interpolação slerp [0,1] entre o frame anterior e o atual.
                Permite renderização suave mesmo quando a física corre a taxa fixa.
        """
        for rd, state in zip(self.dice_renders, states):
            R = state.interpolated_rotation_matrix(alpha)  # 3×3 float64
            pos = state.dice.position                       # float32 (3,)

            M = np.eye(4, dtype=np.float32)
            M[:3, :3] = R.astype(np.float32)
            M[:3,  3] = pos
            rd.model_mat  = M
            rd.is_resting = state.is_resting
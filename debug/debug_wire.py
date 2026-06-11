"""
collision_wire.py – Wireframe de Colisão (Debug OpenGL)

Arquivo de desenvolvimento — pertence à camada GTK/frontend, não à lib core.
Coloque junto com glarena.py e main.py.

Responsabilidade: construir e renderizar o convex hull de colisão de um dado
como GL_LINES, sobreposto ou substituindo o mesh visual durante inspeção.

Diferença em relação à versão anterior
────────────────────────────────────────
Não usa mais scipy.spatial.ConvexHull. As arestas são extraídas diretamente
do DiceMesh (que já tem a lista de faces hardcoded), eliminando o cálculo
de hull em runtime e a dependência de scipy neste arquivo.

Constantes de modo de debug
────────────────────────────
    DEBUG_NONE      = 0   renderização normal
    DEBUG_COLLISION = 1   só wireframe (sem mesh visual)
    DEBUG_OVERLAY   = 2   mesh visual + wireframe sobrepostos

Uso
────
    wire = CollisionWireframe("d20")          # após contexto GL ativo
    wire.draw(mvp_matrix, wire_program)       # a cada frame
    wire.delete()                             # ao destruir contexto

    prog = build_wire_program()               # compila shader de wireframe
"""

from __future__ import annotations

import numpy as np
from OpenGL import GL

from pydice3d.dice_mesh  import get_mesh
from pydice3d.physics    import DICE_TARGET_SIZE
from pydice3d.shaders    import (
    build_program, WIRE_VERT, WIRE_FRAG,
    set_uniform_mat4, set_uniform_vec3,
)


# ────────────────────────────────────────────────────────────────────────────
# Constantes de debug
# ────────────────────────────────────────────────────────────────────────────

DEBUG_NONE      = 0
DEBUG_COLLISION = 1
DEBUG_OVERLAY   = 2

DEFAULT_WIRE_COLOR: tuple[float, float, float] = (0.0, 1.0, 0.3)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def build_wire_program() -> int:
    """Compila e linka o programa GLSL de wireframe. Requer contexto GL ativo."""
    return build_program(WIRE_VERT, WIRE_FRAG)


def _mesh_edges(dice_type: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Extrai vértices e arestas únicas do DiceMesh correspondente ao
    collision shape usado em PhysicsWorld._make_collision_shape.

    Não usa scipy — as faces já estão hardcoded no DiceMesh.
    Cada aresta (i, j) aparece exatamente uma vez (i < j).
    """
    r = DICE_TARGET_SIZE * 1.1
    if dice_type in ("d10", "d100"):
        r = DICE_TARGET_SIZE * 1.1 * 1.5

    mesh  = get_mesh(dice_type)
    verts = (mesh.vertices * r).astype(np.float32)

    edge_set: set[tuple[int, int]] = set()
    for face in mesh.faces:
        n = len(face)
        for k in range(n):
            a, b = int(face[k]), int(face[(k + 1) % n])
            edge_set.add((min(a, b), max(a, b)))

    return verts, list(edge_set)


def _box_edges() -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Arestas do cubo de colisão usado para d6 e df."""
    half = DICE_TARGET_SIZE / 2.0
    verts = np.array([
        [-half, -half, -half], [ half, -half, -half],
        [ half,  half, -half], [-half,  half, -half],
        [-half, -half,  half], [ half, -half,  half],
        [ half,  half,  half], [-half,  half,  half],
    ], dtype=np.float32)
    edges = [
        (0,1),(1,2),(2,3),(3,0),   # face -Z
        (4,5),(5,6),(6,7),(7,4),   # face +Z
        (0,4),(1,5),(2,6),(3,7),   # laterais
    ]
    return verts, edges


# ────────────────────────────────────────────────────────────────────────────
# CollisionWireframe
# ────────────────────────────────────────────────────────────────────────────

class CollisionWireframe:
    """
    Renderiza as arestas do collision shape de um dado como GL_LINES.

    O VAO/VBO é construído lazy na primeira draw(), garantindo que o
    contexto OpenGL já esteja ativo nesse momento.

    Parâmetros
    ----------
    dice_type : tipo do dado — determina a geometria de colisão,
                espelhando PhysicsWorld._make_collision_shape.
    """

    def __init__(self, dice_type: str) -> None:
        self.dice_type = dice_type
        self._vao:     int  = 0
        self._vbo:     int  = 0
        self._n_verts: int  = 0
        self._built:   bool = False

    def _build(self) -> None:
        if self._built:
            return

        if self.dice_type in ("d6", "df"):
            verts, edges = _box_edges()
        else:
            verts, edges = _mesh_edges(self.dice_type)

        if not edges:
            return

        # Cada aresta = 2 vértices
        line_buf = np.array(
            [[verts[i], verts[j]] for i, j in edges],
            dtype=np.float32,
        ).reshape(-1, 3)

        self._vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao)

        self._vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, line_buf.nbytes,
                        line_buf.tobytes(), GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 12,
                                  GL.ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        GL.glBindVertexArray(0)

        self._n_verts = len(line_buf)
        self._built   = True

    # ── API pública ──────────────────────────────────────────────────────────

    def draw(
        self,
        mvp:     np.ndarray,
        program: int,
        color:   tuple[float, float, float] = DEFAULT_WIRE_COLOR,
    ) -> None:
        if not self._built:
            self._build()
        if not self._built or self._n_verts == 0:
            return

        GL.glUseProgram(program)
        set_uniform_mat4(program, "u_mvp",   mvp)
        set_uniform_vec3(program, "u_color", color)
        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_LINES, 0, self._n_verts)
        GL.glBindVertexArray(0)

    def delete(self) -> None:
        """Libera recursos OpenGL. Chamar antes de destruir o contexto GL."""
        if self._built:
            GL.glDeleteVertexArrays(1, [self._vao])
            GL.glDeleteBuffers(1,    [self._vbo])
            self._vao   = 0
            self._vbo   = 0
            self._built = False
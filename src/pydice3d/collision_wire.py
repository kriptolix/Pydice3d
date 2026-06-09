"""
collision_wire.py – Wireframe de Colisão (Debug OpenGL)

Responsabilidade: construir e renderizar o convex hull de colisão de um
dado como linhas OpenGL (GL_LINES), sobreposto ou substituindo o mesh
visual durante inspeção de física.

Pertence à camada OpenGL da lib — depende de PyOpenGL e scipy, mas não de
GTK, Qt ou qualquer outro toolkit de janela.

Constantes de modo de debug
────────────────────────────
    DEBUG_NONE      = 0   renderização normal
    DEBUG_COLLISION = 1   só wireframe (sem mesh visual)
    DEBUG_OVERLAY   = 2   mesh visual + wireframe sobrepostos

Uso
────
    # Uma instância por dado, criada após contexto OpenGL ativo:
    wire = CollisionWireframe("d20")

    # A cada frame (dentro do loop de render):
    wire.draw(mvp_matrix, wire_program)

    # Ao destruir o contexto:
    wire.delete()

    # Compilar o programa de shader de wireframe:
    prog = build_wire_program()
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
# Constantes de modo de debug (espelhadas aqui para evitar import circular
# com glarena em projetos que usam só esta camada)
# ────────────────────────────────────────────────────────────────────────────

DEBUG_NONE      = 0
DEBUG_COLLISION = 1
DEBUG_OVERLAY   = 2

# Cor padrão do wireframe — pode ser sobrescrita em draw()
DEFAULT_WIRE_COLOR: tuple[float, float, float] = (0.0, 1.0, 0.3)   # verde neon


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def build_wire_program() -> int:
    """Compila e linka o programa GLSL de wireframe. Requer contexto GL ativo."""
    return build_program(WIRE_VERT, WIRE_FRAG)


def _hull_edges(verts: np.ndarray) -> list[tuple[int, int]]:
    """Extrai arestas únicas do convex hull de `verts` (N×3)."""
    if len(verts) < 4:
        return []
    try:
        from scipy.spatial import ConvexHull
        hull     = ConvexHull(verts)
        edge_set: set[tuple[int, int]] = set()
        for simplex in hull.simplices:
            for i in range(len(simplex)):
                a, b = simplex[i], simplex[(i + 1) % len(simplex)]
                edge_set.add((min(a, b), max(a, b)))
        return list(edge_set)
    except Exception:
        return []


def _box_verts(half: float) -> np.ndarray:
    """8 cantos de um cubo centrado na origem com half-extent `half`."""
    pts = []
    for sx in (+1, -1):
        for sy in (+1, -1):
            for sz in (+1, -1):
                pts.append([sx * half, sy * half, sz * half])
    return np.array(pts, dtype=np.float32)


def _collision_verts(dice_type: str) -> np.ndarray:
    """
    Retorna os vértices do shape de colisão PyBullet para `dice_type`,
    na mesma escala usada em PhysicsWorld._make_collision_shape.
    """
    r = DICE_TARGET_SIZE * 1.1

    if dice_type in ("d6", "df"):
        # GEOM_BOX: halfExtents = DICE_TARGET_SIZE / 2
        return _box_verts(DICE_TARGET_SIZE / 2.0)

    if dice_type in ("d10", "d100"):
        r = DICE_TARGET_SIZE * 1.1 * 1.5   # mesmo fator aplicado em physics.py

    mesh = get_mesh(dice_type)
    return (mesh.vertices * r).astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
# CollisionWireframe
# ────────────────────────────────────────────────────────────────────────────

class CollisionWireframe:
    """
    Renderiza o convex hull de colisão de um dado como wireframe GL_LINES.

    O VAO/VBO é construído de forma lazy na primeira chamada a draw(),
    garantindo que o contexto OpenGL já esteja ativo nesse momento.

    Parâmetros
    ----------
    dice_type : tipo do dado ("d6", "d20", etc.) — determina a geometria
                de colisão usada, espelhando PhysicsWorld._make_collision_shape.
    """

    def __init__(self, dice_type: str) -> None:
        self.dice_type = dice_type
        self._vao:     int  = 0
        self._vbo:     int  = 0
        self._n_lines: int  = 0
        self._built:   bool = False

    # ── construção lazy do VAO ───────────────────────────────────────────────

    def _build(self) -> None:
        """Constrói VAO/VBO. Chamado automaticamente na primeira draw()."""
        if self._built:
            return

        verts = _collision_verts(self.dice_type)
        edges = _hull_edges(verts)
        if not edges:
            return

        # Buffer de linhas: cada aresta = 2 vértices × 3 floats
        line_buf = np.array(
            [[verts[i], verts[j]] for i, j in edges],
            dtype=np.float32,
        ).reshape(-1, 3)

        self._vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao)

        self._vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, line_buf.nbytes,
            line_buf.tobytes(), GL.GL_STATIC_DRAW,
        )
        GL.glVertexAttribPointer(
            0, 3, GL.GL_FLOAT, GL.GL_FALSE, 12,
            GL.ctypes.c_void_p(0),
        )
        GL.glEnableVertexAttribArray(0)
        GL.glBindVertexArray(0)

        self._n_lines = len(edges) * 2   # 2 vértices por aresta
        self._built   = True

    # ── API pública ──────────────────────────────────────────────────────────

    def draw(
        self,
        mvp:     np.ndarray,
        program: int,
        color:   tuple[float, float, float] = DEFAULT_WIRE_COLOR,
    ) -> None:
        """
        Renderiza o wireframe com a matriz MVP fornecida.

        Parâmetros
        ----------
        mvp     : matriz 4×4 float32 Model-View-Projection.
        program : programa GLSL compilado por build_wire_program().
        color   : cor RGB do wireframe (padrão: verde neon).
        """
        if not self._built:
            self._build()
        if not self._built or self._n_lines == 0:
            return

        GL.glUseProgram(program)
        set_uniform_mat4(program, "u_mvp",   mvp)
        set_uniform_vec3(program, "u_color", color)
        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_LINES, 0, self._n_lines)
        GL.glBindVertexArray(0)

    def delete(self) -> None:
        """Libera recursos OpenGL. Chamar antes de destruir o contexto GL."""
        if self._built:
            GL.glDeleteVertexArrays(1, [self._vao])
            GL.glDeleteBuffers(1,    [self._vbo])
            self._vao   = 0
            self._vbo   = 0
            self._built = False
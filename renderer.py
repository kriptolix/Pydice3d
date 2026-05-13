"""
renderer.py – Renderer OpenGL para Dados 3D

Responsabilidade: gerenciar os objetos OpenGL (VAO, VBO, EBO), configurar
uniforms e executar draw calls para cada dado e para o chão.

Requer contexto OpenGL 3.3 Core Profile ativo (fornecido pelo Gtk.GLArea).

Arquitetura
───────────
DiceGpuObject   : VAO + VBO + EBO para um único dado (criado uma vez por dado).
GroundPlane     : VAO + VBO para o plano do chão.
Renderer        : orquestra todos os DiceGpuObject + GroundPlane por frame.

Separação de responsabilidades
───────────────────────────────
    render_data.py  → prepara dados CPU (buffers NumPy, matrizes)
    renderer.py     → envia dados para GPU, configura uniforms, draw calls

Fluxo por frame
────────────────
    1. scene.update(states, alpha)       # CPU: atualiza matrizes e AABBs
    2. renderer.draw(scene, camera)      # GPU: envia uniforms + draw calls
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from OpenGL import GL

from camera import FixedCamera, OrbitalCamera
from render_data import RenderScene, DiceRenderData
from shaders import (
    build_dice_program, build_ground_program,
    set_uniform_mat4, set_uniform_mat3,
    set_uniform_vec3, set_uniform_float, set_uniform_bool,
)


# ────────────────────────────────────────────────────────────────────────────
# Cores dos dados por tipo
# ────────────────────────────────────────────────────────────────────────────

DICE_COLORS: dict[str, tuple[float, float, float]] = {
    "d4":  (0.85, 0.25, 0.25),   # vermelho
    "d6":  (0.25, 0.55, 0.90),   # azul
    "d8":  (0.25, 0.75, 0.40),   # verde
    "d10": (0.90, 0.65, 0.15),   # laranja
    "d12": (0.70, 0.30, 0.85),   # roxo
    "d20": (0.90, 0.90, 0.90),   # branco/prata
}
DEFAULT_DICE_COLOR = (0.7, 0.7, 0.7)


# ────────────────────────────────────────────────────────────────────────────
# DiceGpuObject — VAO/VBO/EBO de um único dado
# ────────────────────────────────────────────────────────────────────────────

class DiceGpuObject:
    """
    Objeto OpenGL para um único dado.

    Criado uma vez durante a inicialização do contexto OpenGL.
    Atualizado por frame apenas via uniforms (sem re-upload de vértices).

    Layout de vértice (stride = 6 × 4 bytes = 24 bytes):
        attribute 0: posição  (vec3, offset 0)
        attribute 1: normal   (vec3, offset 12)
    """

    def __init__(self, rd: DiceRenderData, dice_type: str) -> None:
        self.dice_type  = dice_type
        self.n_indices  = rd.n_indices
        self.color      = DICE_COLORS.get(dice_type, DEFAULT_DICE_COLOR)

        # Cria VAO
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        # VBO — vértices intercalados [pos(3) | normal(3)] × N
        self.vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        vb = rd.vertex_buffer                       # float32 (N, 6)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            vb.nbytes,
            vb.tobytes(),
            GL.GL_STATIC_DRAW,
        )

        stride = 6 * 4  # 6 floats × 4 bytes
        # Atributo 0: posição (vec3, offset 0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                  GL.ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        # Atributo 1: normal (vec3, offset 12)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                  GL.ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(1)

        # EBO — índices
        self.ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        ib = rd.index_buffer                         # uint32 (M,)
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER,
            ib.nbytes,
            ib.tobytes(),
            GL.GL_STATIC_DRAW,
        )

        GL.glBindVertexArray(0)

    def draw(self) -> None:
        GL.glBindVertexArray(self.vao)
        GL.glDrawElements(GL.GL_TRIANGLES, self.n_indices, GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)

    def delete(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteBuffers(1, [self.vbo])
        GL.glDeleteBuffers(1, [self.ebo])


# ────────────────────────────────────────────────────────────────────────────
# GroundPlane — plano do chão com grade
# ────────────────────────────────────────────────────────────────────────────

class GroundPlane:
    """
    Plano do chão renderizado como dois triângulos (quad) grande.

    Usa o shader de chão (grade via fragment shader).
    """

    HALF_SIZE = 50.0    # extensão do plano em unidades do mundo

    def __init__(self) -> None:
        s = self.HALF_SIZE
        y = 0.0          # GROUND_Y da spec
        # Dois triângulos formando um quad XZ
        verts = np.array([
            [-s, y, -s],
            [ s, y, -s],
            [ s, y,  s],
            [-s, y, -s],
            [ s, y,  s],
            [-s, y,  s],
        ], dtype=np.float32)

        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        self.vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts.tobytes(),
                        GL.GL_STATIC_DRAW)

        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 12,
                                  GL.ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        GL.glBindVertexArray(0)

        self.n_verts = len(verts)

    def draw(self) -> None:
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.n_verts)
        GL.glBindVertexArray(0)

    def delete(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteBuffers(1, [self.vbo])


# ────────────────────────────────────────────────────────────────────────────
# Parâmetros de iluminação
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class LightingParams:
    """Parâmetros da luz direcional Blinn-Phong."""
    light_dir:   np.ndarray = field(
        default_factory=lambda: np.array([0.6, 1.0, 0.8], dtype=np.float32)
    )
    light_color: tuple = (1.0, 0.97, 0.90)
    ambient:     tuple = (0.18, 0.18, 0.22)
    shininess:   float = 48.0

    def normalized_dir(self) -> np.ndarray:
        d = np.asarray(self.light_dir, dtype=float)
        n = np.linalg.norm(d)
        return (d / n).astype(np.float32) if n > 1e-8 else d


# ────────────────────────────────────────────────────────────────────────────
# Renderer principal
# ────────────────────────────────────────────────────────────────────────────

class Renderer:
    """
    Renderer OpenGL 3.3 Core Profile para a simulação de dados.

    Ciclo de vida
    ─────────────
    1. Criar após ter contexto OpenGL ativo:
           renderer = Renderer(scene, dice_types)
    2. A cada frame:
           scene.update(states, alpha)
           renderer.draw(scene, camera, width, height)
    3. Ao destruir o contexto:
           renderer.delete()

    Parâmetros
    ──────────
    scene      : RenderScene com DiceRenderData para cada dado
    dice_types : lista de strings ('d6', 'd20', etc.) na mesma ordem de scene
    lighting   : parâmetros de iluminação (opcional)
    """

    def __init__(
        self,
        scene: RenderScene,
        dice_types: list[str],
        lighting: Optional[LightingParams] = None,
    ) -> None:
        self.lighting = lighting or LightingParams()

        # Compila shaders
        self.dice_prog   = build_dice_program()
        self.ground_prog = build_ground_program()

        # Cria objetos GPU para cada dado
        self.dice_gpu: list[DiceGpuObject] = []
        for rd, dtype in zip(scene.dice_renders, dice_types):
            self.dice_gpu.append(DiceGpuObject(rd, dtype))

        # Plano do chão
        self.ground = GroundPlane()

        # Estado OpenGL fixo
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

    # ── draw call principal ──────────────────────────────────────────

    def draw(
        self,
        scene: RenderScene,
        camera: "FixedCamera | OrbitalCamera",
        width: int,
        height: int,
    ) -> None:
        """
        Renderiza um frame completo.

        Parâmetros
        ----------
        scene  : RenderScene atualizado (scene.update() já chamado)
        camera : câmera ativa
        width  : largura do viewport em pixels
        height : altura do viewport em pixels
        """
        GL.glViewport(0, 0, width, height)
        GL.glClearColor(0.08, 0.08, 0.10, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        VP = camera.view_projection(width, height)
        cam_pos = camera.position

        self._draw_ground(VP)
        self._draw_dice(scene, VP, cam_pos)

    def _draw_ground(self, VP: np.ndarray) -> None:
        """Renderiza o plano do chão."""
        GL.glUseProgram(self.ground_prog)
        set_uniform_mat4(self.ground_prog, "u_view_proj", VP)
        self.ground.draw()

    def _draw_dice(
        self,
        scene: RenderScene,
        VP: np.ndarray,
        cam_pos: np.ndarray,
    ) -> None:
        """Renderiza todos os dados."""
        GL.glUseProgram(self.dice_prog)

        # Uniforms globais (iguais para todos os dados)
        set_uniform_mat4(self.dice_prog,  "u_view_proj",   VP)
        set_uniform_vec3(self.dice_prog,  "u_light_dir",   self.lighting.normalized_dir())
        set_uniform_vec3(self.dice_prog,  "u_light_color", self.lighting.light_color)
        set_uniform_vec3(self.dice_prog,  "u_ambient",     self.lighting.ambient)
        set_uniform_float(self.dice_prog, "u_shininess",   self.lighting.shininess)
        set_uniform_vec3(self.dice_prog,  "u_cam_pos",     cam_pos)

        for gpu_obj, rd in zip(self.dice_gpu, scene.dice_renders):
            self._draw_single_die(gpu_obj, rd)

    def _draw_single_die(self, gpu: DiceGpuObject, rd: DiceRenderData) -> None:
        """Configura uniforms por-dado e executa o draw call."""
        M   = rd.model_mat                      # float32 (4,4)
        R33 = M[:3, :3]                         # parte rotação

        # Matriz de normais: para escala uniforme = transpose(inverse(R)) = R
        # (matriz de rotação é ortogonal, então R⁻¹ = Rᵀ, e Rᵀᵀ = R)
        normal_mat = R33.astype(np.float32)

        set_uniform_mat4(self.dice_prog,  "u_model",      M)
        set_uniform_mat3(self.dice_prog,  "u_normal_mat", normal_mat)
        set_uniform_vec3(self.dice_prog,  "u_dice_color", gpu.color)
        set_uniform_bool(self.dice_prog,  "u_highlight",  rd.is_resting)

        gpu.draw()

    # ── limpeza ──────────────────────────────────────────────────────

    def reload(self, scene: RenderScene, dice_types: list[str]) -> None:
        """
        Recarrega os VAOs/VBOs para um novo conjunto de dados,
        reutilizando os programas GLSL já compilados.

        Chamado por _roll_dice sem recriar o Renderer inteiro.
        """
        # Destrói apenas os objetos GPU dos dados (não o chão nem os shaders)
        for gpu in self.dice_gpu:
            gpu.delete()
        self.dice_gpu = []

        for rd, dtype in zip(scene.dice_renders, dice_types):
            self.dice_gpu.append(DiceGpuObject(rd, dtype))

    def delete(self) -> None:
        """Libera todos os recursos OpenGL."""
        for gpu in self.dice_gpu:
            gpu.delete()
        self.ground.delete()
        GL.glDeleteProgram(self.dice_prog)
        GL.glDeleteProgram(self.ground_prog)
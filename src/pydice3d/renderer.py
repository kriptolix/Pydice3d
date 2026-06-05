"""
renderer.py – Renderer OpenGL para Dados 3D com Glifos SDF

Responsabilidade: gerenciar VAO/VBO/EBO, configurar uniforms e executar
draw calls para cada dado e para o chão.

Mudanças em relação à versão anterior
──────────────────────────────────────
1. Layout de vértice expandido para 9 floats (+ uv + face_idx):
     attr 0: pos    (vec3, offset  0)
     attr 1: normal (vec3, offset 12)
     attr 2: uv     (vec2, offset 24)
     attr 3: face_idx (float, offset 32)  stride = 36 bytes

2. Novos uniforms por-dado:
     vec3  u_glyph_color      — cor do glifo
     int   u_face_glyphs[24]  — índice de glifo por face

3. DICE_VISUAL_SCALE: fator de escala visual independente da colisão física.
   Aplicado apenas na parte rotação/escala da model_mat, sem afetar PyBullet.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from OpenGL import GL

from pydice3d.camera import FixedCamera, OrbitalCamera
from pydice3d.render_data import RenderScene, DiceRenderData
from pydice3d.shaders import (
    build_dice_program, build_ground_program,
    set_uniform_mat4, set_uniform_mat3,
    set_uniform_vec3, set_uniform_vec4, set_uniform_float,
    set_uniform_bool, set_uniform_int_array,
    set_uniform_int, set_uniform_vec4_array,
    build_glyph_uv_table, build_symbol_uvs,
    MAX_FACES, GLYPH_NONE,
)


# ────────────────────────────────────────────────────────────────────────────
# Cores e escalas visuais por tipo de dado
# ────────────────────────────────────────────────────────────────────────────

DICE_COLORS: dict[str, tuple[float, float, float]] = {
    "d4":     (0.85, 0.25, 0.25),   # vermelho
    "d6":     (0.25, 0.55, 0.90),   # azul
    "d8":     (0.25, 0.75, 0.40),   # verde
    "d10":    (0.90, 0.65, 0.15),   # laranja
    "d12":    (0.70, 0.30, 0.85),   # roxo
    "d20":    (0.92, 0.92, 0.92),   # branco/prata
    "d100":   (0.90, 0.65, 0.15),   # laranja (par com d10)
    "dfudge": (0.20, 0.20, 0.20),   # quase preto
}
DEFAULT_DICE_COLOR = (0.7, 0.7, 0.7)

# Escala visual por tipo — afeta apenas a aparência, não a colisão PyBullet
DICE_VISUAL_SCALE: dict[str, float] = {
    "d4":     1.0,
    "d6":     1.00,
    "d8":     1.00,
    "d10":    1.20,
    "d12":    1.00,
    "d20":    1.10,
    "d100":   1.00,
    "dfudge": 1.00,
}


# ────────────────────────────────────────────────────────────────────────────
# DiceGpuObject — VAO/VBO/EBO de um único dado
# ────────────────────────────────────────────────────────────────────────────

class DiceGpuObject:
    """
    Objeto OpenGL para um único dado.

    Layout de vértice (stride = 9 × 4 = 36 bytes):
        attr 0: posição   (vec3,  offset  0)
        attr 1: normal    (vec3,  offset 12)
        attr 2: uv        (vec2,  offset 24)
        attr 3: face_idx  (float, offset 32)
    """

    def __init__(self, rd: DiceRenderData, dice_type: str) -> None:
        self.dice_type   = dice_type
        self.n_indices   = rd.n_indices
        self.color       = DICE_COLORS.get(dice_type, DEFAULT_DICE_COLOR)
        self.glyph_color = rd.glyph_color
        self.face_glyphs = _pad_glyphs(rd.face_glyphs)   # sempre MAX_FACES ints

        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        self.vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        vb = rd.vertex_buffer   # float32 (N, 9)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vb.nbytes, vb.tobytes(), GL.GL_STATIC_DRAW)

        stride = 9 * 4   # 9 floats × 4 bytes = 36

        # attr 0: posição
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                  GL.ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        # attr 1: normal
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                  GL.ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(1)
        # attr 2: uv
        GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                  GL.ctypes.c_void_p(24))
        GL.glEnableVertexAttribArray(2)
        # attr 3: face_idx
        GL.glVertexAttribPointer(3, 1, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                  GL.ctypes.c_void_p(32))
        GL.glEnableVertexAttribArray(3)

        self.ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        ib = rd.index_buffer
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, ib.nbytes, ib.tobytes(),
                        GL.GL_STATIC_DRAW)

        GL.glBindVertexArray(0)

    def draw(self) -> None:
        GL.glBindVertexArray(self.vao)
        GL.glDrawElements(GL.GL_TRIANGLES, self.n_indices, GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)

    def delete(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteBuffers(1, [self.vbo])
        GL.glDeleteBuffers(1, [self.ebo])


def _pad_glyphs(glyphs: list[int]) -> list[int]:
    """Garante lista de exatamente MAX_FACES ints, preenchendo com GLYPH_NONE."""
    padded = list(glyphs)[:MAX_FACES]
    while len(padded) < MAX_FACES:
        padded.append(GLYPH_NONE)
    return padded


# ────────────────────────────────────────────────────────────────────────────
# GroundPlane
# ────────────────────────────────────────────────────────────────────────────

class GroundPlane:
    HALF_SIZE = 50.0

    def __init__(self) -> None:
        s = self.HALF_SIZE
        y = 0.0
        verts = np.array([
            [-s, y, -s], [ s, y, -s], [ s, y,  s],
            [-s, y, -s], [ s, y,  s], [-s, y,  s],
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
# Carregamento da atlas de glifos
# ────────────────────────────────────────────────────────────────────────────

def _load_atlas_texture(png_path: str) -> int:
    """
    Carrega um PNG como textura OpenGL RGBA e retorna o ID da textura.

    Usa PIL/Pillow para decodificar o PNG. A textura é configurada com
    filtragem linear (GL_LINEAR) para suavização ao escalar os glifos.
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError as e:
        raise RuntimeError(
            "Pillow é necessário para carregar a atlas de glifos. "
            "Instale com: pip install Pillow"
        ) from e

    img = Image.open(png_path).convert("RGBA")
    img_data = np.array(img, dtype=np.uint8)

    tex_id = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    h, w = img_data.shape[:2]
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D, 0, GL.GL_RGBA,
        w, h, 0,
        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE,
        img_data.tobytes(),
    )
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return tex_id




@dataclass
class LightingParams:
    light_dir:   np.ndarray = field(
        default_factory=lambda: np.array([0.6, 1.0, 0.8], dtype=np.float32)
    )
    light_color: tuple = (1.0, 0.97, 0.90)
    shininess:   float = 64.0   # ambient removido — hemispheric lighting no shader

    def normalized_dir(self) -> np.ndarray:
        d = np.asarray(self.light_dir, dtype=float)
        n = np.linalg.norm(d)
        return (d / n).astype(np.float32) if n > 1e-8 else d


# ────────────────────────────────────────────────────────────────────────────
# Renderer principal
# ────────────────────────────────────────────────────────────────────────────

class Renderer:
    """
    Renderer OpenGL 3.3 Core Profile.

    Ciclo de vida
    ─────────────
    1. Criar após contexto OpenGL ativo:
           renderer = Renderer(scene, dice_types)
    2. A cada frame:
           scene.update(states, alpha)
           renderer.draw(scene, VP, cam_pos, width, height)
    3. Ao destruir:
           renderer.delete()
    """

    def __init__(
        self,
        scene:           RenderScene,
        dice_types:      list[str],
        lighting:        Optional[LightingParams] = None,
        atlas_png:       Optional[str] = None,
        atlas_json:      Optional[dict] = None,
        atlas_normal_png: Optional[str] = None,
    ) -> None:
        self.lighting   = lighting or LightingParams()
        self.debug_mode = 0   # 0=none, 1=collision, 2=overlay — controla highlight

        self.dice_prog   = build_dice_program()
        self.ground_prog = build_ground_program()

        self.dice_gpu: list[DiceGpuObject] = []
        for rd, dtype in zip(scene.dice_renders, dice_types):
            self.dice_gpu.append(DiceGpuObject(rd, dtype))

        self.ground = GroundPlane()

        # ── Atlas de glifos (diffuse) ────────────────────────────────────
        self._atlas_tex:    int = 0
        self._normal_tex:   int = 0
        self._glyph_uvs:    Optional[np.ndarray] = None
        self._uv_plus:      Optional[np.ndarray] = None
        self._uv_minus:     Optional[np.ndarray] = None

        if atlas_png and atlas_json:
            self._atlas_tex = _load_atlas_texture(atlas_png)
            self._glyph_uvs = build_glyph_uv_table(atlas_json)
            self._uv_plus, self._uv_minus = build_symbol_uvs(atlas_json)

        if atlas_normal_png:
            self._normal_tex = _load_atlas_texture(atlas_normal_png)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

    # ── draw call principal ──────────────────────────────────────────

    def draw(
        self,
        scene:    RenderScene,
        VP:       np.ndarray,
        cam_pos:  np.ndarray,
        width:    int,
        height:   int,
    ) -> None:
        GL.glViewport(0, 0, width, height)
        GL.glClearColor(0.0, 0.0, 0.0, 0.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self._draw_dice(scene, VP, cam_pos)

    def _draw_ground(self, VP: np.ndarray) -> None:
        GL.glUseProgram(self.ground_prog)
        set_uniform_mat4(self.ground_prog, "u_view_proj", VP)
        self.ground.draw()

    def _draw_dice(
        self,
        scene:   RenderScene,
        VP:      np.ndarray,
        cam_pos: np.ndarray,
    ) -> None:
        GL.glUseProgram(self.dice_prog)

        # Uniforms globais de iluminação
        set_uniform_mat4(self.dice_prog,  "u_view_proj",   VP)
        set_uniform_vec3(self.dice_prog,  "u_light_dir",   self.lighting.normalized_dir())
        set_uniform_vec3(self.dice_prog,  "u_light_color", self.lighting.light_color)
        set_uniform_float(self.dice_prog, "u_shininess",   self.lighting.shininess)
        set_uniform_vec3(self.dice_prog,  "u_cam_pos",     cam_pos)

        # Atlas diffuse — unidade de textura 0
        if self._atlas_tex:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._atlas_tex)
            set_uniform_int(self.dice_prog, "u_glyph_atlas", 0)

        # Atlas normal map — unidade de textura 1
        if self._normal_tex:
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._normal_tex)
            set_uniform_int(self.dice_prog, "u_glyph_normal", 1)

        # Tabela de UV dos dígitos e símbolos
        if self._glyph_uvs is not None:
            set_uniform_vec4_array(self.dice_prog, "u_glyph_uvs", self._glyph_uvs)
        if self._uv_plus is not None:
            set_uniform_vec4(self.dice_prog, "u_glyph_uv_plus",  self._uv_plus)
        if self._uv_minus is not None:
            set_uniform_vec4(self.dice_prog, "u_glyph_uv_minus", self._uv_minus)

        for gpu_obj, rd in zip(self.dice_gpu, scene.dice_renders):
            self._draw_single_die(gpu_obj, rd)

    def _draw_single_die(self, gpu: DiceGpuObject, rd: DiceRenderData) -> None:
        M  = rd.model_mat.copy()
        vs = DICE_VISUAL_SCALE.get(gpu.dice_type, 1.0)
        M[:3, :3] *= vs
        normal_mat = M[:3, :3].astype(np.float32)

        # Highlight amarelo só quando um modo de debug estiver ativo
        highlight = rd.is_resting and (self.debug_mode != 0)

        set_uniform_mat4(self.dice_prog,  "u_model",       M)
        set_uniform_mat3(self.dice_prog,  "u_normal_mat",  normal_mat)
        set_uniform_vec3(self.dice_prog,  "u_dice_color",  gpu.color)
        set_uniform_vec3(self.dice_prog,  "u_glyph_color", gpu.glyph_color)
        set_uniform_bool(self.dice_prog,  "u_highlight",   highlight)
        set_uniform_int_array(self.dice_prog, "u_face_glyphs", gpu.face_glyphs)

        gpu.draw()

    # ── reload e delete ──────────────────────────────────────────────

    def reload(self, scene: RenderScene, dice_types: list[str]) -> None:
        """Recria os objetos GPU dos dados sem recarregar a atlas."""
        for gpu in self.dice_gpu:
            gpu.delete()
        self.dice_gpu = []
        for rd, dtype in zip(scene.dice_renders, dice_types):
            self.dice_gpu.append(DiceGpuObject(rd, dtype))

    def delete(self) -> None:
        for gpu in self.dice_gpu:
            gpu.delete()
        self.ground.delete()
        for tex in (self._atlas_tex, self._normal_tex):
            if tex:
                GL.glDeleteTextures(1, [tex])
        self._atlas_tex = self._normal_tex = 0
        GL.glDeleteProgram(self.dice_prog)
        GL.glDeleteProgram(self.ground_prog)
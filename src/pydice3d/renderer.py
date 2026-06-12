"""
renderer.py – OpenGL renderer for 3D data with SDF glyphs. Manages VAO/VBO/EBO, 
configures uniforms, and executes draw calls for each data point and for the floor.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import json
from importlib.resources import files

from OpenGL import GL

from pydice3d.scene import RenderScene, DiceRenderData
from pydice3d.scene import DICE_THEMES, DEFAULT_DICE_COLOR
from pydice3d.shaders import (
    build_dice_program, build_ground_program,
    set_uniform_mat4, set_uniform_mat3,
    set_uniform_vec3, set_uniform_vec4, set_uniform_float,
    set_uniform_bool, set_uniform_int_array,
    set_uniform_int, set_uniform_vec4_array,
    build_glyph_uv_table, build_symbol_uvs,
    MAX_FACES, GLYPH_NONE,
)

_ATLAS_DIR = files("pydice3d.assets").joinpath("atlas")
_ATLAS_NPY = str(_ATLAS_DIR.joinpath("atlas.npy"))
_ATLAS_JSON = _ATLAS_DIR.joinpath("atlas.json")

DICE_VISUAL_SCALE: dict[str, float] = {
    "d4":     1.0,
    "d6":     1.00,
    "d8":     1.00,
    "d10":    1.50,
    "d12":    0.90,
    "d20":    1.00,
    "d100":   1.50,
    "df": 1.00,
}


class DiceGpuObject:
    """
    OpenGL object for a single data point.

    Vertex layout (stride = 9 × 4 = 36 bytes):

    attr 0: position (vec3, offset 0)
    attr 1: normal (vec3, offset 12)
    attr 2: uv (vec2, offset 24)
    attr 3: face_idx (float, offset 32)
    """

    def __init__(self, rd: DiceRenderData, dice_type: str, theme: str = "light") -> None:
        self.dice_type = dice_type
        self.n_indices = rd.n_indices
        self.color = DICE_THEMES[theme].dice_color if theme in DICE_THEMES else DEFAULT_DICE_COLOR
        self.glyph_color = rd.glyph_color
        self.face_glyphs = _pad_glyphs(
            rd.face_glyphs)   # sempre MAX_FACES ints

        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        self.vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        vb = rd.vertex_buffer   # float32 (N, 9)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vb.nbytes,
                        vb.tobytes(), GL.GL_STATIC_DRAW)

        stride = 9 * 4   # 9 floats × 4 bytes = 36

        # attr 0: position
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
        GL.glDrawElements(GL.GL_TRIANGLES, self.n_indices,
                          GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)

    def delete(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteBuffers(1, [self.vbo])
        GL.glDeleteBuffers(1, [self.ebo])


def _pad_glyphs(glyphs: list[int]) -> list[int]:
    """Ensures the list contains exactly MAX_FACES ints, padding it with GLYPH_NONE."""
    padded = list(glyphs)[:MAX_FACES]
    while len(padded) < MAX_FACES:
        padded.append(GLYPH_NONE)
    return padded


class GroundPlane:
    HALF_SIZE = 50.0

    def __init__(self) -> None:
        s = self.HALF_SIZE
        y = 0.0
        verts = np.array([
            [-s, y, -s], [s, y, -s], [s, y,  s],
            [-s, y, -s], [s, y,  s], [-s, y,  s],
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


def _load_atlas_texture(npy_path: str) -> int:

    img_data = np.load(npy_path)
    if img_data.ndim != 3 or img_data.shape[2] != 4:
        raise RuntimeError(
            f"atlas.npy must have shape (H, W, 4) uint8, "
            f"received: {img_data.shape} {img_data.dtype}"
        )
    img_data = np.ascontiguousarray(img_data, dtype=np.uint8)
    h, w = img_data.shape[:2]

    tex_id = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
    # MSDF: sem mipmaps, filtragem linear simples
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D, 0, GL.GL_RGBA,
        w, h, 0,
        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE,
        img_data.tobytes(),
    )
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return tex_id


@dataclass
class LightingParams:
    light_dir:   np.ndarray = field(
        default_factory=lambda: np.array([0.6, 1.0, 0.8], dtype=np.float32)
    )
    light_color: tuple = (1.0, 0.97, 0.90)
    shininess:   float = 64.0

    def normalized_dir(self) -> np.ndarray:
        d = np.asarray(self.light_dir, dtype=float)
        n = np.linalg.norm(d)
        return (d / n).astype(np.float32) if n > 1e-8 else d


class Renderer:
    """
    Renderer OpenGL 3.3 Core Profile.  
    """

    def __init__(
        self,
        scene:      RenderScene,
        dice_types: list[str],
        lighting:   Optional[LightingParams] = None,
        theme:      str = "light",
    ) -> None:

        self.lighting = lighting or LightingParams()
        self.debug_mode = 0
        self._theme = theme

        self.dice_prog = build_dice_program()
        self.ground_prog = build_ground_program()

        self.dice_gpu: list[DiceGpuObject] = []
        for rd, dtype in zip(scene.dice_renders, dice_types):
            self.dice_gpu.append(DiceGpuObject(rd, dtype, self._theme))

        self.ground = GroundPlane()

        self._atlas_tex:  int = 0
        self._glyph_uvs:  Optional[np.ndarray] = None
        self._uv_plus:    Optional[np.ndarray] = None
        self._uv_minus:   Optional[np.ndarray] = None

        try:
            with open(_ATLAS_JSON, "r", encoding="utf-8") as f:
                self._atlas_json = json.load(f)
        except Exception as e:
            print(f"Could not load atlas.json: {e}")

        if _ATLAS_NPY and _ATLAS_JSON:
            self._atlas_tex = _load_atlas_texture(_ATLAS_NPY)
            self._glyph_uvs = build_glyph_uv_table(self._atlas_json)
            self._uv_plus, self._uv_minus = build_symbol_uvs(self._atlas_json)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

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

        set_uniform_mat4(self.dice_prog,  "u_view_proj",   VP)
        set_uniform_vec3(self.dice_prog,  "u_light_dir",
                         self.lighting.normalized_dir())
        set_uniform_vec3(self.dice_prog,  "u_light_color",
                         self.lighting.light_color)
        set_uniform_float(self.dice_prog, "u_shininess",
                          self.lighting.shininess)
        set_uniform_vec3(self.dice_prog,  "u_cam_pos",     cam_pos)

        if self._atlas_tex:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._atlas_tex)
            set_uniform_int(self.dice_prog, "u_glyph_atlas", 0)

        if self._glyph_uvs is not None:
            set_uniform_vec4_array(
                self.dice_prog, "u_glyph_uvs", self._glyph_uvs)
        if self._uv_plus is not None:
            set_uniform_vec4(self.dice_prog, "u_glyph_uv_plus",  self._uv_plus)
        if self._uv_minus is not None:
            set_uniform_vec4(
                self.dice_prog, "u_glyph_uv_minus", self._uv_minus)

        for gpu_obj, rd in zip(self.dice_gpu, scene.dice_renders):
            self._draw_single_die(gpu_obj, rd)

    def _draw_single_die(self, gpu: DiceGpuObject, rd: DiceRenderData) -> None:
        M = rd.model_mat.copy()
        vs = DICE_VISUAL_SCALE.get(gpu.dice_type, 1.0)
        M[:3, :3] *= vs
        normal_mat = M[:3, :3].astype(np.float32)

        # Yellow highlight only when a debug mode is active.
        highlight = rd.is_resting and (self.debug_mode != 0)

        set_uniform_mat4(self.dice_prog,  "u_model",       M)
        set_uniform_mat3(self.dice_prog,  "u_normal_mat",  normal_mat)
        set_uniform_vec3(self.dice_prog,  "u_dice_color",  gpu.color)
        set_uniform_vec3(self.dice_prog,  "u_glyph_color", gpu.glyph_color)
        set_uniform_bool(self.dice_prog,  "u_highlight",   highlight)
        set_uniform_int_array(self.dice_prog, "u_face_glyphs", gpu.face_glyphs)

        gpu.draw()

    @property
    def theme(self) -> str:
        return self._theme

    @theme.setter
    def theme(self, value: str) -> None:

        if value not in DICE_THEMES:
            raise ValueError(
                f"Invalid theme: {value!r}. Use: {list(DICE_THEMES)}")
        self._theme = value
        theme = DICE_THEMES[value]
        for gpu in self.dice_gpu:
            gpu.color = theme.dice_color
            gpu.glyph_color = theme.glyph_color

    # ── reload e delete ──────────────────────────────────────────────

    def reload(self, scene: RenderScene, dice_types: list[str]) -> None:
        """Recreates GPU objects from the data without reloading the atlas."""
        for gpu in self.dice_gpu:
            gpu.delete()
        self.dice_gpu = []
        for rd, dtype in zip(scene.dice_renders, dice_types):
            self.dice_gpu.append(DiceGpuObject(rd, dtype, self._theme))

    def delete(self) -> None:
        for gpu in self.dice_gpu:
            gpu.delete()
        self.ground.delete()
        if self._atlas_tex:
            GL.glDeleteTextures(1, [self._atlas_tex])
        self._atlas_tex = 0
        GL.glDeleteProgram(self.dice_prog)
        GL.glDeleteProgram(self.ground_prog)

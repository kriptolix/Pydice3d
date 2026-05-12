"""
glarena.py — Área OpenGL de renderização dos dados (GTK4 GLArea).

Responsabilidades:
  - Ciclo de vida OpenGL (realize / unrealize / render / resize)
  - Orquestração de carregamento de assets (via asset_loader)
  - Loop de simulação (tick via GLib.timeout)
  - Despacho de leitura de resultados
"""

import gi
from gi.repository import Gtk, GLib

from OpenGL.GL import *
import numpy as np

from physics import PhysicsWorld, DICE_TARGET_SIZE, TRAY_H 
from dice_reader import read_all_dice
from shaders import make_program, upload_mesh, load_texture
from geometrics import expand_obj_with_uv_tangent, build_floor_mesh
from matrices import mat4_persp, mat4_lookat, mat4_from_bullet
from asset_loader import (
    get_obj_path, get_texture_paths, get_floor_texture_paths, compute_dice_scale,
)

try:
    from sound import SoundManager
except ImportError:
    class SoundManager:          # noqa: stub silencioso se módulo ausente
        def play_roll(self): pass
        def play_settle(self): pass


class DiceGLArea(Gtk.GLArea):

    CAM_EYE    = np.array([0.0, 12.0, 0.0],  dtype=np.float32)  # diretamente acima
    CAM_CENTER = np.array([0.0, 0.0,  0.0],  dtype=np.float32)  # olhando para o centro
    CAM_UP     = np.array([0.0, 0.0, -1.0],  dtype=np.float32)  # "cima" aponta para Z-

    def __init__(self, physics: PhysicsWorld):
        super().__init__()
        self.physics = physics
        self.set_required_version(3, 3)
        self.set_has_depth_buffer(True)

        self.prog_pbr = None

        self._dice_vaos:     dict[str, tuple[int, int]] = {}
        self._dice_textures: dict[str, dict]            = {}

        self.floor_vao       = None
        self.floor_vcount    = 0
        self._floor_textures = {"base": None, "normal": None}

        self._current_type = "d6"
        self.width, self.height = 600, 500
        self.timer_id   = None
        self.simulating = False
        self.sound      = SoundManager()

        self.connect("realize",   self._on_realize)
        self.connect("unrealize", self._on_unrealize)
        self.connect("render",    self._on_render)
        self.connect("resize",    self._on_resize)

    # ------------------------------------------------------------------
    # Inicialização OpenGL
    # ------------------------------------------------------------------

    def _on_realize(self, area):
        self.make_current()
        if self.get_error():
            print("GLArea error:", self.get_error().message)
            return

        self.prog_pbr = make_program()
        self._load_dice_type("d6")
        self._load_floor()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def _load_floor(self):
        from physics import TRAY_W, TRAY_D
        fp, fn, fuv, ftan = build_floor_mesh(TRAY_W, TRAY_H, TRAY_D, uv_tile=4.0)
        self.floor_vao, self.floor_vcount = upload_mesh(fp, fn, fuv, ftan)

        tex_paths = get_floor_texture_paths()
        for key, srgb in (("base", True), ("normal", False)):
            p = tex_paths[key]
            if p:
                self._floor_textures[key] = load_texture(p, srgb=srgb)
                print(f"[floor] {key} carregado: {p}")

    def _load_dice_type(self, dice_type: str):
        if dice_type in self._dice_vaos:
            return

        obj_path = get_obj_path(dice_type)
        if obj_path is None:
            return

        from geometrics import load_obj
        import numpy as np

        positions, uvs, faces = load_obj(obj_path)
        pos_arr    = np.array(positions, dtype=np.float32)
        dice_scale = compute_dice_scale(pos_arr, DICE_TARGET_SIZE, dice_type)
        print(f"[{dice_type}] scale={dice_scale:.4f}")

        self.physics.set_dice_scale(dice_scale)

        pos_flat, nor_flat, uv_flat, tan_flat = expand_obj_with_uv_tangent(
            positions, uvs, faces, scale=dice_scale
        )
        vao, vcount = upload_mesh(pos_flat, nor_flat, uv_flat, tan_flat)
        self._dice_vaos[dice_type] = (vao, vcount)

        tex_paths = get_texture_paths(dice_type)
        self._dice_textures[dice_type] = {
            "base":   load_texture(tex_paths["base"],   srgb=True)
                      if tex_paths["base"]   else None,
            "normal": load_texture(tex_paths["normal"], srgb=False)
                      if tex_paths["normal"] else None,
        }
        loaded = [k for k, v in self._dice_textures[dice_type].items() if v]
        print(f"[{dice_type}] Texturas: {', '.join(loaded) or 'nenhuma (fallback cor)'}")

    # ------------------------------------------------------------------
    # Destruição OpenGL
    # ------------------------------------------------------------------

    def _on_unrealize(self, area):
        self.make_current()
        for vao, _ in self._dice_vaos.values():
            glDeleteVertexArrays(1, [vao])
        if self.floor_vao:
            glDeleteVertexArrays(1, [self.floor_vao])
        for texs in [*self._dice_textures.values(), self._floor_textures]:
            for tex in texs.values():
                if tex:
                    glDeleteTextures(1, [tex])
        if self.prog_pbr:
            glDeleteProgram(self.prog_pbr)

    def _on_resize(self, area, w, h):
        self.width, self.height = w, h

    # ------------------------------------------------------------------
    # Helpers de uniform / textura
    # ------------------------------------------------------------------

    def _loc(self, name: str) -> int:
        return glGetUniformLocation(self.prog_pbr, name)

    def _set_mvp_uniforms(self, mvp: np.ndarray, mv: np.ndarray):
        glUniformMatrix4fv(self._loc("uMVP"),       1, GL_TRUE, mvp.flatten())
        glUniformMatrix4fv(self._loc("uModelView"), 1, GL_TRUE, mv.flatten())
        nm = mv[:3, :3].copy()
        try:
            nm = np.linalg.inv(nm).T
        except np.linalg.LinAlgError:
            pass
        glUniformMatrix3fv(self._loc("uNormalMat"), 1, GL_TRUE, nm.flatten())

    def _bind_textures(self, tex_base, tex_normal, fallback_color, alpha=1.0):
        glActiveTexture(GL_TEXTURE0)
        if tex_base:
            glBindTexture(GL_TEXTURE_2D, tex_base)
            glUniform1i(self._loc("uHasBase"), 1)
        else:
            glBindTexture(GL_TEXTURE_2D, 0)
            glUniform1i(self._loc("uHasBase"), 0)

        glActiveTexture(GL_TEXTURE1)
        if tex_normal:
            glBindTexture(GL_TEXTURE_2D, tex_normal)
            glUniform1i(self._loc("uHasNormal"), 1)
        else:
            glBindTexture(GL_TEXTURE_2D, 0)
            glUniform1i(self._loc("uHasNormal"), 0)

        glUniform1i(self._loc("uTexBase"),    0)
        glUniform1i(self._loc("uTexNormal"), 1)
        # Luz principal: lateral direita/frente, baixa — Y=3 vs X=12 → ~14° acima do horizonte.
        # Realça faces laterais e números sem achatar o volume do dado.
        glUniform3f(self._loc("uLightPos"),  12.0, 3.0, 8.0)
        # Luz de preenchimento: lado oposto, mais fraca — evita sombras totalmente negras
        # e elimina a ambiguidade visual de aresta vs face no D20/D8.
        glUniform3f(self._loc("uFillPos"),  -10.0, 2.0, -6.0)
        glUniform3f(self._loc("uColor"), *fallback_color)
        glUniform1f(self._loc("uAlpha"), alpha)

    # ------------------------------------------------------------------
    # Renderização
    # ------------------------------------------------------------------

    def _on_render(self, area, context):
        glClearColor(0.0, 0.0, 0.0, 0.0)   # totalmente transparente
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if not self.prog_pbr:
            return True

        glUseProgram(self.prog_pbr)

        aspect = self.width / max(self.height, 1)
        proj   = mat4_persp(45.0, aspect, 0.1, 50.0)
        view   = mat4_lookat(self.CAM_EYE, self.CAM_CENTER, self.CAM_UP)

        # Piso
        '''floor_model = np.eye(4, dtype=np.float32)
        floor_model[1, 3] = -TRAY_H / 2
        mv  = view @ floor_model
        mvp = proj @ mv
        self._set_mvp_uniforms(mvp, mv)
        self._bind_textures(
            self._floor_textures["base"],
            self._floor_textures["normal"],
            fallback_color=(0.55, 0.35, 0.18),
        )
        glBindVertexArray(self.floor_vao)
        glDrawArrays(GL_TRIANGLES, 0, self.floor_vcount)'''

        # Dados
        for dtype, (dice_vao, dice_vcount) in self._dice_vaos.items():
            texs = self._dice_textures.get(dtype, {})
            self._bind_textures(
                texs.get("base"),
                texs.get("normal"),
                fallback_color=(0.85, 0.18, 0.12),
            )
            
            for pos, orn in self.physics.get_transforms_for_type(dtype):
                model = mat4_from_bullet(pos, orn)
                mv    = view @ model
                mvp   = proj @ mv
                self._set_mvp_uniforms(mvp, mv)
                glBindVertexArray(dice_vao)
                glDrawArrays(GL_TRIANGLES, 0, dice_vcount)

        glBindVertexArray(0)
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
        return True

    # ------------------------------------------------------------------
    # Controle de simulação
    # ------------------------------------------------------------------

    def start_simulation(self, n_dice: int = 1, dice_type: str = "d6"):
        self.make_current()
        
        '''for dt in ["d4", "d6", "d8", "d10", "d12", "d20"]:
            self._load_dice_type(dt)

        self.physics.remove_all_dice()
        for dt in ["d4", "d6", "d8", "d10", "d12", "d20"]:
            self.physics.add_dice(dt)
        
        self._current_type = "d6"
            '''

        self._load_dice_type(dice_type)
        self._current_type = dice_type

        self.physics.remove_all_dice()
        for _ in range(n_dice):
            self.physics.add_dice(dice_type)
        
        self.simulating = True
        self.sound.play_roll()
        if self.timer_id:
            GLib.source_remove(self.timer_id)
        self.timer_id = GLib.timeout_add(12, self._tick)

    def _tick(self) -> bool:
        if self.simulating:
            self.physics.step()
            if self.physics.all_sleeping():
                self.simulating = False
                read_all_dice(
                    self._current_type,
                    self.physics.dice_ids,
                    self.physics.client,
                )
                self.sound.play_settle()

        self.queue_render()
        return True

    def stop_timer(self):
        if self.timer_id:
            GLib.source_remove(self.timer_id)
            self.timer_id = None
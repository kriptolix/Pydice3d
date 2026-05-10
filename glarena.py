import gi
from gi.repository import Gtk, GLib

from OpenGL.GL import *
import numpy as np
import os

from originals.physics import PhysicsWorld, DICE_TARGET_SIZE, TRAY_D, TRAY_H, TRAY_W
from originals.shaders import make_program, upload_mesh, load_texture
from originals.geometrics import (build_flat_box, expand_flat_shading, load_obj,
                        expand_obj_with_uv_tangent)
from originals.matrices import mat4_persp, mat4_lookat, mat4_from_bullet
from originals.dicereader import read_all_dice
from originals.sound import SoundManager


# ---------------------------------------------------------------------------
# Assets dos dados
# ---------------------------------------------------------------------------
DICE_ASSETS = {
    "d4":  {"obj": "assets/d4/d4.obj",   "tex": "assets/d4"},
    "d6":  {"obj": "assets/d6/d6.obj", "tex": "assets/d6"},
    "d8":  {"obj": "assets/d8/d8.obj",   "tex": "assets/d8"},
    "d10": {"obj": "assets/d10/d10.obj", "tex": "assets/d10"},
    "d12": {"obj": "assets/d12/d12.obj", "tex": "assets/d12"},
    "d20": {"obj": "assets/d20/d20.obj", "tex": "assets/d20"},
}

TEX_BASE   = "DefaultMaterial_Base_color.png"
TEX_NORMAL = "DefaultMaterial_Normal_DirectX.png"

# Textura do piso — coloque seu PNG de madeira aqui.
FLOOR_TEX_BASE   = "assets/floor/wood_base_color.png"
FLOOR_TEX_NORMAL = "assets/floor/wood_normal_directx.png"


def _find_tex(folder, filename):
    p = os.path.join(folder, filename)
    return p if os.path.isfile(p) else None


def _build_floor_with_uv(w, h, d, uv_tile=4.0):
    """
    Constrói a caixa do piso com UV tiled na face superior (Y+)
    e UV simples nas faces laterais.

    uv_tile : quantas vezes a textura repete nos 'w' metros da bandeja.
              Ajuste ao tamanho real do seu tile de madeira.

    Retorna (pos_flat, nor_flat, uv_flat, tan_flat) — arrays float32.
    """
    hw, hh, hd = w / 2, h / 2, d / 2
    t = uv_tile

    rows = []   # cada linha: [x,y,z, nx,ny,nz, u,v, tx,ty,tz]

    def add_tri(verts3, n, uvs3):
        """Empacota 3 vértices de um triângulo calculando a tangente."""
        p0, p1, p2 = [np.array(v, dtype=np.float64) for v in verts3]
        uv0, uv1, uv2 = [np.array(u, dtype=np.float64) for u in uvs3]
        e1, e2   = p1 - p0, p2 - p0
        du1, dv1 = uv1 - uv0
        du2, dv2 = uv2 - uv0
        denom = du1 * dv2 - du2 * dv1
        if abs(denom) > 1e-8:
            f = 1.0 / denom
            t_vec = f * (dv2 * e1 - dv1 * e2)
            tl = np.linalg.norm(t_vec)
            t_vec = t_vec / tl if tl > 1e-8 else np.array([1., 0., 0.])
        else:
            t_vec = np.array([1., 0., 0.])
        nn = np.array(n, dtype=np.float64)
        for p, uv in zip([p0, p1, p2], [uv0, uv1, uv2]):
            rows.append([*p, *nn, *uv, *t_vec])

    # Face +Y — topo (recebe UV tiled)
    add_tri([[-hw, hh,-hd],[hw, hh,-hd],[hw, hh, hd]], [0,1,0],
            [[0,0],[t,0],[t,t]])
    add_tri([[-hw, hh,-hd],[hw, hh, hd],[-hw, hh, hd]], [0,1,0],
            [[0,0],[t,t],[0,t]])

    # Face -Y — base
    add_tri([[-hw,-hh,-hd],[-hw,-hh, hd],[ hw,-hh, hd]], [0,-1,0],
            [[0,0],[0,1],[1,1]])
    add_tri([[-hw,-hh,-hd],[ hw,-hh, hd],[ hw,-hh,-hd]], [0,-1,0],
            [[0,0],[1,1],[1,0]])

    # Face -Z
    add_tri([[-hw,-hh,-hd],[ hw,-hh,-hd],[ hw, hh,-hd]], [0,0,-1],
            [[0,0],[1,0],[1,1]])
    add_tri([[-hw,-hh,-hd],[ hw, hh,-hd],[-hw, hh,-hd]], [0,0,-1],
            [[0,0],[1,1],[0,1]])

    # Face +Z
    add_tri([[-hw,-hh, hd],[-hw, hh, hd],[ hw, hh, hd]], [0,0,1],
            [[0,0],[0,1],[1,1]])
    add_tri([[-hw,-hh, hd],[ hw, hh, hd],[ hw,-hh, hd]], [0,0,1],
            [[0,0],[1,1],[1,0]])

    # Face -X
    add_tri([[-hw,-hh,-hd],[-hw, hh,-hd],[-hw, hh, hd]], [-1,0,0],
            [[0,0],[0,1],[1,1]])
    add_tri([[-hw,-hh,-hd],[-hw, hh, hd],[-hw,-hh, hd]], [-1,0,0],
            [[0,0],[1,1],[1,0]])

    # Face +X
    add_tri([[ hw,-hh,-hd],[ hw,-hh, hd],[ hw, hh, hd]], [1,0,0],
            [[0,0],[1,0],[1,1]])
    add_tri([[ hw,-hh,-hd],[ hw, hh, hd],[ hw, hh,-hd]], [1,0,0],
            [[0,0],[1,1],[0,1]])

    data = np.array(rows, dtype=np.float32)  # shape: (N, 11)
    return (
        np.ascontiguousarray(data[:, 0:3].flatten()),
        np.ascontiguousarray(data[:, 3:6].flatten()),
        np.ascontiguousarray(data[:, 6:8].flatten()),
        np.ascontiguousarray(data[:, 8:11].flatten()),
    )


class DiceGLArea(Gtk.GLArea):

    CAM_EYE    = np.array([0.0, 9.0, 8.0],  dtype=np.float32)
    CAM_CENTER = np.array([0.0, 0.0, 0.0],  dtype=np.float32)
    CAM_UP     = np.array([0.0, 1.0, 0.0],  dtype=np.float32)

    def __init__(self, physics: PhysicsWorld):
        super().__init__()
        self.physics = physics
        self.set_required_version(3, 3)
        self.set_has_depth_buffer(True)

        self.prog_pbr = None

        self._dice_vaos: dict[str, tuple[int, int]] = {}
        self._dice_textures: dict[str, dict] = {}

        self.floor_vao    = None
        self.floor_vcount = 0
        self._floor_textures: dict = {"base": None, "normal": None}

        self._current_type = "d6"

        self.width    = 600
        self.height   = 500
        self.timer_id = None
        self.simulating = False

        self.sound = SoundManager()

        self.connect('realize',   self._on_realize)
        self.connect('unrealize', self._on_unrealize)
        self.connect('render',    self._on_render)
        self.connect('resize',    self._on_resize)

    # ------------------------------------------------------------------
    def _on_realize(self, area):
        self.make_current()
        if self.get_error():
            print("GLArea error:", self.get_error().message)
            return

        # Um único programa PBR para dados e piso
        self.prog_pbr = make_program()

        self._load_dice_type("d6")
        self._load_floor()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def _load_floor(self):
        fp, fn, fuv, ftan = _build_floor_with_uv(TRAY_W, TRAY_H, TRAY_D, uv_tile=4.0)
        self.floor_vao, self.floor_vcount = upload_mesh(fp, fn, fuv, ftan)

        if os.path.isfile(FLOOR_TEX_BASE):
            self._floor_textures["base"] = load_texture(FLOOR_TEX_BASE, srgb=True)
            print("[floor] base_color carregado")
        else:
            print(f"[floor] base_color não encontrado — {FLOOR_TEX_BASE}")

        if os.path.isfile(FLOOR_TEX_NORMAL):
            self._floor_textures["normal"] = load_texture(FLOOR_TEX_NORMAL, srgb=False)
            print("[floor] normal_map carregado")
        else:
            print(f"[floor] normal_map não encontrado — {FLOOR_TEX_NORMAL}")

    def _load_dice_type(self, dice_type: str):
        if dice_type in self._dice_vaos:
            return

        assets = DICE_ASSETS.get(dice_type)
        if assets is None:
            print(f"[glarena] Tipo desconhecido: {dice_type}")
            return

        obj_path = assets["obj"]
        tex_dir  = assets["tex"]

        if not os.path.isfile(obj_path):
            print(f"[glarena] OBJ não encontrado: {obj_path}")
            return

        positions, uvs, faces = load_obj(obj_path)
        positions = np.array(positions, dtype=np.float32)

        obj_size   = float(np.max(positions) - np.min(positions))
        dice_scale = DICE_TARGET_SIZE / obj_size if obj_size > 1e-8 else 1.0
        print(f"[{dice_type}] OBJ size: {obj_size:.4f}  scale: {dice_scale:.4f}  "
              f"→ {obj_size * dice_scale:.3f}m")

        self.physics.set_dice_scale(dice_scale)

        pos_flat, nor_flat, uv_flat, tan_flat = expand_obj_with_uv_tangent(
            positions.tolist(), uvs, faces, scale=dice_scale
        )
        vao, vcount = upload_mesh(pos_flat, nor_flat, uv_flat, tan_flat)
        self._dice_vaos[dice_type] = (vao, vcount)

        tex_base_path   = _find_tex(tex_dir, TEX_BASE)
        tex_normal_path = _find_tex(tex_dir, TEX_NORMAL)

        self._dice_textures[dice_type] = {
            "base":   load_texture(tex_base_path,   srgb=True)  if tex_base_path   else None,
            "normal": load_texture(tex_normal_path, srgb=False) if tex_normal_path else None,
        }
        loaded = [k for k, v in self._dice_textures[dice_type].items() if v]
        print(f"[{dice_type}] Texturas: {', '.join(loaded) or 'nenhuma (fallback cor)'}")

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
    def _set_mvp_uniforms(self, mvp, mv):
        def L(n): return glGetUniformLocation(self.prog_pbr, n)
        glUniformMatrix4fv(L("uMVP"),       1, GL_TRUE, mvp.flatten())
        glUniformMatrix4fv(L("uModelView"), 1, GL_TRUE, mv.flatten())
        nm = mv[:3, :3].copy()
        try:
            nm = np.linalg.inv(nm).T
        except np.linalg.LinAlgError:
            pass
        glUniformMatrix3fv(L("uNormalMat"), 1, GL_TRUE, nm.flatten())

    def _bind_textures(self, tex_base, tex_normal, fallback_color, alpha=1.0):
        def L(n): return glGetUniformLocation(self.prog_pbr, n)

        glActiveTexture(GL_TEXTURE0)
        if tex_base:
            glBindTexture(GL_TEXTURE_2D, tex_base)
            glUniform1i(L("uHasBase"), 1)
        else:
            glBindTexture(GL_TEXTURE_2D, 0)
            glUniform1i(L("uHasBase"), 0)

        glActiveTexture(GL_TEXTURE1)
        if tex_normal:
            glBindTexture(GL_TEXTURE_2D, tex_normal)
            glUniform1i(L("uHasNormal"), 1)
        else:
            glBindTexture(GL_TEXTURE_2D, 0)
            glUniform1i(L("uHasNormal"), 0)

        glUniform1i(L("uTexBase"),   0)
        glUniform1i(L("uTexNormal"), 1)
        glUniform3f(L("uLightPos"), 3.0, 6.0, 5.0)
        glUniform3f(L("uColor"), *fallback_color)
        glUniform1f(L("uAlpha"), alpha)

    # ------------------------------------------------------------------
    def _on_render(self, area, context):
        glClearColor(0.08, 0.08, 0.12, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if not self.prog_pbr:
            return True

        glUseProgram(self.prog_pbr)

        aspect = self.width / max(self.height, 1)
        proj   = mat4_persp(45.0, aspect, 0.1, 50.0)
        view   = mat4_lookat(self.CAM_EYE, self.CAM_CENTER, self.CAM_UP)

        # ---- Piso ----
        floor_model = np.eye(4, dtype=np.float32)
        floor_model[1, 3] = -TRAY_H / 2
        mv  = view @ floor_model
        mvp = proj @ mv
        self._set_mvp_uniforms(mvp, mv)
        self._bind_textures(
            self._floor_textures["base"],
            self._floor_textures["normal"],
            fallback_color=(0.55, 0.35, 0.18),  # marrom madeira
        )
        glBindVertexArray(self.floor_vao)
        glDrawArrays(GL_TRIANGLES, 0, self.floor_vcount)

        # ---- Dados ----
        dtype = self._current_type
        if dtype in self._dice_vaos:
            dice_vao, dice_vcount = self._dice_vaos[dtype]
            texs = self._dice_textures.get(dtype, {})
            self._bind_textures(
                texs.get("base"),
                texs.get("normal"),
                fallback_color=(0.85, 0.18, 0.12),
            )
            for pos, orn in self.physics.get_transforms():
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
    def start_simulation(self, n_dice: int = 1, dice_type: str = "d6"):
        self.make_current()
        self._load_dice_type(dice_type)
        self._current_type = dice_type

        self.physics.remove_all_dice()
        for _ in range(n_dice):
            self.physics.add_dice(dice_type)

        self.simulating = True
        self.sound.play_roll()

        if self.timer_id:
            GLib.source_remove(self.timer_id)
        self.timer_id = GLib.timeout_add(16, self._tick)

    def _tick(self):
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
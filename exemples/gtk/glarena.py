"""
glarena.py – GTK4 GLArea: loop de física + renderização

Responsabilidade: conectar o ciclo GTK (realize/render/unrealize) à simulação
PyBullet e ao Renderer OpenGL. Expõe start_simulation() para a AppWindow.

Modos de debug
──────────────
DEBUG_NONE      : renderização normal
DEBUG_COLLISION : só wireframe do hull de colisão (sem mesh visual)
DEBUG_OVERLAY   : mesh visual + wireframe de colisão sobrepostos

Alternância via propriedade `debug_mode` ou teclas:
    N  → normal
    C  → só colisão
    O  → overlay (colisão sobre visual)
"""

from __future__ import annotations

import math
import numpy as np

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib

from OpenGL import GL

import json
from importlib.resources import files

from pydice3d.camera       import FixedCamera, look_at, perspective
from pydice3d.physics      import PhysicsWorld
from pydice3d.render_data  import RenderScene
from pydice3d.renderer     import Renderer
from pydice3d.dice_state   import DiceState
from pydice3d.dice         import Dice
from pydice3d.spawner      import spawn_dice, SpawnConfig
from pydice3d.roll_result  import RollMonitor, RollResult
from pydice3d.dice_mesh    import get_mesh
from pydice3d.physics      import DICE_TARGET_SIZE


# ────────────────────────────────────────────────────────────────────────────
# Constantes
# ────────────────────────────────────────────────────────────────────────────

ATLAS_DIR = files("pydice3d.assets").joinpath("atlas")

ATLAS_PNG = str(ATLAS_DIR.joinpath("atlas.png"))
ATLAS_NORMAL_PNG = str(ATLAS_DIR.joinpath("atlas_normal.png"))
ATLAS_JSON = ATLAS_DIR.joinpath("atlas.json")

PHYSICS_STEPS_PER_FRAME = 4    # quantos steps de física por frame de render
DEBUG_NONE      = 0
DEBUG_COLLISION = 1
DEBUG_OVERLAY   = 2

# Cor do wireframe de colisão (R, G, B)
COLLISION_WIRE_COLOR = (0.0, 1.0, 0.3)   # verde neon


# ────────────────────────────────────────────────────────────────────────────
# Shaders de wireframe (debug)
# ────────────────────────────────────────────────────────────────────────────

def _compile_wire_program() -> int:
    from pydice3d.shaders import build_program, WIRE_FRAG, WIRE_VERT
    return build_program(WIRE_VERT, WIRE_FRAG)


# ────────────────────────────────────────────────────────────────────────────
# CollisionWireframe — VAO do hull de colisão de um dado
# ────────────────────────────────────────────────────────────────────────────

class CollisionWireframe:
    """
    Renderiza o convex hull de colisão de um dado como wireframe.

    Obtém os vértices diretamente de get_mesh() (dice_mesh.py),
    computa o convex hull via scipy e cria arestas únicas.

    Para o d6 (GEOM_BOX) os vértices são os 8 cantos do cubo.
    """

    def __init__(self, dice_type: str) -> None:
        self.dice_type = dice_type
        self._vao = 0
        self._vbo_v = 0
        self._vbo_e = 0
        self._n_lines = 0
        self._built = False

    def _build(self) -> None:
        """Constrói VAO/VBO (deve ser chamado com contexto OpenGL ativo)."""
        if self._built:
            return

        r = DICE_TARGET_SIZE   # mesmo valor usado em _make_collision_shape
        verts = self._get_hull_verts(r)
        edges = self._hull_edges(verts)

        if not edges:
            return

        # Constrói buffer de linhas: cada aresta = 2 vértices
        line_buf = np.array(
            [[verts[i], verts[j]] for i, j in edges],
            dtype=np.float32,
        ).reshape(-1, 3)

        self._vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao)
        self._vbo_v = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo_v)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, line_buf.nbytes,
                        line_buf.tobytes(), GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 12,
                                  GL.ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        GL.glBindVertexArray(0)

        self._n_lines = len(edges) * 2
        self._built = True

    def _get_hull_verts(self, r: float) -> np.ndarray:
        if self.dice_type == "d6":
            # GEOM_BOX com halfExtents = r/2
            half = r / 2.0
            pts = []
            for sx in (+1, -1):
                for sy in (+1, -1):
                    for sz in (+1, -1):
                        pts.append([sx * half, sy * half, sz * half])
            return np.array(pts, dtype=np.float32)

        mesh = get_mesh(self.dice_type)
        # Mesma escala que _make_collision_shape: vertices * r
        return (mesh.vertices * r).astype(np.float32)

    @staticmethod
    def _hull_edges(verts: np.ndarray) -> list[tuple[int, int]]:
        """Extrai arestas únicas do convex hull."""
        if len(verts) < 4:
            return []
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(verts)
            edge_set: set[tuple[int, int]] = set()
            for simplex in hull.simplices:
                for i in range(len(simplex)):
                    a, b = simplex[i], simplex[(i + 1) % len(simplex)]
                    edge_set.add((min(a, b), max(a, b)))
            return list(edge_set)
        except Exception:
            return []

    def draw(self, mvp: np.ndarray, program: int) -> None:
        if not self._built:
            self._build()
        if not self._built or self._n_lines == 0:
            return

        from pydice3d.shaders import set_uniform_mat4, set_uniform_vec3
        GL.glUseProgram(program)
        set_uniform_mat4(program, "u_mvp", mvp)
        set_uniform_vec3(program, "u_color", COLLISION_WIRE_COLOR)
        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_LINES, 0, self._n_lines)
        GL.glBindVertexArray(0)

    def delete(self) -> None:
        if self._built:
            GL.glDeleteVertexArrays(1, [self._vao])
            GL.glDeleteBuffers(1, [self._vbo_v])
            self._built = False


# ────────────────────────────────────────────────────────────────────────────
# DiceGLArea
# ────────────────────────────────────────────────────────────────────────────

class DiceGLArea(Gtk.GLArea):
    """
    Widget GTK4 GLArea que gerencia toda a renderização e física dos dados.

    API pública
    ───────────
    start_simulation(n, dice_type)  : inicia nova rolagem
    simulating                      : True enquanto dados estão se movendo
    debug_mode                      : DEBUG_NONE | DEBUG_COLLISION | DEBUG_OVERLAY
    """

    def __init__(self, physics: PhysicsWorld) -> None:
        super().__init__()
        self.physics = physics

        self.set_required_version(3, 3)
        self.set_has_depth_buffer(True)

        self._renderer:   Renderer | None = None
        self._scene:      RenderScene | None = None
        self._states:     list[DiceState] = []
        self._monitor:    RollMonitor | None = None
        self._wire_prog:  int = 0
        self._wire_objs:  list[CollisionWireframe] = []

        # Câmera top-down — diretamente acima da bandeja, olhando para o centro.
        # up=[0,0,-1] porque o eixo Y está ocupado pela direção de visão.
        self._cam_eye    = np.array([0.0, 12.0, 0.0], dtype=np.float32)
        self._cam_center = np.array([0.0,  0.0, 0.0], dtype=np.float32)
        self._cam_up     = np.array([0.0,  0.0,-1.0], dtype=np.float32)
        self._cam_fov    = 35.0
        self._cam_near   = 0.1
        self._cam_far    = 50.0

        # Dimensões do framebuffer em pixels físicos — atualizadas pelo sinal resize.
        # NÃO usar get_allocated_width/height em _on_render: em HiDPI eles retornam
        # pixels lógicos e o viewport cobriria apenas 1/4 da área.
        self._vp_w: int = 660
        self._vp_h: int = 460

        self.simulating   = False
        self.timer_id:    int = 0
        self._debug_mode: int = DEBUG_NONE

        # Conecta sinais GTK
        self.connect("realize",   self._on_realize)
        self.connect("unrealize", self._on_unrealize)
        self.connect("render",    self._on_render)
        self.connect("resize",    self._on_resize)

        # Captura de teclado para alternar debug
        key_ctrl = Gtk.EventControllerKey()
        key_ctrl.connect("key-pressed", self._on_key)
        self.add_controller(key_ctrl)
        self.set_focusable(True)

    # ── Propriedade debug_mode ────────────────────────────────────────

    @property
    def debug_mode(self) -> int:
        return self._debug_mode

    @debug_mode.setter
    def debug_mode(self, mode: int) -> None:
        self._debug_mode = mode
        if self._renderer:
            self._renderer.debug_mode = mode
        self.queue_render()

    # ── Sinais GTK/GL ────────────────────────────────────────────────

    def _on_resize(self, _area, width: int, height: int) -> None:
        self._vp_w = max(width, 1)
        self._vp_h = max(height, 1)
        # Recalcula limites da bandeja física com base no frustum da câmera
        # Câmera está a Y=12, FOV=45°. Half-height no chão:
        import math
        half_h = math.tan(math.radians(self._cam_fov / 2)) * self._cam_eye[1]
        aspect = self._vp_w / self._vp_h
        half_w = half_h * aspect
        self.physics.resize_tray(half_w * 0.95, half_h * 0.95)

    def _view_matrix(self) -> np.ndarray:
        return look_at(self._cam_eye, self._cam_center, self._cam_up)

    def _projection_matrix(self) -> np.ndarray:
        import math
        return perspective(
            math.radians(self._cam_fov),
            self._vp_w / self._vp_h,
            self._cam_near,
            self._cam_far,
        )

    def _view_projection(self) -> np.ndarray:
        return (self._projection_matrix() @ self._view_matrix()).astype(np.float32)

    def _cam_position(self) -> np.ndarray:
        return self._cam_eye

    def _on_realize(self, _area) -> None:
        self.make_current()
        if self.get_error():
            return
        self._wire_prog = _compile_wire_program()

        # Carrega atlas de glifos
        atlas_json = None
        try:
            with open(ATLAS_JSON, "r", encoding="utf-8") as f:
                atlas_json = json.load(f)
        except Exception as e:
            print(f"[AVISO] Não foi possível carregar atlas.json: {e}")
        self._atlas_json = atlas_json

        empty_scene = RenderScene([])
        self._renderer = Renderer(
            empty_scene, [],
            atlas_png=ATLAS_PNG,
            atlas_normal_png=ATLAS_NORMAL_PNG,
            atlas_json=atlas_json,
        )

    def _on_unrealize(self, _area) -> None:
        self.make_current()
        if self._renderer:
            self._renderer.delete()
            self._renderer = None
        for w in self._wire_objs:
            w.delete()
        if self._wire_prog:
            GL.glDeleteProgram(self._wire_prog)
            self._wire_prog = 0

    def _on_render(self, _area, _ctx) -> bool:
        w, h = self._vp_w, self._vp_h

        # Se está simulando, avança física
        if self.simulating and self._states:
            for _ in range(PHYSICS_STEPS_PER_FRAME):
                self.physics.step()
                for s in self._states:
                    s.update_status()
            if self._monitor:
                self._monitor.tick()
            if self.physics.all_sleeping():
                self.simulating = False

        # Atualiza dados de render (alpha=1 — sem interpolação extra por ora)
        if self._scene and self._states:
            self._scene.update(self._states, alpha=1.0)

        # Renderiza
        if self._renderer and self._scene:
            if self._debug_mode == DEBUG_COLLISION:
                GL.glViewport(0, 0, w, h)
                GL.glClearColor(0.0, 0.0, 0.0, 0.0)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
                self._draw_collision_wire()
            elif self._debug_mode == DEBUG_OVERLAY:
                self._renderer.draw(self._scene, self._view_projection(),
                                    self._cam_position(), w, h)
                GL.glEnable(GL.GL_POLYGON_OFFSET_LINE)
                GL.glPolygonOffset(-1.0, -1.0)
                self._draw_collision_wire()
                GL.glDisable(GL.GL_POLYGON_OFFSET_LINE)
            else:
                self._renderer.draw(self._scene, self._view_projection(),
                                    self._cam_position(), w, h)
        else:
            # Sem renderer ainda — limpa com transparente
            GL.glViewport(0, 0, w, h)
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        return True  # indica que o render foi tratado

    # ── Wireframe de colisão ─────────────────────────────────────────

    def _draw_collision_wire(self) -> None:
        """Desenha os hulls de colisão de todos os dados."""
        if not self._wire_prog or not self._states:
            return

        VP = self._view_projection()

        for wire, rd in zip(self._wire_objs, self._scene.dice_renders):
            M   = rd.model_mat
            MVP = (VP @ M).astype(np.float32)
            wire.draw(MVP, self._wire_prog)

    # ── Iniciar simulação ────────────────────────────────────────────

    def start_simulation(self, pool: dict[str, int]) -> None:
        """
        Remove dados anteriores e inicia uma nova rolagem.

        Parâmetros
        ----------
        pool : dicionário {dice_type: quantidade}, ex: {"d6": 2, "d20": 1}
        """
        self.make_current()
        if self.get_error():
            return

        # Limpa estado anterior
        self.simulating = False
        self.physics.remove_all_dice()
        self._states.clear()

        # Libera wireframes antigos
        for w in self._wire_objs:
            w.delete()
        self._wire_objs.clear()

        # Spawn com o pool completo
        result = spawn_dice(
            spec=pool,
            physics=self.physics,
            cfg=SpawnConfig(),
        )
        self._states = result.states

        # Cria wireframes de colisão
        for state in self._states:
            self._wire_objs.append(CollisionWireframe(state.dice.dice_type))

        # Cria/recarrega cena e renderer
        self._scene = RenderScene.from_states(self._states)
        dice_types  = [s.dice.dice_type for s in self._states]

        if self._renderer is None:
            self._renderer = Renderer(
                self._scene, dice_types,
                atlas_png=ATLAS_PNG,
                atlas_normal_png=ATLAS_NORMAL_PNG,
                atlas_json=getattr(self, "_atlas_json", None),
            )
        else:
            self._renderer.reload(self._scene, dice_types)

        # Sincroniza debug_mode para controle do highlight
        if self._renderer:
            self._renderer.debug_mode = self._debug_mode

        # Monitor de resultado
        self._monitor = RollMonitor(
            self._states,
            on_complete=self._on_roll_complete,
        )

        self.simulating = True
        self.grab_focus()

    # ── Resultado ────────────────────────────────────────────────────

    def _on_roll_complete(self, result: RollResult) -> None:
        """Chamado pelo RollMonitor quando todos os dados param."""
        print(f"[RESULTADO] {result.summary()}")
        # A AppWindow detecta simulating=False via _check_done timeout

    # ── Teclado (debug) ──────────────────────────────────────────────

    def _on_key(self, _ctrl, keyval, _keycode, _state) -> bool:
        from gi.repository import Gdk
        key = Gdk.keyval_name(keyval)
        if key == "n" or key == "N":
            self.debug_mode = DEBUG_NONE
            print("[DEBUG] Modo normal")
            return True
        if key == "c" or key == "C":
            self.debug_mode = DEBUG_COLLISION
            print("[DEBUG] Apenas colisão")
            return True
        if key == "o" or key == "O":
            self.debug_mode = DEBUG_OVERLAY
            print("[DEBUG] Overlay colisão+visual")
            return True
        return False
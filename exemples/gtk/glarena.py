"""
glarena.py – GTK4 GLArea: integração entre GTK e DiceSimulation

Responsabilidade: conectar os sinais do ciclo de vida GTK/GL
(realize / unrealize / render / resize) ao DiceSimulation e ao Renderer.

Modos de debug
──────────────
  DEBUG_NONE      : renderização normal
  DEBUG_COLLISION : só wireframe do hull de colisão (sem mesh visual)
  DEBUG_OVERLAY   : mesh visual + wireframe de colisão sobrepostos
"""

from __future__ import annotations

import json
import numpy as np
from importlib.resources import files

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from OpenGL import GL

from pydice3d.simulation     import DiceSimulation
from pydice3d.renderer       import Renderer
from pydice3d.render_data    import RenderScene
from pydice3d.roll_result    import RollResult
 
from debug_wire import (
    CollisionWireframe, build_wire_program,
    DEBUG_NONE, DEBUG_COLLISION, DEBUG_OVERLAY,
)



# ────────────────────────────────────────────────────────────────────────────
# Caminhos dos assets (pertencem à camada GTK: só aqui sabe onde estão)
# ────────────────────────────────────────────────────────────────────────────

_ATLAS_DIR  = files("pydice3d.assets").joinpath("atlas")
_ATLAS_NPY  = str(_ATLAS_DIR.joinpath("atlas.npy"))
_ATLAS_JSON = _ATLAS_DIR.joinpath("atlas.json")


# ────────────────────────────────────────────────────────────────────────────
# DiceGLArea
# ────────────────────────────────────────────────────────────────────────────

class DiceGLArea(Gtk.GLArea):
    """
    Widget GTK4 que exibe a simulação de dados em um contexto OpenGL 3.3.

    API pública
    ───────────
    start_simulation(spec)  inicia nova rolagem; spec = {"d6": 2, "d20": 1}
    simulation              acesso direto ao DiceSimulation (leitura)
    simulating              True enquanto os dados estão em movimento
    debug_mode              DEBUG_NONE | DEBUG_COLLISION | DEBUG_OVERLAY
    on_roll_complete        callback(RollResult) chamado quando todos param
    """

    def __init__(self) -> None:
        super().__init__()

        self.set_required_version(3, 3)
        self.set_has_depth_buffer(True)
        self.set_focusable(True)

        # Dimensões do framebuffer em pixels físicos.
        # NÃO usar get_allocated_width/height em _on_render: em HiDPI eles
        # retornam pixels lógicos e o viewport cobriria só 1/4 da área.
        self._vp_w: int = 660
        self._vp_h: int = 460

        # Núcleo da simulação — sem GTK, sem OpenGL
        self._sim = DiceSimulation(on_result=self._on_roll_complete)

        # Recursos OpenGL — criados em realize, destruídos em unrealize
        self._renderer:   Renderer | None = None
        self._scene:      RenderScene | None = None
        self._wire_prog:  int = 0
        self._wire_objs:  list[CollisionWireframe] = []
        self._atlas_json: dict | None = None

        self._debug_mode: int = DEBUG_NONE

        # Callback externo opcional: AppWindow pode sobrescrever
        self.on_roll_complete: object = None   # callable(RollResult) | None

        # Conecta sinais GTK
        self.connect("realize",   self._on_realize)
        self.connect("unrealize", self._on_unrealize)
        self.connect("render",    self._on_render)
        self.connect("resize",    self._on_resize)

    # ── Propriedades públicas ─────────────────────────────────────────────────

    @property
    def simulation(self) -> DiceSimulation:
        return self._sim

    @property
    def simulating(self) -> bool:
        return self._sim.is_rolling

    @property
    def debug_mode(self) -> int:
        return self._debug_mode

    @debug_mode.setter
    def debug_mode(self, mode: int) -> None:
        self._debug_mode = mode
        if self._renderer:
            self._renderer.debug_mode = mode
        self.queue_render()

    # ── Sinais GTK ────────────────────────────────────────────────────────────

    def _on_resize(self, _area, width: int, height: int) -> None:
        self._vp_w = max(width, 1)
        self._vp_h = max(height, 1)
        self._sim.resize(self._vp_w, self._vp_h)

    def _on_realize(self, _area) -> None:
        self.make_current()
        if self.get_error():
            return

        self._wire_prog = build_wire_program()

        # Carrega atlas de glifos (I/O feito aqui onde o contexto GL existe)
        try:
            with open(_ATLAS_JSON, "r", encoding="utf-8") as f:
                self._atlas_json = json.load(f)
        except Exception as e:
            print(f"[AVISO] Não foi possível carregar atlas.json: {e}")

        self._scene    = RenderScene([])
        self._renderer = Renderer(
            self._scene, [],
            atlas_npy=_ATLAS_NPY,
            atlas_json=self._atlas_json,
        )

    def _on_unrealize(self, _area) -> None:
        self.make_current()
        if self._renderer:
            self._renderer.delete()
            self._renderer = None
        for w in self._wire_objs:
            w.delete()
        self._wire_objs.clear()
        if self._wire_prog:
            GL.glDeleteProgram(self._wire_prog)
            self._wire_prog = 0

    def _on_render(self, _area, _ctx) -> bool:
        w, h = self._vp_w, self._vp_h

        # Avança física e monitora término
        self._sim.step()

        # Atualiza cena de render com as poses atuais dos dados
        if self._scene and self._sim.states:
            self._scene.update(self._sim.states, alpha=1.0)

        # Renderiza
        if self._renderer and self._scene:
            VP      = self._sim.view_projection()
            cam_pos = self._sim.camera_position()

            if self._debug_mode == DEBUG_COLLISION:
                GL.glViewport(0, 0, w, h)
                GL.glClearColor(0.0, 0.0, 0.0, 0.0)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
                self._draw_wire(VP)

            elif self._debug_mode == DEBUG_OVERLAY:
                self._renderer.draw(self._scene, VP, cam_pos, w, h)
                GL.glEnable(GL.GL_POLYGON_OFFSET_LINE)
                GL.glPolygonOffset(-1.0, -1.0)
                self._draw_wire(VP)
                GL.glDisable(GL.GL_POLYGON_OFFSET_LINE)

            else:
                self._renderer.draw(self._scene, VP, cam_pos, w, h)
        else:
            GL.glViewport(0, 0, w, h)
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        return True

    # ── API pública ───────────────────────────────────────────────────────────

    def start_simulation(self, spec: dict[str, int]) -> None:
        """
        Inicia uma nova rolagem, descartando a anterior.

        Parâmetros
        ----------
        spec : dicionário {tipo: quantidade}, ex: {"d6": 2, "d100": 1}.
               Tipos suportados: d4, d6, d8, d10, d12, d20, d100, df.
               d100 adiciona automaticamente 1 d10 parceiro.
        """
        self.make_current()
        if self.get_error():
            return

        # Libera wireframes do roll anterior
        for w in self._wire_objs:
            w.delete()
        self._wire_objs.clear()

        # Inicia simulação — spawn, estados, monitor
        self._sim.roll(spec)

        # Cria um CollisionWireframe por dado (lazy: VAO construído no 1º draw)
        for state in self._sim.states:
            self._wire_objs.append(CollisionWireframe(state.dice.dice_type))

        # Cria/recarrega cena e objetos GPU
        self._scene     = RenderScene.from_states(self._sim.states)
        dice_types      = [s.dice.dice_type for s in self._sim.states]

        if self._renderer is None:
            self._renderer = Renderer(
                self._scene, dice_types,
                atlas_npy=_ATLAS_NPY,
                atlas_json=self._atlas_json,
            )
        else:
            self._renderer.reload(self._scene, dice_types)

        if self._renderer:
            self._renderer.debug_mode = self._debug_mode

        self.grab_focus()

    # ── Internos ──────────────────────────────────────────────────────────────

    def _draw_wire(self, VP: np.ndarray) -> None:
        """Desenha os hulls de colisão de todos os dados ativos."""
        if not self._wire_prog or not self._scene:
            return
        for wire, rd in zip(self._wire_objs, self._scene.dice_renders):
            MVP = (VP @ rd.model_mat).astype(np.float32)
            wire.draw(MVP, self._wire_prog)

    def _on_roll_complete(self, result: RollResult) -> None:
        """Recebe o resultado do DiceSimulation e repassa ao callback externo."""
        print(f"[RESULT] {result.summary()}")
        if callable(self.on_roll_complete):
            self.on_roll_complete(result)
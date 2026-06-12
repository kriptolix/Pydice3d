"""
glarena.py - GTK4 GLArea.
"""

from __future__ import annotations

import numpy as np

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from OpenGL import GL

from pydice3d.simulation import DiceSimulation, RollResult
from pydice3d.renderer   import Renderer

from debug_wire import (
    CollisionWireframe, build_wire_program,
    DEBUG_NONE, DEBUG_COLLISION, DEBUG_OVERLAY,
)

class DiceGLArea(Gtk.GLArea):
    
    def __init__(self) -> None:
        super().__init__()

        self.set_required_version(3, 3)
        self.set_has_depth_buffer(True)
        self.set_focusable(True)

        # Framebuffer dimensions in physical pixels.
        # DO NOT use get_allocated_width/height in _on_render: in HiDPI they
        # return logical pixels and the viewport would only cover 1/4 of the area.
        self._vp_w: int = 660
        self._vp_h: int = 460

        self._sim = DiceSimulation(on_result=self._on_roll_complete)
        
        self._renderer:  Renderer | None = None
        self._wire_prog: int = 0
        self._wire_objs: list[CollisionWireframe] = []
        self._atlas_json: dict | None = None

        self._debug_mode: int = DEBUG_NONE
        
        self.on_roll_complete: object = None   # callable(RollResult) | None

        self.theme = "dark"
        
        self.connect("realize",   self._on_realize)
        self.connect("unrealize", self._on_unrealize)
        self.connect("render",    self._on_render)
        self.connect("resize",    self._on_resize)

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

    @property
    def theme(self) -> str:
        return self._sim.theme

    @theme.setter
    def theme(self, value: str) -> None:
        self._sim.theme = value          
        if self._renderer:
            self._renderer.theme = value
        self.queue_render()    

    def _on_resize(self, _area, width: int, height: int) -> None:
        self._vp_w = max(width, 1)
        self._vp_h = max(height, 1)
        self._sim.resize(self._vp_w, self._vp_h)

    def _on_realize(self, _area) -> None:
        self.make_current()
        if self.get_error():
            return

        self._wire_prog = build_wire_program()        

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
        
        self._sim.step()

        scene = self._sim.scene
        if self._renderer and scene:
            VP      = self._sim.view_projection()
            cam_pos = self._sim.camera_position()

            if self._debug_mode == DEBUG_COLLISION:
                GL.glViewport(0, 0, w, h)
                GL.glClearColor(0.0, 0.0, 0.0, 0.0)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
                self._draw_wire(VP, scene)

            elif self._debug_mode == DEBUG_OVERLAY:
                self._renderer.draw(scene, VP, cam_pos, w, h)
                GL.glEnable(GL.GL_POLYGON_OFFSET_LINE)
                GL.glPolygonOffset(-1.0, -1.0)
                self._draw_wire(VP, scene)
                GL.glDisable(GL.GL_POLYGON_OFFSET_LINE)

            else:
                self._renderer.draw(scene, VP, cam_pos, w, h)
        else:
            GL.glViewport(0, 0, w, h)
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        return True    

    def start_simulation(self, spec: dict[str, int]) -> None:
        """
        Start a new roll, discarding the previous one.     
        """
        self.make_current()
        if self.get_error():
            return

        for w in self._wire_objs:
            w.delete()
        self._wire_objs.clear()
        
        self._sim.roll(spec, theme=self._sim.theme)
       
        for dtype in self._sim.dice_types:
            self._wire_objs.append(CollisionWireframe(dtype))
        
        if self._renderer is None:
            self._renderer = Renderer(
                self._sim.scene, 
                self._sim.dice_types,                
                theme=self._sim.theme,
            )
        else:
            self._renderer.reload(self._sim.scene, self._sim.dice_types)

        if self._renderer:
            self._renderer.debug_mode = self._debug_mode

        self.grab_focus()  

    def _draw_wire(self, VP: np.ndarray, scene) -> None:
        
        if not self._wire_prog:
            return
        for wire, rd in zip(self._wire_objs, scene.dice_renders):
            MVP = (VP @ rd.model_mat).astype(np.float32)
            wire.draw(MVP, self._wire_prog)

    def _on_roll_complete(self, result: RollResult) -> None:
        
        print(f"[RESULT] {result.summary()}")
        if callable(self.on_roll_complete):
            self.on_roll_complete(result)
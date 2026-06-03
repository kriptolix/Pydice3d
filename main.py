"""
main.py — Ponto de entrada da aplicação GTK4.

Responsabilidades:
  - Janela principal e layout de controles
  - Seleção de tipo/quantidade de dado
  - Disparo e monitoramento da simulação
"""

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib

import sys

from physics import PhysicsWorld
from glarena import DiceGLArea, DEBUG_NONE, DEBUG_COLLISION, DEBUG_OVERLAY

# from dice_reader import start_calibration



DICE_TYPES = ["d4", "d6", "d8", "d10", "d12", "d20"]


class AppWindow(Gtk.ApplicationWindow):
    def __init__(self, app):
        super().__init__(
            application=app,
            title="Rolador de Dados 3D — PyBullet + GTK4 + OpenGL",
        )
        self.set_default_size(680, 600)        

        self.physics = PhysicsWorld()

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        root.set_margin_top(10);    root.set_margin_bottom(10)
        root.set_margin_start(10);  root.set_margin_end(10)
        self.set_child(root)

        # Título
        title = Gtk.Label(label="🎲  Rolador de Dados — Física Real")
        title.add_css_class("title-2")
        root.append(title)

        # Área GL
        self.gl = DiceGLArea(self.physics)
        self.gl.set_size_request(660, 460)
        self.gl.set_vexpand(True)
        root.append(self.gl)

        # ── Pool de dados ────────────────────────────────────────────
        # Botões para adicionar cada tipo ao pool
        add_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        add_row.set_halign(Gtk.Align.CENTER)
        root.append(add_row)

        add_row.append(Gtk.Label(label="Adicionar:"))
        for dtype in DICE_TYPES:
            b = Gtk.Button(label=dtype.upper())
            b.connect("clicked", self._on_add_die, dtype)
            add_row.append(b)

        # Linha de pool atual + botões de ação
        pool_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        pool_row.set_halign(Gtk.Align.CENTER)
        root.append(pool_row)

        self._pool_label = Gtk.Label(label="Pool: —")
        pool_row.append(self._pool_label)

        btn_roll = Gtk.Button(label="🎲  Rolar")
        btn_roll.add_css_class("suggested-action")
        btn_roll.connect("clicked", self._on_roll)
        pool_row.append(btn_roll)

        btn_clear = Gtk.Button(label="✕  Limpar")
        btn_clear.connect("clicked", self._on_clear_pool)
        pool_row.append(btn_clear)

        # Pool interno: {tipo: quantidade}
        self._pool: dict[str, int] = {}

        # ── Debug ────────────────────────────────────────────────────
        debug_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        debug_row.set_halign(Gtk.Align.CENTER)
        root.append(debug_row)

        debug_row.append(Gtk.Label(label="Debug:"))
        self._debug_btns: list[Gtk.ToggleButton] = []
        for label, mode in [("Normal [N]", DEBUG_NONE),
                             ("Só Colisão [C]", DEBUG_COLLISION),
                             ("Overlay [O]", DEBUG_OVERLAY)]:
            btn_d = Gtk.ToggleButton(label=label)
            if mode == DEBUG_NONE:
                btn_d.set_active(True)
            btn_d.connect("toggled", self._on_debug_toggle, mode)
            self._debug_btns.append(btn_d)
            debug_row.append(btn_d)

        # Status
        self.status = Gtk.Label(label="Adicione dados ao pool e clique em Rolar.")
        self.status.add_css_class("dim-label")
        root.append(self.status)

        # Render idle
        self.gl.timer_id = GLib.timeout_add(32, self._idle_render)

        # start_calibration("d4")   # troque pelo tipo que quer calibrar

    def _on_debug_toggle(self, btn: "Gtk.ToggleButton", mode: int) -> None:
        if not btn.get_active():
            return
        for b in self._debug_btns:
            if b is not btn:
                b.handler_block_by_func(self._on_debug_toggle)
                b.set_active(False)
                b.handler_unblock_by_func(self._on_debug_toggle)
        self.gl.debug_mode = mode

    def _on_add_die(self, _btn, dtype: str) -> None:
        self._pool[dtype] = self._pool.get(dtype, 0) + 1
        self._update_pool_label()

    def _on_clear_pool(self, _btn) -> None:
        self._pool.clear()
        self._update_pool_label()

    def _update_pool_label(self) -> None:
        if self._pool:
            parts = [f"{qty}×{dt.upper()}" for dt, qty in sorted(self._pool.items())]
            self._pool_label.set_label("Pool: " + "  ".join(parts))
        else:
            self._pool_label.set_label("Pool: —")

    def _idle_render(self) -> bool:
        self.gl.queue_render()
        return True

    def _on_roll(self, _btn):
        if not self._pool:
            self.status.set_label("Adicione dados ao pool primeiro.")
            return
        total = sum(self._pool.values())
        summary = ", ".join(f"{q}×{t.upper()}" for t, q in sorted(self._pool.items()))
        self.status.set_label(f"Rolando {summary}…")
        self.gl.start_simulation(self._pool.copy())
        GLib.timeout_add(300, self._check_done)

    def _check_done(self) -> bool:
        if self.gl.simulating:
            return True
        self.status.set_label("Dados parados. Role de novo!")
        return False


class App(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="com.example.DicePhysics")
        self.connect("activate", self._on_activate)

    def _on_activate(self, app):
        AppWindow(app).present()


if __name__ == "__main__":
    app = App()
    sys.exit(app.run(sys.argv))
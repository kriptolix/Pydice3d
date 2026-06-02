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

        # Controles
        ctrl = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        ctrl.set_halign(Gtk.Align.CENTER)
        root.append(ctrl)

        ctrl.append(Gtk.Label(label="Tipo:"))
        self.dice_combo = Gtk.DropDown.new_from_strings(DICE_TYPES)
        self.dice_combo.set_selected(1)   # d6 padrão
        ctrl.append(self.dice_combo)

        ctrl.append(Gtk.Label(label="Qtd:"))
        adj = Gtk.Adjustment(value=1, lower=1, upper=5,
                             step_increment=1, page_increment=1)
        self.spin = Gtk.SpinButton()
        self.spin.set_adjustment(adj)
        self.spin.set_numeric(True)
        ctrl.append(self.spin)

        btn = Gtk.Button(label="🎲  Rolar")
        btn.add_css_class("suggested-action")
        btn.connect("clicked", self._on_roll)
        ctrl.append(btn)

        # Controles de debug
        debug_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        debug_row.set_halign(Gtk.Align.CENTER)
        root.append(debug_row)

        debug_row.append(Gtk.Label(label="Debug:"))

        self._debug_btns: list[Gtk.ToggleButton] = []
        debug_labels = [("Normal [N]", DEBUG_NONE),
                        ("Só Colisão [C]", DEBUG_COLLISION),
                        ("Overlay [O]", DEBUG_OVERLAY)]

        for label, mode in debug_labels:
            btn_d = Gtk.ToggleButton(label=label)
            if mode == DEBUG_NONE:
                btn_d.set_active(True)
            btn_d.connect("toggled", self._on_debug_toggle, mode)
            self._debug_btns.append(btn_d)
            debug_row.append(btn_d)


        # Status
        self.status = Gtk.Label(label="Selecione o dado e clique em Rolar.")
        self.status.add_css_class("dim-label")
        root.append(self.status)

        # Render idle (só piso, sem dados)
        self.gl.timer_id = GLib.timeout_add(32, self._idle_render)

        # start_calibration("d4")   # troque pelo tipo que quer calibrar

    def _on_debug_toggle(self, btn: "Gtk.ToggleButton", mode: int) -> None:
        if not btn.get_active():
            return
        # Garante exclusividade (radio behavior manual)
        for b in self._debug_btns:
            if b is not btn:
                b.handler_block_by_func(self._on_debug_toggle)
                b.set_active(False)
                b.handler_unblock_by_func(self._on_debug_toggle)
        self.gl.debug_mode = mode

    def _idle_render(self) -> bool:
        self.gl.queue_render()
        return True

    def _on_roll(self, _btn):        
        
        n         = int(self.spin.get_value())
        idx       = self.dice_combo.get_selected()
        dice_type = DICE_TYPES[idx]
        self.status.set_label(f"Rolando {n}× {dice_type.upper()}…")
        self.gl.start_simulation(n, dice_type)
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
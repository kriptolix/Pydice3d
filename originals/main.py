import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, GLib

import sys

from originals.physics import PhysicsWorld
from originals.glarena import DiceGLArea


DICE_TYPES = ["d4", "d6", "d8", "d10", "d12", "d20"]


class AppWindow(Gtk.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app,
                         title="Rolador de Dados 3D — PyBullet + GTK4 + OpenGL")
        self.set_default_size(680, 600)

        self.physics = PhysicsWorld()

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        root.set_margin_top(10); root.set_margin_bottom(10)
        root.set_margin_start(10); root.set_margin_end(10)
        self.set_child(root)

        # Título
        title = Gtk.Label(label="🎲  Rolador de Dados — Física Real")
        title.add_css_class("title-2")
        root.append(title)

        # GL Area
        self.gl = DiceGLArea(self.physics)
        self.gl.set_size_request(660, 460)
        self.gl.set_vexpand(True)
        root.append(self.gl)

        # Controles
        ctrl = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        ctrl.set_halign(Gtk.Align.CENTER)
        root.append(ctrl)

        # Tipo de dado
        ctrl.append(Gtk.Label(label="Tipo:"))
        self.dice_combo = Gtk.DropDown.new_from_strings(DICE_TYPES)
        self.dice_combo.set_selected(1)   # d6 por padrão
        ctrl.append(self.dice_combo)

        # Quantidade
        ctrl.append(Gtk.Label(label="Qtd:"))
        adj = Gtk.Adjustment(value=1, lower=1, upper=5,
                             step_increment=1, page_increment=1)
        self.spin = Gtk.SpinButton()
        self.spin.set_adjustment(adj)
        self.spin.set_numeric(True)
        ctrl.append(self.spin)

        # Botão rolar
        btn = Gtk.Button(label="🎲  Rolar")
        btn.add_css_class("suggested-action")
        btn.connect("clicked", self._on_roll)
        ctrl.append(btn)

        # Status
        self.status = Gtk.Label(label="Selecione o dado e clique em Rolar.")
        self.status.add_css_class("dim-label")
        root.append(self.status)

        # Inicia render idle (sem dados ainda, só mostra o piso)
        self.gl.timer_id = GLib.timeout_add(32, self._idle_render)

    def _idle_render(self):
        self.gl.queue_render()
        return True

    def _on_roll(self, btn):
        n = int(self.spin.get_value())
        idx = self.dice_combo.get_selected()
        dice_type = DICE_TYPES[idx]
        self.status.set_label(f"Rolando {n}× {dice_type.upper()}…")
        self.gl.start_simulation(n, dice_type)
        GLib.timeout_add(300, self._check_done)

    def _check_done(self):
        if self.gl.simulating:
            return True
        self.status.set_label("Dados parados. Role de novo!")
        return False


class App(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="com.example.DicePhysics")
        self.connect("activate", self._on_activate)

    def _on_activate(self, app):
        win = AppWindow(app)
        win.present()


if __name__ == "__main__":
    app = App()
    sys.exit(app.run(sys.argv))
#!/usr/bin/env python3
"""
D10 com física real: PyBullet (simulação) + GTK4 GLArea + PyOpenGL (render)

Dependências:
    pip install pybullet PyOpenGL PyOpenGL-accelerate numpy
    (PyGObject via sistema: python3-gi)

Arquitetura:
    PhysicsWorld  → PyBullet headless, avança simulação a cada frame
    DiceGLArea    → Gtk.GLArea, lê transforms do PhysicsWorld e renderiza
    GLib.timeout  → cola os dois ao main loop do GTK (~60 fps)
"""

import sys
import math
import random
import numpy as np

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, GLib

from OpenGL.GL import *
import pybullet as pb
import pybullet_data


# ============================================================================
# GEOMETRIA
# ============================================================================

def build_d10():
    """
    Trapezoedro pentagonal (D10).
    Retorna (vertices_Nx3, indices_Mx3) para uso tanto no renderer
    quanto como ConvexHullShape no Bullet.
    """
    verts = []
    verts.append([0.0,  1.0,  0.0])   # 0 topo
    verts.append([0.0, -1.0,  0.0])   # 1 base

    r_top, y_top = 0.85, 0.22
    for i in range(5):
        a = 2 * math.pi * i / 5
        verts.append([r_top * math.cos(a), y_top, r_top * math.sin(a)])

    r_bot, y_bot = 0.85, -0.22
    for i in range(5):
        a = 2 * math.pi * i / 5 + math.pi / 5
        verts.append([r_bot * math.cos(a), y_bot, r_bot * math.sin(a)])

    indices = []
    for i in range(5):
        s0, s1 = 2 + i, 2 + (i + 1) % 5
        b0, b1 = 7 + i, 7 + (i + 1) % 5
        indices.extend([[0, s0, s1]])          # cap superior
        indices.extend([[1, b1, b0]])          # cap inferior
        indices.extend([[s0, b0, s1]])         # lateral A
        indices.extend([[s1, b0, b1]])         # lateral B

    return np.array(verts, dtype=np.float32), np.array(indices, dtype=np.int32)


def build_flat_box(w, h, d):
    """Caixa simples para a superfície visível."""
    hw, hh, hd = w / 2, h / 2, d / 2
    verts = np.array([
        [-hw, -hh, -hd], [ hw, -hh, -hd], [ hw,  hh, -hd], [-hw,  hh, -hd],
        [-hw, -hh,  hd], [ hw, -hh,  hd], [ hw,  hh,  hd], [-hw,  hh,  hd],
    ], dtype=np.float32)
    indices = np.array([
        [0,1,2],[0,2,3],  # -Z
        [4,6,5],[4,7,6],  # +Z
        [0,5,1],[0,4,5],  # -Y
        [2,6,3],[3,6,7],  # +Y  (topo — mais usada)
        [0,3,7],[0,7,4],  # -X
        [1,5,6],[1,6,2],  # +X
    ], dtype=np.int32)
    return verts, indices


def expand_flat_shading(vertices, indices):
    """Expande indexed mesh para flat shading (normal por face)."""
    pos_out, nor_out = [], []
    for tri in indices:
        v0 = np.array(vertices[tri[0]])
        v1 = np.array(vertices[tri[1]])
        v2 = np.array(vertices[tri[2]])
        n = np.cross(v1 - v0, v2 - v0)
        nl = np.linalg.norm(n)
        if nl > 1e-8:
            n /= nl
        for v in [v0, v1, v2]:
            pos_out.extend(v)
            nor_out.extend(n)
    return (np.array(pos_out, dtype=np.float32),
            np.array(nor_out, dtype=np.float32))


# ============================================================================
# SHADERS
# ============================================================================

VERT_SRC = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;

uniform mat4 uMVP;
uniform mat4 uModelView;
uniform mat3 uNormalMat;

out vec3 vNormal;
out vec3 vFragPos;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vFragPos    = vec3(uModelView * vec4(aPos, 1.0));
    vNormal     = normalize(uNormalMat * aNormal);
}
"""

FRAG_SRC = """
#version 330 core
in vec3 vNormal;
in vec3 vFragPos;

uniform vec3 uLightPos;   // em view-space
uniform vec3 uColor;
uniform float uAlpha;

out vec4 FragColor;

void main() {
    vec3 L = normalize(uLightPos - vFragPos);
    float ambient = 0.20;
    float diff    = max(dot(vNormal, L), 0.0);
    vec3  H       = normalize(L + normalize(-vFragPos));
    float spec    = pow(max(dot(vNormal, H), 0.0), 48.0) * 0.5;
    vec3  col     = uColor * (ambient + diff * 0.75) + vec3(spec);
    FragColor = vec4(col, uAlpha);
}
"""


def _compile(src, kind):
    sh = glCreateShader(kind)
    glShaderSource(sh, src)
    glCompileShader(sh)
    if not glGetShaderiv(sh, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(sh).decode())
    return sh


def make_program():
    vs = _compile(VERT_SRC, GL_VERTEX_SHADER)
    fs = _compile(FRAG_SRC, GL_FRAGMENT_SHADER)
    p = glCreateProgram()
    glAttachShader(p, vs); glAttachShader(p, fs)
    glLinkProgram(p)
    if not glGetProgramiv(p, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(p).decode())
    glDeleteShader(vs); glDeleteShader(fs)
    return p


def upload_mesh(pos_flat, nor_flat):
    """Cria VAO+VBOs e retorna (vao, vertex_count)."""
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo_p = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_p)
    glBufferData(GL_ARRAY_BUFFER, pos_flat.nbytes, pos_flat, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    vbo_n = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_n)
    glBufferData(GL_ARRAY_BUFFER, nor_flat.nbytes, nor_flat, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(1)

    glBindVertexArray(0)
    return vao, len(pos_flat) // 3


# ============================================================================
# MATRIZES
# ============================================================================

def mat4_persp(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0,0] = f / aspect
    m[1,1] = f
    m[2,2] = (far + near) / (near - far)
    m[2,3] = (2 * far * near) / (near - far)
    m[3,2] = -1.0
    return m


def mat4_lookat(eye, center, up):
    f = np.array(center) - np.array(eye)
    f /= np.linalg.norm(f)
    r = np.cross(f, np.array(up)); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4, dtype=np.float32)
    m[0,:3] = r;  m[0,3] = -np.dot(r, eye)
    m[1,:3] = u;  m[1,3] = -np.dot(u, eye)
    m[2,:3] = -f; m[2,3] =  np.dot(f, eye)
    return m


def mat4_from_bullet(pos, orn):
    """Converte posição+quaternion do Bullet para matriz 4x4 column-major→row-major."""
    # Bullet retorna quaternion (x,y,z,w)
    rm = pb.getMatrixFromQuaternion(orn)  # lista de 9 floats, row-major
    m = np.eye(4, dtype=np.float32)
    m[0,0]=rm[0]; m[0,1]=rm[1]; m[0,2]=rm[2]; m[0,3]=pos[0]
    m[1,0]=rm[3]; m[1,1]=rm[4]; m[1,2]=rm[5]; m[1,3]=pos[1]
    m[2,0]=rm[6]; m[2,1]=rm[7]; m[2,2]=rm[8]; m[2,3]=pos[2]
    return m


# ============================================================================
# MUNDO FÍSICO (PyBullet headless)
# ============================================================================

# Dimensões da bandeja (em metros do Bullet)
TRAY_W  = 5.0   # largura X
TRAY_D  = 5.0   # profundidade Z
TRAY_H  = 0.15  # espessura do piso
WALL_H  = 1.2   # altura das paredes
WALL_T  = 0.15  # espessura das paredes
DICE_R  = 0.5   # raio aproximado do dado (escala)

class PhysicsWorld:
    def __init__(self):
        self.client = pb.connect(pb.DIRECT)
        pb.setGravity(0, -9.8, 0, physicsClientId=self.client)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                   physicsClientId=self.client)

        self.dice_ids = []
        self.dice_verts = None  # verts originais para ConvexHull
        self._build_tray()

    def _plane_shape(self, half_extents):
        return pb.createCollisionShape(
            pb.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.client
        )

    def _static_box(self, half_extents, position, orientation=(0,0,0,1)):
        shape = self._plane_shape(half_extents)
        pb.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=shape,
            basePosition=position,
            baseOrientation=orientation,
            physicsClientId=self.client
        )

    def _build_tray(self):
        hw, hd, ht = TRAY_W/2, TRAY_D/2, TRAY_H/2
        wt = WALL_T / 2
        wh = WALL_H / 2

        # Piso
        self._static_box([hw, ht, hd], [0, -ht, 0])

        # Paredes (invisíveis no render, visíveis só na física)
        # Frente / Fundo  (eixo Z)
        self._static_box([hw, wh, wt], [ 0, wh, -(hd + wt)])
        self._static_box([hw, wh, wt], [ 0, wh,  (hd + wt)])
        # Esquerda / Direita (eixo X)
        self._static_box([wt, wh, hd], [-(hw + wt), wh, 0])
        self._static_box([wt, wh, hd], [ (hw + wt), wh, 0])    
        
    def add_dice(self, vertices):
        """Carrega o dado a partir de URDF."""

        # Posição aleatória acima da bandeja
        x = random.uniform(-TRAY_W * 0.3, TRAY_W * 0.3)
        z = random.uniform(-TRAY_D * 0.3, TRAY_D * 0.3)
        pos = [x, 2.5, z]

        # Orientação aleatória
        angle = random.uniform(0, 2 * math.pi)
        axis  = [random.gauss(0,1), random.gauss(0,1), random.gauss(0,1)]
        al    = math.sqrt(sum(a*a for a in axis))
        axis  = [a/al for a in axis]
        orn   = pb.getQuaternionFromAxisAngle(axis, angle,
                                              physicsClientId=self.client)

        # 👇 AQUI está a principal mudança
        body = pb.loadURDF(
            "cubo.urdf",
            basePosition=pos,
            baseOrientation=orn,
            physicsClientId=self.client
        )

        # Velocidade inicial: impulso lateral + spin
        vx = random.uniform(-4.0, 4.0)
        vz = random.uniform(-4.0, 4.0)

        pb.resetBaseVelocity(
            body,
            linearVelocity=[vx, random.uniform(1.0, 3.0), vz],
            angularVelocity=[random.uniform(-8,8),
                             random.uniform(-8,8),
                             random.uniform(-8,8)],
            physicsClientId=self.client
        )

        # Física (continua igual)
        pb.changeDynamics(
            body, -1,
            restitution=0.35,
            linearDamping=0.05,
            angularDamping=0.05,
            rollingFriction=0.02,
            spinningFriction=0.02,
            lateralFriction=0.5,
            physicsClientId=self.client
        )

        self.dice_ids.append(body)
        return body

    def remove_all_dice(self):
        for bid in self.dice_ids:
            pb.removeBody(bid, physicsClientId=self.client)
        self.dice_ids.clear()

    def step(self, dt=1/60.0):
        # Bullet usa passos fixos; chamamos quantas vezes necessário
        steps = max(1, int(dt / (1/240.0)))
        for _ in range(steps):
            pb.stepSimulation(physicsClientId=self.client)

    def get_transforms(self):
        """Retorna lista de (pos, orn) para cada dado."""
        result = []
        for bid in self.dice_ids:
            pos, orn = pb.getBasePositionAndOrientation(
                bid, physicsClientId=self.client)
            result.append((pos, orn))
        return result

    def all_sleeping(self):
        """Verifica se todos os dados pararam (velocidade baixa)."""
        for bid in self.dice_ids:
            lv, av = pb.getBaseVelocity(bid, physicsClientId=self.client)
            speed = math.sqrt(sum(v*v for v in lv) + sum(v*v for v in av))
            if speed > 0.02:
                return False
        return True

    def __del__(self):
        try:
            pb.disconnect(self.client)
        except Exception:
            pass


# ============================================================================
# GL AREA
# ============================================================================

class DiceGLArea(Gtk.GLArea):

    # Câmera isométrica ligeiramente inclinada
    CAM_EYE    = np.array([0.0, 9.0, 8.0], dtype=np.float32)
    CAM_CENTER = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    CAM_UP     = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def __init__(self, physics: PhysicsWorld):
        super().__init__()
        self.physics = physics
        self.set_required_version(3, 3)
        self.set_has_depth_buffer(True)

        self.prog       = None
        self.dice_vao   = None
        self.dice_vcount = 0
        self.floor_vao  = None
        self.floor_vcount = 0

        self.width  = 600
        self.height = 500
        self.timer_id = None
        self.simulating = False

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

        self.prog = make_program()

        # Geometria do dado
        raw_v, raw_i = build_d10()
        # Escalar para DICE_R
        raw_v_scaled = raw_v * DICE_R
        pos_flat, nor_flat = expand_flat_shading(raw_v_scaled, raw_i)
        self.dice_vao, self.dice_vcount = upload_mesh(pos_flat, nor_flat)

        # Registrar verts escalados no mundo físico
        self.physics.dice_verts = raw_v_scaled

        # Geometria da bandeja (superfície visível)
        floor_v, floor_i = build_flat_box(TRAY_W, TRAY_H, TRAY_D)
        fp, fn = expand_flat_shading(floor_v, floor_i)
        self.floor_vao, self.floor_vcount = upload_mesh(fp, fn)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def _on_unrealize(self, area):
        self.make_current()
        for vao in [self.dice_vao, self.floor_vao]:
            if vao:
                glDeleteVertexArrays(1, [vao])
        if self.prog:
            glDeleteProgram(self.prog)

    def _on_resize(self, area, w, h):
        self.width, self.height = w, h

    # ------------------------------------------------------------------
    def _uniforms(self, mvp, mv, color, alpha=1.0):
        def L(n): return glGetUniformLocation(self.prog, n)
        glUniformMatrix4fv(L("uMVP"),      1, GL_TRUE, mvp.flatten())
        glUniformMatrix4fv(L("uModelView"),1, GL_TRUE, mv.flatten())
        nm = mv[:3, :3].copy()
        # Inverse-transpose para normais corretas com escala não uniforme
        try:
            nm = np.linalg.inv(nm).T
        except np.linalg.LinAlgError:
            pass
        glUniformMatrix3fv(L("uNormalMat"), 1, GL_TRUE, nm.flatten())
        glUniform3f(L("uColor"), *color)
        glUniform1f(L("uAlpha"), alpha)
        # Luz fixa em view-space (posição relativa à câmera)
        glUniform3f(L("uLightPos"), 3.0, 6.0, 5.0)

    def _on_render(self, area, context):
        glClearColor(0.08, 0.08, 0.12, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if not self.prog:
            return True

        glUseProgram(self.prog)

        aspect = self.width / max(self.height, 1)
        proj   = mat4_persp(45.0, aspect, 0.1, 50.0)
        view   = mat4_lookat(self.CAM_EYE, self.CAM_CENTER, self.CAM_UP)

        # ---- Piso ----
        floor_model = np.eye(4, dtype=np.float32)
        # Centralizar verticalmente: piso vai de -TRAY_H a 0
        floor_model[1, 3] = -TRAY_H / 2
        mv  = view @ floor_model
        mvp = proj @ mv
        self._uniforms(mvp, mv, (0.22, 0.35, 0.22))   # verde escuro
        glBindVertexArray(self.floor_vao)
        glDrawArrays(GL_TRIANGLES, 0, self.floor_vcount)

        # ---- Dados ----
        for pos, orn in self.physics.get_transforms():
            model = mat4_from_bullet(pos, orn)
            mv    = view @ model
            mvp   = proj @ mv
            self._uniforms(mvp, mv, (0.85, 0.18, 0.12))   # vermelho
            glBindVertexArray(self.dice_vao)
            glDrawArrays(GL_TRIANGLES, 0, self.dice_vcount)

        glBindVertexArray(0)
        glUseProgram(0)
        return True

    # ------------------------------------------------------------------
    def start_simulation(self, n_dice=1):
        """Lança n_dice dados e inicia o loop de simulação."""
        self.physics.remove_all_dice()
        raw_v, _ = build_d10()
        raw_v_scaled = raw_v * DICE_R
        for _ in range(n_dice):
            self.physics.add_dice(raw_v_scaled)

        self.simulating = True
        if self.timer_id:
            GLib.source_remove(self.timer_id)
        self.timer_id = GLib.timeout_add(16, self._tick)   # ~60 fps

    def _tick(self):
        if self.simulating:
            self.physics.step(1 / 60.0)
            if self.physics.all_sleeping():
                self.simulating = False   # para mas mantém o frame final
        self.queue_render()
        return True   # sempre mantém o timer (para re-renderizar se necessário)

    def stop_timer(self):
        if self.timer_id:
            GLib.source_remove(self.timer_id)
            self.timer_id = None


# ============================================================================
# JANELA
# ============================================================================

class AppWindow(Gtk.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app,
                         title="Rolador de Dados 3D — PyBullet + GTK4 + OpenGL")
        self.set_default_size(680, 580)

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
        ctrl = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        ctrl.set_halign(Gtk.Align.CENTER)
        root.append(ctrl)

        # Seletor de quantidade de dados
        lbl = Gtk.Label(label="Dados:")
        ctrl.append(lbl)

        self.spin = Gtk.SpinButton()
        adj = Gtk.Adjustment(value=1, lower=1, upper=5,
                             step_increment=1, page_increment=1)
        self.spin.set_adjustment(adj)
        self.spin.set_numeric(True)
        ctrl.append(self.spin)

        # Botão rolar
        btn = Gtk.Button(label="  Rolar")
        btn.add_css_class("suggested-action")
        btn.connect("clicked", self._on_roll)
        ctrl.append(btn)

        # Status
        self.status = Gtk.Label(label="Clique em Rolar para lançar os dados.")
        self.status.add_css_class("dim-label")
        root.append(self.status)

        # Inicia render idle (sem dados ainda, só mostra o piso)
        self.gl.timer_id = GLib.timeout_add(32, self._idle_render)

    def _idle_render(self):
        self.gl.queue_render()
        return True

    def _on_roll(self, btn):
        n = int(self.spin.get_value())
        self.status.set_label(f"Rolando {n} dado(s)…")
        self.gl.start_simulation(n)
        GLib.timeout_add(200, self._check_done)

    def _check_done(self):
        if self.gl.simulating:
            return True   # ainda rolando, checa de novo
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

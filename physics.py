import pybullet as pb
import pybullet_data
import math
import random
import numpy as np

# Dimensões da bandeja (em metros do Bullet)
TRAY_W  = 7.0    # largura X
TRAY_D  = 7.0    # profundidade Z
TRAY_H  = 0.15   # espessura do piso
WALL_H  = 4.0    # altura das paredes
WALL_T  = 0.6    # espessura das paredes

# Tamanho alvo do dado no mundo físico (em metros).
DICE_TARGET_SIZE = 0.8

LAUNCH_Y       = 2.5
LAUNCH_VEL_MAX = 5.0

# Timestep fixo da simulação Bullet
SIM_TIMESTEP   = 1.0 / 240.0
# Sub-passos por chamada de _tick (1 tick ≈ 16ms → 4 × 1/240 ≈ 16.7ms)
SIM_SUBSTEPS   = 4


def _icosahedron_verts(r=1.0):
    """12 vértices de icosaedro regular (D20)."""
    phi = (1 + math.sqrt(5)) / 2
    pts = []
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            pts.append([0,       s1 * 1, s2 * phi])
            pts.append([s1 * 1,  s2 * phi, 0])
            pts.append([s1 * phi, 0,      s2 * 1])
    norm = math.sqrt(1 + phi * phi)
    return [[x / norm * r, y / norm * r, z / norm * r] for x, y, z in pts]


def _octahedron_verts(r=1.0):
    """6 vértices de octaedro regular (D8)."""
    return [
        [ r,  0,  0], [-r,  0,  0],
        [ 0,  r,  0], [ 0, -r,  0],
        [ 0,  0,  r], [ 0,  0, -r],
    ]


def _tetrahedron_verts(r=1.0):
    """4 vértices de tetraedro regular (D4)."""
    return [
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
    ]  # já normalizados para distância sqrt(3) do centro


def _dodecahedron_verts(r=1.0):
    """20 vértices de dodecaedro regular (D12)."""
    phi = (1 + math.sqrt(5)) / 2
    pts = []
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            for s3 in (+1, -1):
                pts.append([s1 * 1, s2 * 1, s3 * 1])
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            pts.append([0,        s1 * phi, s2 / phi])
            pts.append([s1 / phi, 0,        s2 * phi])
            pts.append([s1 * phi, s2 / phi, 0       ])
    norm = math.sqrt(3)
    return [[x / norm * r, y / norm * r, z / norm * r] for x, y, z in pts]


def _trapezoid_d10_verts(r=1.0):
    """10 vértices do trapezoedro pentagonal (D10)."""
    verts = []
    r_top, y_top = 0.85 * r, 0.22 * r
    for i in range(5):
        a = 2 * math.pi * i / 5
        verts.append([r_top * math.cos(a), y_top, r_top * math.sin(a)])
    r_bot, y_bot = 0.85 * r, -0.22 * r
    for i in range(5):
        a = 2 * math.pi * i / 5 + math.pi / 5
        verts.append([r_bot * math.cos(a), y_bot, r_bot * math.sin(a)])
    return verts


# Mapa: tipo → função que gera vértices do convex hull de colisão
_COLLISION_VERTS = {
    "d4":  _tetrahedron_verts,
    "d8":  _octahedron_verts,
    "d10": _trapezoid_d10_verts,
    "d12": _dodecahedron_verts,
    "d20": _icosahedron_verts,
}


class PhysicsWorld:
    def __init__(self):
        self.client = pb.connect(pb.DIRECT)
        pb.setGravity(0, -9.8, 0, physicsClientId=self.client)
        pb.setTimeStep(SIM_TIMESTEP, physicsClientId=self.client)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                   physicsClientId=self.client)

        self.dice_ids   = []
        self.dice_verts = None   # verts visuais escalados (setado pelo renderer)
        self._static_ids = []
        self._dice_scale = 1.0
        self._build_tray()

    # ------------------------------------------------------------------
    # Escala visual → física
    # ------------------------------------------------------------------
    def set_dice_scale(self, scale: float):
        """Recebe a escala calculada pelo renderer a partir do bounding box do OBJ."""
        self._dice_scale = scale

    # ------------------------------------------------------------------
    # Bandeja estática
    # ------------------------------------------------------------------
    def _static_box(self, half_extents, position, orientation=(0, 0, 0, 1)):
        shape = pb.createCollisionShape(
            pb.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.client
        )
        body = pb.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=shape,
            basePosition=position,
            baseOrientation=orientation,
            physicsClientId=self.client
        )
        self._static_ids.append(body)
        return body

    def _build_tray(self):
        hw = TRAY_W / 2
        hd = TRAY_D / 2
        ht = TRAY_H / 2
        wt = WALL_T / 2
        wh = WALL_H / 2

        # Piso
        floor_id = self._static_box([hw, ht, hd], [0, -ht, 0])
        pb.changeDynamics(
            floor_id, -1,
            restitution=0.45,
            lateralFriction=0.6,
            physicsClientId=self.client
        )

        # Paredes: extensão extra nos cantos para eliminar gaps
        ext = hw + wt
        for pos, he in [
            ([0,  wh, -(hd + wt)], [ext, wh, wt]),   # frente
            ([0,  wh,  (hd + wt)], [ext, wh, wt]),   # fundo
            ([-(hw + wt), wh, 0],  [wt, wh, hd]),    # esquerda
            ([ (hw + wt), wh, 0],  [wt, wh, hd]),    # direita
        ]:
            wall = self._static_box(he, pos)
            pb.changeDynamics(
                wall, -1,
                restitution=0.35,
                lateralFriction=0.4,
                physicsClientId=self.client
            )

    # ------------------------------------------------------------------
    # Criação de dados — shape de colisão via primitivas matemáticas
    # ------------------------------------------------------------------
    def _make_collision_shape(self, dice_type: str) -> int:
        """
        Retorna um collisionShapeIndex para o tipo de dado.
        D6 usa GEOM_BOX (mais eficiente e preciso).
        Demais usam GEOM_MESH convex hull com vértices matemáticos mínimos.
        """
        r = DICE_TARGET_SIZE / 2.0   # raio = metade do tamanho alvo

        if dice_type == "d6":
            return pb.createCollisionShape(
                pb.GEOM_BOX,
                halfExtents=[r, r, r],
                physicsClientId=self.client
            )

        gen = _COLLISION_VERTS.get(dice_type)
        if gen is None:
            raise ValueError(f"Tipo de dado desconhecido: {dice_type!r}. "
                             f"Use um de: d4, d6, d8, d10, d12, d20")

        verts = gen(r)
        return pb.createCollisionShape(
            pb.GEOM_MESH,
            vertices=verts,
            physicsClientId=self.client,
            flags=pb.GEOM_FORCE_CONCAVE_TRIMESH  # força convex hull
        )

    def add_dice(self, dice_type: str = "d6"):
        """
        Cria um dado do tipo especificado com shape de colisão preciso.

        O mesh visual (OBJ) é carregado separadamente em glarena.py;
        aqui criamos APENAS o corpo físico com primitivas matemáticas.

        dice_type: "d4" | "d6" | "d8" | "d10" | "d12" | "d20"
        """
        shape = self._make_collision_shape(dice_type)

        # Posição de lançamento aleatória dentro da bandeja
        x = random.uniform(-TRAY_W * 0.25, TRAY_W * 0.25)
        z = random.uniform(-TRAY_D * 0.25, TRAY_D * 0.25)
        pos = [x, LAUNCH_Y, z]

        # Orientação aleatória
        axis_raw = [random.gauss(0, 1) for _ in range(3)]
        al = math.sqrt(sum(a * a for a in axis_raw))
        axis = [a / al for a in axis_raw]
        angle = random.uniform(0, 2 * math.pi)
        orn = pb.getQuaternionFromAxisAngle(axis, angle,
                                            physicsClientId=self.client)

        # Massa proporcional ao volume (aprox. esfera de raio r)
        r = DICE_TARGET_SIZE / 2.0
        mass = 0.02 * (4 / 3) * math.pi * r ** 3 * 1000   # ~20g para d6 padrão

        body = pb.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=shape,
            basePosition=pos,
            baseOrientation=orn,
            physicsClientId=self.client
        )

        # CCD: raio menor que metade do menor lado do dado
        ccd_r = DICE_TARGET_SIZE * 0.25

        pb.changeDynamics(
            body, -1,
            restitution=0.4,
            linearDamping=0.03,
            angularDamping=0.04,
            rollingFriction=0.08,
            spinningFriction=0.05,
            lateralFriction=0.55,
            ccdSweptSphereRadius=ccd_r,
            physicsClientId=self.client
        )

        # Velocidade de lançamento
        vx = random.uniform(-LAUNCH_VEL_MAX, LAUNCH_VEL_MAX)
        vz = random.uniform(-LAUNCH_VEL_MAX, LAUNCH_VEL_MAX)
        pb.resetBaseVelocity(
            body,
            linearVelocity=[vx, random.uniform(0.5, 2.5), vz],
            angularVelocity=[random.uniform(-8, 8),
                             random.uniform(-8, 8),
                             random.uniform(-8, 8)],
            physicsClientId=self.client
        )

        self.dice_ids.append(body)
        return body

    # ------------------------------------------------------------------
    def remove_all_dice(self):
        for bid in self.dice_ids:
            pb.removeBody(bid, physicsClientId=self.client)
        self.dice_ids.clear()

    def step(self):
        """Avança a simulação por SIM_SUBSTEPS passos de SIM_TIMESTEP cada."""
        for _ in range(SIM_SUBSTEPS):
            pb.stepSimulation(physicsClientId=self.client)

    def get_transforms(self):
        result = []
        for bid in self.dice_ids:
            pos, orn = pb.getBasePositionAndOrientation(
                bid, physicsClientId=self.client)
            result.append((pos, orn))
        return result

    def all_sleeping(self):
        """
        Verifica se todos os dados pararam de se mover.

        getActivationState não está disponível em todas as versões do PyBullet,
        então usamos velocidade linear + angular com um limiar conservador.
        O contador _still_frames exige que o dado fique parado por ao menos
        30 ticks consecutivos antes de declarar sleeping — evita falso positivo
        logo após o lançamento (quando a velocidade inicial ainda é zero por
        um frame antes do primeiro impulso).
        """
        if not self.dice_ids:
            return True

        all_still = True
        for bid in self.dice_ids:
            lv, av = pb.getBaseVelocity(bid, physicsClientId=self.client)
            speed = sum(v * v for v in lv) + sum(v * v for v in av)
            if speed > 0.01 ** 2:   # limiar em velocidade² (0.01 m/s)
                all_still = False
                break

        if all_still:
            self._still_frames = getattr(self, "_still_frames", 0) + 1
        else:
            self._still_frames = 0

        return self._still_frames >= 30

    def __del__(self):
        try:
            pb.disconnect(self.client)
        except Exception:
            pass
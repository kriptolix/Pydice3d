"""
physics.py — Simulação física dos dados via PyBullet.

Responsabilidades:
  - Construção da bandeja estática (piso + paredes)
  - Criação e destruição de corpos rígidos dos dados
  - Shapes de colisão por tipo (D4–D20)
  - Stepping da simulação e detecção de repouso
"""

import pybullet as pb
import pybullet_data
import math
import random
import numpy as np

from dice_mesh import get_mesh

# Dimensões da bandeja (em metros do Bullet)
TRAY_W  = 13.0    # largura X
TRAY_D  = 9.0    # profundidade Z
TRAY_H  = 0.15   # espessura do piso
WALL_H  = 9.0    # altura das paredes
WALL_T  = 0.6    # espessura das paredes

# Tamanho alvo do dado no mundo físico (em metros).
DICE_TARGET_SIZE = 1.0

LAUNCH_Y       = 1.5
LAUNCH_VEL_MAX = 5.0

# Timestep fixo da simulação Bullet
SIM_TIMESTEP   = 1.0 / 240.0
# Sub-passos por chamada de step — mais passos reduzem interpenetração entre dados
SIM_SUBSTEPS   = 6


# ---------------------------------------------------------------------------
# PhysicsWorld
# ---------------------------------------------------------------------------

class PhysicsWorld:
    def __init__(self):
        self.client = pb.connect(pb.DIRECT)
        pb.setGravity(0, -9.8, 0, physicsClientId=self.client)
        pb.setTimeStep(SIM_TIMESTEP, physicsClientId=self.client)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                   physicsClientId=self.client)

        self.dice_ids  = []
        self._dice_types: dict[int, str] = {}   # body_id → dice_type
        self._static_ids = []
        self._dice_scale = 1.0
        self._still_frames = 0

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
        ext = hw + wt

        # Piso
        floor_id = self._static_box([hw, ht, hd], [0, -ht, 0])
        pb.changeDynamics(
            floor_id, -1,
            restitution=0.45,
            lateralFriction=1.2,
            physicsClientId=self.client
        )

        # Paredes
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
    # Criação de dados
    # ------------------------------------------------------------------

    def _make_collision_shape(self, dice_type: str) -> int:
        # r = tamanho alvo completo (sem dividir por 2).
        # Vértices da DiceMesh normalizados para raio unitário → escala por r.
        # Não encolhemos pela margem do Bullet: isso causaria afundar no chão.
        r = DICE_TARGET_SIZE * 1.1

        if dice_type == "d6":
            half = r / 2.0
            return pb.createCollisionShape(
                pb.GEOM_BOX,
                halfExtents=[half, half, half],
                physicsClientId=self.client
            )

        mesh = get_mesh(dice_type)
        verts = (mesh.vertices * r).tolist()
        return pb.createCollisionShape(
            pb.GEOM_MESH,
            vertices=verts,
            physicsClientId=self.client,
        )

    def add_dice(self, dice_type: str = "d6") -> int:
        """Cria um dado do tipo especificado e retorna seu body ID."""
        shape = self._make_collision_shape(dice_type)

        x   = random.uniform(-TRAY_W * 0.2, TRAY_W * 0.2)   # centralizado horizontalmente
        z   = TRAY_D * 0.45                                   # próximo à parede frontal
        pos = [x, LAUNCH_Y, z]

        axis_raw = [random.gauss(0, 1) for _ in range(3)]
        al       = math.sqrt(sum(a * a for a in axis_raw))
        axis     = [a / al for a in axis_raw]
        angle    = random.uniform(0, 2 * math.pi)
        orn      = pb.getQuaternionFromAxisAngle(axis, angle,
                                                 physicsClientId=self.client)

        # Massa fixa igual para todos os tipos — evita que dados menores
        # (D4, D8) acelerem mais que o D6 com o mesmo impulso de lançamento.
        mass = 0.020   # 20g, equivalente a um dado de resina padrão

        body = pb.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=shape,
            basePosition=pos,
            baseOrientation=orn,
            physicsClientId=self.client
        )

        pb.changeDynamics(
            body, -1,
            restitution=0.4,
            linearDamping=0.01,
            angularDamping=0.01,
            rollingFriction=0.01,
            spinningFriction=0.02,
            lateralFriction=1.5,
            # CCD: raio = metade do tamanho do dado — detecta colisões dentro de um passo
            ccdSweptSphereRadius=DICE_TARGET_SIZE * 0.25,
            # Margem de contato: empurra os dados para fora quando sobrepostos
            contactProcessingThreshold=0.0,
            physicsClientId=self.client
        )
               
        pb.resetBaseVelocity(            
            body,
            linearVelocity=[
                random.uniform(-0.5, 0.5),        # desvio lateral mínimo
                random.uniform(0.5, 1.0),         # leve vertical
                random.uniform(-8.0, -6.0),       # impulso principal para dentro
            ],
            angularVelocity=[
                random.uniform(-8, 8),
                random.uniform(-8, 8),
                random.uniform(-8, 8),
            ],
            physicsClientId=self.client
        )

        self.dice_ids.append(body)
        self._dice_types[body] = dice_type
        return body

    # ------------------------------------------------------------------
    # Controle de simulação
    # ------------------------------------------------------------------

    def create_dice_body(self, dice_type: str, position: tuple, scale: float = 1.0) -> int:
        """
        Alias semântico para add_dice, usado por dice.Dice.create().
        `position` e `scale` são registrados; a posição inicial é sobrescrita
        pelo lançamento em add_dice, mas pode ser usada futuramente.
        """
        self._dice_scale = scale
        return self.add_dice(dice_type)

    def remove_all_dice(self):
        for bid in self.dice_ids:
            pb.removeBody(bid, physicsClientId=self.client)
        self.dice_ids.clear()
        self._dice_types.clear()

    def step(self):
        """Avança a simulação por SIM_SUBSTEPS passos de SIM_TIMESTEP cada."""
        for _ in range(SIM_SUBSTEPS):
            pb.stepSimulation(physicsClientId=self.client)

    def get_transforms(self) -> list[tuple]:
        """Retorna lista de (pos, orn) para todos os dados."""
        return [
            pb.getBasePositionAndOrientation(bid, physicsClientId=self.client)
            for bid in self.dice_ids
        ]
    
    def get_transforms_for_type(self, dice_type: str) -> list[tuple]:
        return [
            pb.getBasePositionAndOrientation(bid, physicsClientId=self.client)
            for bid in self.dice_ids
            if self._dice_types.get(bid) == dice_type
        ]

    def all_sleeping(self) -> bool:
        """
        Retorna True quando todos os dados ficaram parados por ≥30 ticks.
        Usa velocidade linear+angular para evitar dependência de activation state.
        """
        if not self.dice_ids:
            return True

        all_still = all(
            sum(v * v for v in lv) + sum(v * v for v in av) <= 0.01 ** 2
            for bid in self.dice_ids
            for lv, av in [pb.getBaseVelocity(bid, physicsClientId=self.client)]
        )

        self._still_frames = (self._still_frames + 1) if all_still else 0
        return self._still_frames >= 30

    def __del__(self):
        try:
            pb.disconnect(self.client)
        except Exception:
            pass
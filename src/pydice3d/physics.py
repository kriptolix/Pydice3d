"""
physics.py — Physical simulation of data via PyBullet.

Responsibilities:
- Construction of the static tray (floor + walls)
- Creation and destruction of rigid bodies in the data
- Collision shapes by data type
- Simulation stepping and rest detection
"""

import pybullet as pb
import pybullet_data
import math
import random
import numpy as np

from pydice3d.dice_mesh import get_mesh
from pydice3d.audio     import CollisionEvent, Surface

# Tray dimensions (in meters of the Bullet)
TRAY_W  = 13.0   
TRAY_D  = 12.0    
TRAY_H  = 0.15   
WALL_H  = 9.0    
WALL_T  = 0.6    

# Target size of the data in the physical world (in meters)
DICE_TARGET_SIZE = 1.0

# Spawn height 
LAUNCH_Y = 1.8

# Fixed timestep for Bullet simulation
SIM_TIMESTEP = 1.0 / 240.0
# Substeps per step call
SIM_SUBSTEPS = 6

# Collision groups (bitmask)
# TRAY: collides with all
# WARM: newly created data — collides with the tray but NOT with other data
# COLD: stabilized data — collides with everything normally
COL_GROUP_TRAY = 0b001
COL_GROUP_WARM = 0b010
COL_GROUP_COLD = 0b100

COL_MASK_TRAY  = 0b111          
COL_MASK_WARM  = COL_GROUP_TRAY 
COL_MASK_COLD  = 0b111          

# How many frames (of step()) does each die spend in the WARM group before being promoted?
# 30 frames × 6 substeps × (1/240s) ≈ 0.75 s — enough time for the dice
# to spread out without explosively pushing each other at birth.
WARM_FRAMES = 30

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

        self.dice_ids:   list[int]      = []
        self._dice_types: dict[int, str] = {}
        self._static_ids: list[int]     = []
        self._dice_scale  = 1.0
        self._still_frames = 0

        # body_id → frames restantes no grupo WARM (sem colisão entre dados)
        self._warm_frames: dict[int, int] = {}

        # Rastreia posições já usadas nesta rodada para distribuição sem
        # interpenetração quando add_dice é chamado sequencialmente.
        self._pending_positions: list[tuple[float, float]] = []
        self._pending_dice_type: str = ""

        # IDs da geometria estática — usados para classificar o tipo de
        # superfície em poll_collision_events().
        self._floor_id: int = -1   # definido em _build_tray
        self._wall_ids: set[int] = set()

        # Pares de corpos que estavam em contato no tick anterior.
        # Usado para emitir eventos apenas na *nova* colisão, não no contato contínuo.
        self._prev_contacts: set[tuple[int, int]] = set()

        self._build_tray()

    # ------------------------------------------------------------------
    # Escala visual → física
    # ------------------------------------------------------------------

    def set_dice_scale(self, scale: float) -> None:
        self._dice_scale = scale

    # ------------------------------------------------------------------
    # Bandeja estática
    # ------------------------------------------------------------------

    def _static_box(self, half_extents, position, orientation=(0, 0, 0, 1)):
        shape = pb.createCollisionShape(
            pb.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.client,
        )
        body = pb.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=shape,
            basePosition=position,
            baseOrientation=orientation,
            physicsClientId=self.client,
        )
        pb.setCollisionFilterGroupMask(
            body, -1,
            COL_GROUP_TRAY, COL_MASK_TRAY,
            physicsClientId=self.client,
        )
        self._static_ids.append(body)
        return body

    def _build_tray(self) -> None:
        hw  = TRAY_W / 2
        hd  = TRAY_D / 2
        ht  = TRAY_H / 2
        wt  = WALL_T / 2
        wh  = WALL_H / 2
        ext = hw + wt

        # Piso
        floor_id = self._static_box([hw, ht, hd], [0, -ht, 0])
        pb.changeDynamics(
            floor_id, -1,
            restitution=0.45,          
            lateralFriction=1.2,       
            physicsClientId=self.client,
        )
        self._floor_id = floor_id

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
                restitution=0.65,     
                lateralFriction=0.5,
                physicsClientId=self.client,
            )
            self._wall_ids.add(wall)

    # ------------------------------------------------------------------
    # Dice creation
    # ------------------------------------------------------------------

    def _make_collision_shape(self, dice_type: str) -> int:
        r = DICE_TARGET_SIZE * 1.1

        if dice_type in ("d6", "df"):
            half = r / 2.0
            return pb.createCollisionShape(
                pb.GEOM_BOX,
                halfExtents=[half, half, half],
                physicsClientId=self.client,
            )
        
        if dice_type in ("d10", "d100"):
            r = DICE_TARGET_SIZE * 1.65

        mesh  = get_mesh(dice_type)
        verts = (mesh.vertices * r).tolist()
        return pb.createCollisionShape(
            pb.GEOM_MESH,
            vertices=verts,
            physicsClientId=self.client,
        )    

    def add_dice(self, dice_type: str = "d6") -> int:
        """
        Cria um dado do tipo especificado e retorna seu body ID.

        Usa posições pré-calculadas por prepare_roll() se disponíveis para
        o mesmo tipo. Caso contrário, gera posição aleatória sem garantia
        de separação (adequado para dados únicos).
        """
        shape = self._make_collision_shape(dice_type)

        x = random.uniform(-TRAY_W * 0.2, TRAY_W * 0.2)
        z = TRAY_D * 0.45
        pos = [x, LAUNCH_Y, z]

        pos = [x, LAUNCH_Y, z]

        # Orientação inicial aleatória
        axis_raw = [random.gauss(0, 1) for _ in range(3)]
        al       = math.sqrt(sum(a * a for a in axis_raw))
        axis     = [a / al for a in axis_raw]
        angle    = random.uniform(0, 2 * math.pi)
        orn      = pb.getQuaternionFromAxisAngle(axis, angle,
                                                  physicsClientId=self.client)

        mass = 0.020   # 20g — igual para todos os tipos

        body = pb.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=shape,
            basePosition=pos,
            baseOrientation=orn,
            physicsClientId=self.client,
        )

        pb.changeDynamics(
            body, -1,
            restitution=0.35,           
            linearDamping=0.02,         # resistência do ar leve — não "engole" a jogada
            angularDamping=0.02,        # rola livremente
            rollingFriction=0.03,
            spinningFriction=0.03,
            lateralFriction=0.8,
            ccdSweptSphereRadius=DICE_TARGET_SIZE * 0.30,
            contactProcessingThreshold=0.001,
            contactStiffness=3000,      # suaviza só colisão dado→dado (não afeta chão rígido)
            contactDamping=150,
            physicsClientId=self.client,
        )

        # Nasce no grupo WARM: colide com a bandeja mas não com outros dados,
        # evitando repulsão explosiva quando há sobreposição inicial.
        pb.setCollisionFilterGroupMask(
            body, -1,
            COL_GROUP_WARM, COL_MASK_WARM,
            physicsClientId=self.client,
        )
        self._warm_frames[body] = WARM_FRAMES        

        pb.resetBaseVelocity(
            body,
            linearVelocity=[
                random.uniform(-0.5, 0.5),
                random.uniform(0.5, 1.0),
                random.uniform(-8.0, -6.0),
            ],
            angularVelocity=[
                random.uniform(-8, 8),
                random.uniform(-8, 8),
                random.uniform(-8, 8),
            ],
            physicsClientId=self.client,
        )

        self.dice_ids.append(body)
        self._dice_types[body] = dice_type
        return body

    # ------------------------------------------------------------------
    # Controle de simulação
    # ------------------------------------------------------------------

    def resize_tray(self, half_w: float, half_d: float) -> None:
        """Rebuild the tray walls to the new size."""
        for bid in self._static_ids:
            pb.removeBody(bid, physicsClientId=self.client)
        self._static_ids.clear()
        self._wall_ids.clear()
        self._floor_id = -1
        # Salva novas dimensões e reconstrói
        global TRAY_W, TRAY_D
        TRAY_W = half_w * 2
        TRAY_D = half_d * 2
        self._build_tray()

    def create_dice_body(self, dice_type: str, position: tuple,
                         scale: float = 1.0) -> int:
        self._dice_scale = scale
        return self.add_dice(dice_type)

    def remove_all_dice(self) -> None:
        for bid in self.dice_ids:
            pb.removeBody(bid, physicsClientId=self.client)
        self.dice_ids.clear()
        self._dice_types.clear()
        self._pending_positions.clear()
        self._pending_dice_type = ""
        self._still_frames      = 0
        self._warm_frames.clear()
        self._prev_contacts.clear()

    # ------------------------------------------------------------------
    # Detecção de colisões para áudio
    # ------------------------------------------------------------------

    def poll_collision_events(self) -> list[CollisionEvent]:
        """
        Detecta novas colisões ocorridas no último step() e retorna
        uma lista de CollisionEvent para o motor de áudio processar.

        Somente colisões *novas* (pares que não estavam em contato no tick
        anterior) são emitidas, evitando múltiplos disparos de som enquanto
        os objetos permanecem encostados ou vibram micro-contatos.

        O impulso reportado é a soma das forças normais de todos os pontos
        de contato do par — proxy fiel da energia transferida no impacto.

        Retorna lista vazia se não há dados ou não houve contatos novos.
        """
        if not self.dice_ids:
            return []

        dice_set   = set(self.dice_ids)
        curr_contacts: set[tuple[int, int]] = set()
        events: list[CollisionEvent] = []

        # Acumula impulso por par (PyBullet pode retornar vários pontos
        # de contato para o mesmo par de corpos no mesmo tick)
        pair_impulse: dict[tuple[int, int], float] = {}

        for bid in self.dice_ids:
            contacts = pb.getContactPoints(
                bodyA=bid,
                physicsClientId=self.client,
            )
            if not contacts:
                continue

            for c in contacts:
                # getContactPoints retorna tupla; índices:
                #   [1] = bodyA, [2] = bodyB, [9] = normalForce
                body_a = int(c[1])
                body_b = int(c[2])
                force  = float(c[9])

                # Normaliza par para chave consistente
                pair = (min(body_a, body_b), max(body_a, body_b))
                curr_contacts.add(pair)
                pair_impulse[pair] = pair_impulse.get(pair, 0.0) + abs(force)

        # Emite eventos apenas para pares que são NOVOS neste tick
        new_pairs = curr_contacts - self._prev_contacts
        for pair in new_pairs:
            body_a, body_b = pair
            impulse = pair_impulse.get(pair, 0.0)

            # Classifica superfície
            if body_a == self._floor_id or body_b == self._floor_id:
                surface = Surface.FLOOR
            elif body_a in self._wall_ids or body_b in self._wall_ids:
                surface = Surface.WALL
            elif body_a in dice_set and body_b in dice_set:
                surface = Surface.DICE
            else:
                continue  # contato com objeto desconhecido — ignora

            events.append(CollisionEvent(
                body_a=body_a,
                body_b=body_b,
                surface=surface,
                impulse=impulse,
            ))

        self._prev_contacts = curr_contacts
        return events

    def step(self) -> None:
        for _ in range(SIM_SUBSTEPS):
            pb.stepSimulation(physicsClientId=self.client)

        # Promotes WARM → COLD dice after WARM_FRAMES frames
        graduated = [bid for bid, n in self._warm_frames.items() if n <= 1]
        for bid in graduated:
            pb.setCollisionFilterGroupMask(
                bid, -1,
                COL_GROUP_COLD, COL_MASK_COLD,
                physicsClientId=self.client,
            )
            del self._warm_frames[bid]
        for bid in list(self._warm_frames):
            self._warm_frames[bid] -= 1

    def get_transforms(self) -> list[tuple]:
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
        if not self.dice_ids:
            return True

        all_still = all(
            sum(v * v for v in lv) + sum(v * v for v in av) <= 0.02 ** 2
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
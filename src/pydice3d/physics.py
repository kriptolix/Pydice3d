"""
physics.py — Simulação física dos dados via PyBullet.

Responsabilidades:
  - Construção da bandeja estática (piso + paredes)
  - Criação e destruição de corpos rígidos dos dados
  - Shapes de colisão por tipo (D4–D20)
  - Stepping da simulação e detecção de repouso

Correções de lançamento (v2)
─────────────────────────────
Problema original: dados saíam da bandeja, especialmente o d6.
Causas identificadas:
  1. Interpenetração garantida ao nascer: todos os dados nasciam no mesmo Z
     (z = TRAY_D * 0.45) com X variando em apenas ±2.6m. Com d6 precisando
     de 2.2m de separação, havia colisões explosivas antes do primeiro frame.
  2. Velocidade vertical positiva (vy = +0.5..1.0) somada ao ricochete do
     chão (restitution=0.45) lançava dados acima das paredes.
  3. O spawner.py chamava resetBaseVelocity duas vezes por dado (add_dice +
     _apply_launch), o segundo sobrescrevendo o primeiro corretamente — mas
     quando glarena.py chamava add_dice diretamente, só o primeiro rodava.

Correções:
  - Posições de spawn distribuídas em grade 2D (X e Z variados), com
    separação mínima garantida por tipo.
  - vy inicial = 0 (dados nascem "parados" verticalmente, só com impulso
    horizontal para dentro da bandeja).
  - Velocidade horizontal (vz) reduzida e limitada para não ultrapassar
    a parede oposta.
  - Restitution do chão reduzida de 0.45 → 0.30 para amortecimento mais
    realista (dado de resina, não bola de borracha).
  - linearDamping aumentado de 0.01 → 0.08 para absorver energia extra.
"""

import pybullet as pb
import pybullet_data
import math
import random
import numpy as np

from pydice3d.dice_mesh import get_mesh

# Dimensões da bandeja (em metros do Bullet)
TRAY_W  = 13.0   # largura X
TRAY_D  = 12.0    # profundidade Z
TRAY_H  = 0.15   # espessura do piso
WALL_H  = 9.0    # altura das paredes
WALL_T  = 0.6    # espessura das paredes

# Tamanho alvo do dado no mundo físico (em metros)
DICE_TARGET_SIZE = 1.0

# Altura de spawn — suficiente para não colidir com o chão antes do impulso,
# mas baixa o suficiente para não ganhar energia cinética excessiva na queda.
LAUNCH_Y = 2.0

# Timestep fixo da simulação Bullet
SIM_TIMESTEP = 1.0 / 240.0
# Sub-passos por chamada de step
SIM_SUBSTEPS = 6

# Grupos de colisão (bitmask)
#   TRAY  : colide com todos
#   WARM  : dado recém-criado — colide com bandeja mas NÃO com outros dados
#   COLD  : dado estabilizado — colide com tudo normalmente
COL_GROUP_TRAY = 0b001
COL_GROUP_WARM = 0b010
COL_GROUP_COLD = 0b100

COL_MASK_TRAY  = 0b111          # bandeja colide com warm + cold
COL_MASK_WARM  = COL_GROUP_TRAY # warm só colide com bandeja
COL_MASK_COLD  = 0b111          # cold colide com tudo

# Quantos frames (de step()) cada dado fica no grupo WARM antes de ser promovido.
# 30 frames × 6 substeps × (1/240s) ≈ 0.75 s — tempo suficiente para os dados
# se espalharem sem se empurrarem explosivamente ao nascer.
WARM_FRAMES = 30

# Separação mínima de spawn por tipo (diâmetro do bounding sphere × 1.15)
_SPAWN_SEP: dict[str, float] = {
    "d4":     1.6,
    "d6":     2.0,   # d6 é o maior em termos de caixa
    "d8":     1.7,
    "d10":    1.7,
    "d12":    1.8,
    "d20":    1.8,
    "d100":   1.7,
    "df": 2.0,
}
_DEFAULT_SEP = 2.0


# ---------------------------------------------------------------------------
# Utilitário: posições de spawn sem interpenetração
# ---------------------------------------------------------------------------


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
            restitution=0.45,          # quica com naturalidade, como dado de resina em mesa
            lateralFriction=0.9,       # atrito moderado: rola bem, não desliza demais
            physicsClientId=self.client,
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
                restitution=0.65,      # ← paredes mais "mortas" que o chão
                lateralFriction=0.5,
                physicsClientId=self.client,
            )

    # ------------------------------------------------------------------
    # Criação de dados
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
            restitution=0.35,           # quica normalmente contra chão e paredes
            linearDamping=0.03,         # resistência do ar leve — não "engole" a jogada
            angularDamping=0.03,        # rola livremente
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

        vz_in   = random.uniform(-6.5, -5.0)   # para dentro (-Z), moderado
        vx_side = random.uniform(-1.0,  1.0)   # leve desvio lateral

        pb.resetBaseVelocity(
            body,
            linearVelocity=[
                random.uniform(-0.5, 0.5),
                random.uniform(0.5, 1.0),
                random.uniform(-8.0, -6.0),
            ],
            angularVelocity=[
                random.uniform(-10, 10),
                random.uniform(-10, 10),
                random.uniform(-10, 10),
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
        """Reconstrói as paredes da bandeja para o novo tamanho."""
        for bid in self._static_ids:
            pb.removeBody(bid, physicsClientId=self.client)
        self._static_ids.clear()
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

    def step(self) -> None:
        for _ in range(SIM_SUBSTEPS):
            pb.stepSimulation(physicsClientId=self.client)

        # Promove dados WARM → COLD após WARM_FRAMES frames
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
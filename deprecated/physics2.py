"""
physics.py — Simulação física dos dados via PyBullet.

Responsabilidades:
  - Construção da bandeja estática (piso + paredes)
  - Criação e destruição de corpos rígidos dos dados
  - Shapes de colisão por tipo (D4–D20)
  - Stepping da simulação com física guiada em 3 fases:
      Fase 1 — Rolling:   física pura
      Fase 2 — Settling:  monitoramento de estabilidade
      Fase 3 — Guided Finalization: correção assistida
  - Detecção de repouso e leitura de face
"""

import pybullet as pb
import pybullet_data
import math
import random
import numpy as np

from collision import _COLLISION_VERTS

# ---------------------------------------------------------------------------
# Dimensões da bandeja
# ---------------------------------------------------------------------------
TRAY_W  = 9.0
TRAY_D  = 9.0
TRAY_H  = 0.15
WALL_H  = 4.0
WALL_T  = 0.6

DICE_TARGET_SIZE = 0.8

LAUNCH_Y       = 1.5
LAUNCH_VEL_MAX = 5.0

SIM_TIMESTEP = 1.0 / 240.0
SIM_SUBSTEPS = 6

# ---------------------------------------------------------------------------
# Parâmetros da física guiada
# ---------------------------------------------------------------------------

# Fase 1 → Fase 2 (Rolling → Settling)
LINEAR_SETTLE_THRESHOLD  = 0.35
ANGULAR_SETTLE_THRESHOLD = 1.2
SETTLE_FRAMES_MIN        = 15

# Fase 2 → Fase 3 (Settling → Finalizing)
FINAL_LINEAR_THRESHOLD  = 0.12
FINAL_ANGULAR_THRESHOLD = 0.35

# Alinhamento mínimo da face de chão com -Y para considerar repouso válido.
# 0.985 ≈ cos(10°): abaixo disso o dado está sobre aresta ou vértice.
REQUIRED_ALIGNMENT      = 0.985

# Frames consecutivos com face estável antes de considerar repouso válido
# sem precisar de correção (caminho rápido Rolling→Locked).
SETTLE_STABLE_FRAMES    = 20

# Frames consecutivos acima de REQUIRED_ALIGNMENT necessários para
# considerar que o dado estabilizou naturalmente (sem torque).
SETTLE_GOOD_ALIGN_FRAMES = 10

# Histerese: quantos frames a face dominante precisa ser a mesma antes
# de ser registrada como candidata estável.
SETTLE_FACE_STABLE_FRAMES = 5

# Timeout máximo que o dado pode ficar em settling antes de forçar
# transição para finalizing (evita loop eterno por jitter lento).
SETTLE_TIMEOUT          = 3.0    # segundos

# Fase 3 → Locked (Guided Finalization → Locked)
LOCK_ALIGNMENT       = 0.998     # cos(~3.6°): face de chão alinhada com -Y
LOCK_LINEAR_THRESH   = 0.03      # m/s
LOCK_ANGULAR_THRESH  = 0.03      # rad/s
FINALIZATION_TIMEOUT = 0.5       # segundos antes de forçar lock

# Controlador PD de torque (Proporcional + Derivativo)
# Kp: torque por radiano de erro angular
TORQUE_KP            = 0.25
# Kd: amortece velocidade angular residual no eixo de correção
TORQUE_KD            = 0.08
# Teto absoluto do torque por frame (evita kick violento na entrada)
MAX_GUIDED_TORQUE    = 0.18

# Damping progressivo: cresce linearmente do valor inicial até o máximo
# à medida que o timer de finalização avança até FINALIZATION_TIMEOUT
FINALIZE_ANG_DAMP_START = 0.35
FINALIZE_ANG_DAMP_END   = 0.85
FINALIZE_LIN_DAMP_START = 0.25
FINALIZE_LIN_DAMP_END   = 0.60

# Frames consecutivos dentro dos critérios de lock antes de confirmar
LOCK_CONFIRM_FRAMES  = 6

# Ganho de impulso de separação para pares com penetração (m/s por metro)
PENETRATION_IMPULSE_GAIN = 2.0

# Penetração entre dados: distância negativa = interpenetração
PENETRATION_THRESH   = -0.005

# Solver
SOLVER_ITERATIONS    = 80

# ---------------------------------------------------------------------------
# Normais de face LOCAL por tipo (usadas para detectar face de contato)
# ---------------------------------------------------------------------------

def _build_face_normals():
    """Retorna {dice_type: np.array shape (N,3)} com normais locais por face."""
    import math

    s = 1.0 / math.sqrt(3)
    phi = (1 + math.sqrt(5)) / 2

    # D4 — 4 faces, normal = vértice oposto normalizado
    d4 = np.array([
        [ s,  s,  s],
        [ s, -s, -s],
        [-s,  s, -s],
        [-s, -s,  s],
    ], dtype=np.float32)

    # D6 — 6 faces axiais
    d6 = np.array([
        [ 1,  0,  0], [-1,  0,  0],
        [ 0,  1,  0], [ 0, -1,  0],
        [ 0,  0,  1], [ 0,  0, -1],
    ], dtype=np.float32)

    # D8 — octaedro
    o = 1.0 / math.sqrt(3)
    d8 = np.array([
        [ o,  o,  o], [ o,  o, -o],
        [ o, -o,  o], [ o, -o, -o],
        [-o,  o,  o], [-o,  o, -o],
        [-o, -o,  o], [-o, -o, -o],
    ], dtype=np.float32)

    # D10 — trapezóide: usa vértices (leitura especial via argmin Y)
    # Guardamos as normais aproximadas das 10 faces para uso no settling;
    # a leitura final ainda usa o vértice mais alto via dice_reader.
    from collision import _trapezoid_d10_verts
    verts10 = np.array(_trapezoid_d10_verts(r=1.0), dtype=np.float32)
    # Considera cada vértice de pico como representante de uma face
    d10 = np.zeros((10, 3), dtype=np.float32)
    for i, v in enumerate(verts10):
        n = v / (np.linalg.norm(v) + 1e-8)
        d10[i] = n

    # D12 — dodecaedro
    raw12 = [
        [ 0,  1,  phi], [ 0,  1, -phi],
        [ 0, -1,  phi], [ 0, -1, -phi],
        [ 1,  phi,  0], [ 1, -phi,  0],
        [-1,  phi,  0], [-1, -phi,  0],
        [ phi,  0,  1], [ phi,  0, -1],
        [-phi,  0,  1], [-phi,  0, -1],
    ]
    n12 = math.sqrt(1 + phi * phi)
    d12 = np.array([[x/n12, y/n12, z/n12] for x,y,z in raw12], dtype=np.float32)

    # D20 — icosaedro (20 faces = centróides dos triângulos)
    phi20 = (1 + math.sqrt(5)) / 2
    ico_raw = []
    for s1 in (+1,-1):
        for s2 in (+1,-1):
            ico_raw += [
                [0,           s1,           s2*phi20],
                [s1,          s2*phi20,     0       ],
                [s1*phi20,    0,            s2      ],
            ]
    ico_n = math.sqrt(1 + phi20*phi20)
    ico_v = np.array(ico_raw, dtype=np.float64) / ico_n
    ICO_FACES = [
        (0,4,1),(0,9,4),(9,5,4),(4,5,8),(4,8,1),
        (8,10,1),(8,3,10),(5,3,8),(5,2,3),(2,7,3),
        (7,10,3),(7,6,10),(7,11,6),(11,0,6),(0,1,6),
        (6,1,10),(9,0,11),(9,11,2),(9,2,5),(7,2,11),
    ]
    d20_normals = []
    for a,b,c in ICO_FACES:
        n = (ico_v[a]+ico_v[b]+ico_v[c])/3.0
        nl = np.linalg.norm(n)
        d20_normals.append(n/nl if nl>1e-8 else n)
    d20 = np.array(d20_normals, dtype=np.float32)

    return {"d4":d4, "d6":d6, "d8":d8, "d10":d10, "d12":d12, "d20":d20}

_FACE_NORMALS_LOCAL = _build_face_normals()

# ---------------------------------------------------------------------------
# Faces opostas por tipo (para mapear face-chão → face-topo)
# D4 não tem oposto claro; usa face de chão diretamente.
# ---------------------------------------------------------------------------

def _build_opposite_faces():
    opp = {}
    # D6: face 0↔1, 2↔3, 4↔5
    opp["d6"] = {0:1,1:0,2:3,3:2,4:5,5:4}
    # D8: faces aos pares (±x, ±y, ±z grupos de octaedro)
    opp["d8"] = {0:7,1:6,2:5,3:4,4:3,5:2,6:1,7:0}
    # D12: faces aos pares (dodecaedro tem faces opostas por simetria central)
    opp["d12"] = {i: (i^1) if i%2==0 else (i^1) for i in range(12)}
    # D20: face i oposta a face 19-i (pela simetria do icosaedro)
    opp["d20"] = {i: 19-i for i in range(20)}
    return opp

_FACE_OPPOSITE = _build_opposite_faces()

# ---------------------------------------------------------------------------
# DieState — estado da física guiada por dado
# ---------------------------------------------------------------------------

class DieState:
    __slots__ = (
        "body_id", "phase", "stable_frames",
        "final_face", "target_quat", "finalize_timer",
        # Fase 2 — Settling
        "settle_timer",           # tempo total em settling (s)
        "settle_ground_idx",      # índice da face de chão neste frame
        "settle_face_frames",     # frames consecutivos com a mesma face de chão
        "settle_good_frames",     # frames consecutivos com align >= REQUIRED
        "settle_candidate_idx",   # face de chão estabilizada após histerese
        # Fase 3 — Guided Finalization
        "finalize_ground_idx",    # face de chão travada ao entrar em finalizing
        "finalize_locked_frames", # frames consecutivos dentro dos critérios de lock
    )

    def __init__(self, body_id: int):
        self.body_id        = body_id
        self.phase          = "rolling"
        self.stable_frames  = 0
        self.final_face     = None
        self.target_quat    = None
        self.finalize_timer = 0.0
        # Settling
        self.settle_timer         = 0.0
        self.settle_ground_idx    = -1
        self.settle_face_frames   = 0
        self.settle_good_frames   = 0
        self.settle_candidate_idx = -1
        # Finalizing
        self.finalize_ground_idx    = -1
        self.finalize_locked_frames = 0


# ---------------------------------------------------------------------------
# Utilitários de rotação
# ---------------------------------------------------------------------------

def _quat_to_rot(q) -> np.ndarray:
    rm = pb.getMatrixFromQuaternion(q)
    return np.array(rm, dtype=np.float64).reshape(3, 3)

def _rotation_between(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) que rotaciona v_from para v_to."""
    v_from = v_from / (np.linalg.norm(v_from) + 1e-12)
    v_to   = v_to   / (np.linalg.norm(v_to)   + 1e-12)
    dot    = float(np.clip(np.dot(v_from, v_to), -1, 1))
    if dot > 0.9999:
        return np.array([0,0,0,1], dtype=np.float64)
    if dot < -0.9999:
        perp = np.array([1,0,0]) if abs(v_from[0]) < 0.9 else np.array([0,1,0])
        axis = np.cross(v_from, perp); axis /= np.linalg.norm(axis)
        return np.array([*axis*1.0, 0], dtype=np.float64)
    axis  = np.cross(v_from, v_to)
    axis /= np.linalg.norm(axis)
    half  = math.acos(dot) / 2
    return np.array([*(axis*math.sin(half)), math.cos(half)], dtype=np.float64)

def _quat_mul(q1, q2) -> np.ndarray:
    """Multiplica dois quaternions (x,y,z,w)."""
    x1,y1,z1,w1 = q1
    x2,y2,z2,w2 = q2
    return np.array([
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2,
        w1*w2-x1*x2-y1*y2-z1*z2,
    ], dtype=np.float64)

def _axis_angle_from_quat(q) -> tuple[np.ndarray, float]:
    """Extrai (axis, angle) de um quaternion (x,y,z,w)."""
    x,y,z,w = q
    w = float(np.clip(w, -1, 1))
    angle = 2.0 * math.acos(abs(w))
    s = math.sqrt(max(0, 1 - w*w))
    if s < 1e-8:
        return np.array([1.0,0.0,0.0]), 0.0
    return np.array([x/s, y/s, z/s]), angle

def _detect_ground_face_by_idx(dice_type: str, orn, face_idx: int) -> tuple[int, float]:
    """
    Retorna (face_idx, alignment) para uma face específica em vez da
    face dominante do frame.  Usado na Fase 3 para medir o progresso
    da correção em relação à face de chão travada.

    alignment = dot(world_normal_ground_face, -Y):
      1.0  → face de chão perfeitamente alinhada com o plano
      0.0  → face perpendicular ao plano (dado de lado)
     -1.0  → face de chão apontando para cima (dado de cabeça para baixo)

    Retorna alignment=0.0 (não 1.0) quando face_idx é inválido, para não
    confundir com alinhamento perfeito e travar o dado prematuramente.
    """
    normals = _FACE_NORMALS_LOCAL.get(dice_type)
    if normals is None or face_idx < 0 or face_idx >= len(normals):
        return face_idx, 0.0
    R       = _quat_to_rot(orn)
    world_n = R @ normals[face_idx].astype(np.float64)
    align   = float(np.dot(world_n, np.array([0., -1., 0.])))
    return face_idx, align


def _detect_ground_face(dice_type: str, orn) -> tuple[int, float]:
    """
    Retorna (ground_face_idx, alignment) onde ground_face é a face
    mais alinhada ao vetor -Y (chão).
    alignment = -dot(world_normal, Y+) → quanto a normal aponta para baixo.
    """
    normals = _FACE_NORMALS_LOCAL.get(dice_type)
    if normals is None:
        return 0, 1.0
    R           = _quat_to_rot(orn)
    world_norms = (R @ normals.T).T            # (N, 3)
    down        = np.array([0., -1., 0.])
    dots        = world_norms @ down           # dot com -Y
    idx         = int(np.argmax(dots))
    return idx, float(dots[idx])


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
        pb.setPhysicsEngineParameter(
            numSolverIterations=SOLVER_ITERATIONS,
            physicsClientId=self.client
        )

        self.dice_ids:    list[int]        = []
        self._dice_types: dict[int, str]   = {}
        self._die_states: dict[int, DieState] = {}
        self._static_ids: list[int]        = []
        self._dice_scale: float            = 1.0
        self._still_frames: int            = 0

        self._build_tray()

    # ------------------------------------------------------------------
    def set_dice_scale(self, scale: float):
        self._dice_scale = scale

    # ------------------------------------------------------------------
    # Bandeja estática
    # ------------------------------------------------------------------

    def _static_box(self, half_extents, position, orientation=(0,0,0,1)):
        shape = pb.createCollisionShape(
            pb.GEOM_BOX, halfExtents=half_extents,
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
        hw, hd, ht = TRAY_W/2, TRAY_D/2, TRAY_H/2
        wt, wh     = WALL_T/2, WALL_H/2
        ext        = hw + wt

        floor_id = self._static_box([hw, ht, hd], [0, -ht, 0])
        pb.changeDynamics(floor_id, -1,
            restitution=0.45, lateralFriction=1.2,
            physicsClientId=self.client)

        for pos, he in [
            ([0,  wh, -(hd+wt)], [ext, wh, wt]),
            ([0,  wh,  (hd+wt)], [ext, wh, wt]),
            ([-(hw+wt), wh, 0],  [wt, wh, hd]),
            ([ (hw+wt), wh, 0],  [wt, wh, hd]),
        ]:
            wall = self._static_box(he, pos)
            pb.changeDynamics(wall, -1,
                restitution=0.35, lateralFriction=0.4,
                physicsClientId=self.client)

    # ------------------------------------------------------------------
    # Criação de dados
    # ------------------------------------------------------------------

    def _make_collision_shape(self, dice_type: str) -> int:
        r = DICE_TARGET_SIZE / 2.0
        if dice_type == "d6":
            return pb.createCollisionShape(
                pb.GEOM_BOX, halfExtents=[r,r,r],
                physicsClientId=self.client)
        gen = _COLLISION_VERTS.get(dice_type)
        if gen is None:
            raise ValueError(f"Tipo desconhecido: {dice_type!r}")
        return pb.createCollisionShape(
            pb.GEOM_MESH, vertices=gen(r),
            physicsClientId=self.client)

    def add_dice(self, dice_type: str = "d6") -> int:
        shape = self._make_collision_shape(dice_type)
        x   = random.uniform(-TRAY_W*0.2, TRAY_W*0.2)
        z   = TRAY_D*0.45
        pos = [x, LAUNCH_Y, z]

        axis_raw = [random.gauss(0,1) for _ in range(3)]
        al       = math.sqrt(sum(a*a for a in axis_raw))
        axis     = [a/al for a in axis_raw]
        angle    = random.uniform(0, 2*math.pi)
        orn      = pb.getQuaternionFromAxisAngle(axis, angle,
                                                 physicsClientId=self.client)
        mass     = 0.025

        body = pb.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=shape,
            basePosition=pos,
            baseOrientation=orn,
            physicsClientId=self.client
        )
        pb.changeDynamics(body, -1,
            restitution=0.4, linearDamping=0.01, angularDamping=0.01,
            rollingFriction=0.01, spinningFriction=0.02, lateralFriction=1.5,
            ccdSweptSphereRadius=DICE_TARGET_SIZE*0.25,
            physicsClientId=self.client)
        pb.resetBaseVelocity(body,
            linearVelocity=[
                random.uniform(-0.5, 0.5),
                random.uniform(1.0, 3.0),
                random.uniform(-8.0, -6.0),
            ],
            angularVelocity=[random.uniform(-8,8) for _ in range(3)],
            physicsClientId=self.client)

        self.dice_ids.append(body)
        self._dice_types[body]  = dice_type
        self._die_states[body]  = DieState(body)
        return body

    # ------------------------------------------------------------------
    # Controle de simulação
    # ------------------------------------------------------------------

    def remove_all_dice(self):
        for bid in self.dice_ids:
            pb.removeBody(bid, physicsClientId=self.client)
        self.dice_ids.clear()
        self._dice_types.clear()
        self._die_states.clear()

    def step(self):
        """Avança SIM_SUBSTEPS passos e aplica física guiada em cada dado."""
        for _ in range(SIM_SUBSTEPS):
            pb.stepSimulation(physicsClientId=self.client)

        dt = SIM_TIMESTEP * SIM_SUBSTEPS
        for bid in self.dice_ids:
            self._update_die_state(bid, dt)

    # ------------------------------------------------------------------
    # Física guiada — estado por dado
    # ------------------------------------------------------------------

    def _get_velocity_magnitudes(self, bid: int) -> tuple[float, float]:
        lv, av = pb.getBaseVelocity(bid, physicsClientId=self.client)
        return (
            math.sqrt(sum(v*v for v in lv)),
            math.sqrt(sum(v*v for v in av)),
        )

    def _update_die_state(self, bid: int, dt: float):
        state = self._die_states.get(bid)
        if state is None or state.phase == "locked":
            return

        lin_spd, ang_spd = self._get_velocity_magnitudes(bid)
        _, orn = pb.getBasePositionAndOrientation(bid, physicsClientId=self.client)
        dtype  = self._dice_types[bid]

        # ---- FASE 1: Rolling ----------------------------------------
        if state.phase == "rolling":
            if (lin_spd < LINEAR_SETTLE_THRESHOLD and
                    ang_spd < ANGULAR_SETTLE_THRESHOLD):
                state.stable_frames += 1
            else:
                state.stable_frames = 0

            if state.stable_frames >= SETTLE_FRAMES_MIN:
                state.phase        = "settling"
                state.stable_frames = 0
                print(f"[guided] {dtype} #{bid} → settling")
            return

        # ---- FASE 2: Settling ---------------------------------------
        if state.phase == "settling":
            state.settle_timer += dt

            ground_idx, ground_align = _detect_ground_face(dtype, orn)

            # --- 2a. Histerese de face: só registra mudança após
            #         SETTLE_FACE_STABLE_FRAMES frames consecutivos
            #         com o mesmo índice, evitando flip por micro-giro.
            if ground_idx == state.settle_ground_idx:
                state.settle_face_frames += 1
            else:
                state.settle_face_frames = 1
                state.settle_ground_idx  = ground_idx

            if state.settle_face_frames >= SETTLE_FACE_STABLE_FRAMES:
                state.settle_candidate_idx = ground_idx

            # --- 2b. Contagem de alinhamento bom
            if ground_align >= REQUIRED_ALIGNMENT:
                state.settle_good_frames += 1
            else:
                state.settle_good_frames = 0

            # --- 2c. Detectar estado inválido de repouso
            #   - dado sobre aresta/vértice (alignment baixo)
            #   - interpenetração com outro dado
            on_edge   = ground_align < REQUIRED_ALIGNMENT
            has_penet = self._has_penetration(bid)
            invalid_rest = on_edge or has_penet

            # Log periódico (a cada ~0.5 s) para diagnóstico
            frames_per_log = max(1, int(0.5 / (SIM_TIMESTEP * SIM_SUBSTEPS)))
            if int(state.settle_timer / dt) % frames_per_log == 0:
                state_str = "INVÁLIDO" if invalid_rest else "ok"
                penet_str = " +penetração" if has_penet else ""
                print(f"[settling] {dtype} #{bid}  "
                      f"align={ground_align:.3f}  "
                      f"lin={lin_spd:.3f}  ang={ang_spd:.3f}  "
                      f"face={state.settle_candidate_idx}  "
                      f"[{state_str}{penet_str}]")

            # --- 2d. Caminho rápido: repouso válido sem correção
            #   Dado está plano, parado, sem penetração e a face
            #   candidata está estável há bastante tempo → lock direto.
            fast_lock = (
                not invalid_rest and
                state.settle_good_frames  >= SETTLE_GOOD_ALIGN_FRAMES and
                state.settle_face_frames  >= SETTLE_FACE_STABLE_FRAMES and
                lin_spd < LOCK_LINEAR_THRESH and
                ang_spd < LOCK_ANGULAR_THRESH
            )
            if fast_lock:
                print(f"[settling] {dtype} #{bid} → lock direto "
                      f"(align={ground_align:.3f}, face={state.settle_candidate_idx})")
                self._lock_die(bid, dtype, orn)
                return

            # --- 2e. Transição para Finalizing
            #   Dado quase parado mas em posição inválida, OU
            #   timeout esgotado (jitter residual interminável).
            slow_enough = (
                lin_spd < FINAL_LINEAR_THRESHOLD and
                ang_spd < FINAL_ANGULAR_THRESHOLD
            )
            timed_out = state.settle_timer >= SETTLE_TIMEOUT

            if (slow_enough and invalid_rest) or timed_out:
                reason = "timeout" if timed_out else f"align={ground_align:.3f}"
                print(f"[settling] {dtype} #{bid} → finalizing ({reason})")
                state.phase                 = "finalizing"
                state.stable_frames         = 0
                state.finalize_timer        = 0.0
                state.finalize_locked_frames = 0
                # Trava a face de chão candidata para que o torque corrija
                # sempre em direção à mesma face, mesmo que micro-giros
                # façam _detect_ground_face oscilar durante a correção.
                state.finalize_ground_idx = (
                    state.settle_candidate_idx
                    if state.settle_candidate_idx >= 0
                    else ground_idx
                )
                state.target_quat = self._compute_target_quat_for_face(
                    dtype, orn, state.finalize_ground_idx
                )
                self._set_finalize_dynamics(bid, t=0.0)
            return

        # ---- FASE 3: Guided Finalization ----------------------------
        if state.phase == "finalizing":
            state.finalize_timer += dt
            t_norm = min(state.finalize_timer / FINALIZATION_TIMEOUT, 1.0)

            # 3a. Progresso do damping: aumenta linearmente com o tempo
            #     para desacelerar o dado de forma suave e progressiva.
            self._set_finalize_dynamics(bid, t=t_norm)

            # 3b. Torque PD orientado pela face de chão TRAVADA.
            #     Usa finalize_ground_idx em vez de recalcular a cada
            #     frame, evitando oscilação de alvo durante a correção.
            self._apply_pd_torque(bid, dtype, orn, state, ang_spd)

            # 3c. Resolução de penetração por impulso (não teleporte).
            #     Só executa se houver penetração real; evita custo de
            #     getClosestPoints em frames sem colisão.
            if self._has_penetration(bid):
                self._resolve_penetration_impulse(bid)

            # 3d. Medir alinhamento pela face TRAVADA (não a dominante
            #     do frame, que pode oscilar).
            _, ground_align = _detect_ground_face_by_idx(
                dtype, orn, state.finalize_ground_idx
            )

            # Log periódico
            frames_per_log = max(1, int(0.3 / (SIM_TIMESTEP * SIM_SUBSTEPS)))
            if int(state.finalize_timer / dt) % frames_per_log == 0:
                print(f"[finalizing] {dtype} #{bid}  "
                      f"align={ground_align:.4f}  "
                      f"lin={lin_spd:.3f}  ang={ang_spd:.3f}  "
                      f"t={state.finalize_timer:.2f}s  "
                      f"face={state.finalize_ground_idx}")

            # 3e. Critério de lock: acumula frames consecutivos dentro
            #     dos três limites antes de confirmar, evitando lock
            #     prematuro por frame atípico.
            within_criteria = (
                ground_align >= LOCK_ALIGNMENT and
                lin_spd      <  LOCK_LINEAR_THRESH and
                ang_spd      <  LOCK_ANGULAR_THRESH
            )
            if within_criteria:
                state.finalize_locked_frames += 1
            else:
                state.finalize_locked_frames = 0

            confirmed = state.finalize_locked_frames >= LOCK_CONFIRM_FRAMES
            timed_out = state.finalize_timer >= FINALIZATION_TIMEOUT

            if confirmed or timed_out:
                if timed_out and not confirmed:
                    print(f"[finalizing] {dtype} #{bid} timeout "
                          f"— forçando lock (align={ground_align:.4f})")
                _, orn_final = pb.getBasePositionAndOrientation(
                    bid, physicsClientId=self.client)
                self._lock_die(bid, dtype, orn_final,
                               locked_face_idx=state.finalize_ground_idx)

    # ------------------------------------------------------------------
    # Fase 3 — métodos auxiliares
    # ------------------------------------------------------------------

    def _compute_target_quat_for_face(self, dtype: str, orn,
                                      face_idx: int) -> np.ndarray:
        """
        Quaternion alvo que alinha a normal LOCAL da face `face_idx`
        com o vetor mundo -Y (chão).  Usa a face travada ao entrar em
        finalizing, não a face dominante do frame atual.
        """
        normals = _FACE_NORMALS_LOCAL.get(dtype)
        if normals is None or face_idx < 0:
            return np.array(orn, dtype=np.float64)

        R       = _quat_to_rot(orn)
        local_n = normals[face_idx].astype(np.float64)
        world_n = R @ local_n
        target  = np.array([0., -1., 0.], dtype=np.float64)

        q_corr  = _rotation_between(world_n, target)
        q_curr  = np.array(orn, dtype=np.float64)
        q_target = _quat_mul(q_corr, q_curr)
        norm = np.linalg.norm(q_target)
        return q_target / (norm + 1e-12)

    # Mantido para compatibilidade com o caminho rápido do settling
    def _compute_target_quat(self, dtype: str, orn) -> np.ndarray:
        ground_idx, _ = _detect_ground_face(dtype, orn)
        return self._compute_target_quat_for_face(dtype, orn, ground_idx)

    def _apply_pd_torque(self, bid: int, dtype: str, orn,
                         state: "DieState", ang_spd: float):
        """
        Controlador PD de torque.

        Componente P: torque proporcional ao ângulo entre a normal da
        face de chão TRAVADA e o vetor -Y.  Vira o dado na direção certa.

        Componente D: torque oposto à velocidade angular projetada no
        eixo de correção.  Amortece overshooting e oscilação.

        A face usada é state.finalize_ground_idx — travada ao entrar em
        finalizing — para que o alvo não mude durante a correção.
        """
        normals = _FACE_NORMALS_LOCAL.get(dtype)
        if normals is None or state.finalize_ground_idx < 0:
            return

        R       = _quat_to_rot(orn)
        local_n = normals[state.finalize_ground_idx].astype(np.float64)
        world_n = R @ local_n
        target  = np.array([0., -1., 0.], dtype=np.float64)

        # Eixo e ângulo de erro (componente P)
        err_axis = np.cross(world_n, target)
        err_len  = np.linalg.norm(err_axis)
        if err_len < 1e-8:
            return
        err_axis /= err_len
        err_angle = math.acos(float(np.clip(np.dot(world_n, target), -1.0, 1.0)))

        torque_p = err_axis * (err_angle * TORQUE_KP)

        # Componente D: velocidade angular projetada no eixo de correção
        lv, av = pb.getBaseVelocity(bid, physicsClientId=self.client)
        av_vec  = np.array(av, dtype=np.float64)
        # Projeta a vel. angular no eixo de erro para amortecê-la
        av_proj = np.dot(av_vec, err_axis)
        torque_d = -err_axis * (av_proj * TORQUE_KD)

        torque = torque_p + torque_d
        # Clamp por norma (preserva direção)
        t_norm = np.linalg.norm(torque)
        if t_norm > MAX_GUIDED_TORQUE:
            torque = torque / t_norm * MAX_GUIDED_TORQUE

        pb.applyExternalTorque(
            bid, -1, torque.tolist(), pb.WORLD_FRAME,
            physicsClientId=self.client
        )

    def _set_finalize_dynamics(self, bid: int, t: float = 0.0):
        """
        Damping progressivo: interpola linearmente entre os valores
        iniciais e finais conforme t avança de 0 → 1.

        Chamado tanto ao entrar em finalizing (t=0) quanto a cada frame
        (t=progresso normalizado), garantindo que o dado desacelere
        suavemente sem corte abrupto.
        """
        ang = FINALIZE_ANG_DAMP_START + t * (FINALIZE_ANG_DAMP_END - FINALIZE_ANG_DAMP_START)
        lin = FINALIZE_LIN_DAMP_START + t * (FINALIZE_LIN_DAMP_END - FINALIZE_LIN_DAMP_START)
        pb.changeDynamics(bid, -1,
            angularDamping=float(ang),
            linearDamping=float(lin),
            physicsClientId=self.client)

    # ------------------------------------------------------------------
    # Penetração entre dados
    # ------------------------------------------------------------------

    def _has_penetration(self, bid: int) -> bool:
        """True se qualquer ponto de contato com outro dado tiver distância < PENETRATION_THRESH."""
        for other_bid in self.dice_ids:
            if other_bid == bid:
                continue
            pts = pb.getClosestPoints(
                bid, other_bid, distance=0.01,
                physicsClientId=self.client
            )
            if pts:
                for pt in pts:
                    if pt[8] < PENETRATION_THRESH:
                        return True
        return False

    def _resolve_penetration_impulse(self, bid: int):
        """
        Resolve interpenetrações por impulso de separação em vez de
        teleporte.  Para cada par penetrado, aplica velocidade linear
        oposta à normal de contato, proporcional à profundidade.
        Ambos os dados recebem metade do impulso (conservação de momento).
        """
        for other_bid in self.dice_ids:
            if other_bid == bid:
                continue
            pts = pb.getClosestPoints(
                bid, other_bid, distance=0.0,
                physicsClientId=self.client
            )
            if not pts:
                continue
            for pt in pts:
                depth = pt[8]
                if depth >= PENETRATION_THRESH:
                    continue

                # pt[7] = normal mundo B→A; queremos separar A de B
                normal      = np.array(pt[7], dtype=np.float64)
                sep_speed   = abs(depth) * PENETRATION_IMPULSE_GAIN

                lv_a, av_a = pb.getBaseVelocity(bid,       physicsClientId=self.client)
                lv_b, av_b = pb.getBaseVelocity(other_bid, physicsClientId=self.client)

                # Aplica metade do impulso a cada lado
                new_lv_a = np.array(lv_a) +  normal * sep_speed * 0.5
                new_lv_b = np.array(lv_b) + -normal * sep_speed * 0.5

                pb.resetBaseVelocity(bid,       new_lv_a.tolist(), list(av_a),
                                     physicsClientId=self.client)
                pb.resetBaseVelocity(other_bid, new_lv_b.tolist(), list(av_b),
                                     physicsClientId=self.client)

    # ------------------------------------------------------------------
    # Lock final
    # ------------------------------------------------------------------

    def _lock_die(self, bid: int, dtype: str, orn,
                  locked_face_idx: int = -1):
        """
        Snap invisível: zera velocidade, aplica orientação final correta
        e corrige a posição Y para que a face de chão fique exatamente
        tangente ao plano do piso, sem enterrar nem flutuar o dado.

        Passos (spec "Correção da Fase de Finalização"):
          1. Selecionar a face de chão (locked_face_idx ou detecção live)
          2. Calcular q_correction = rotation_between(world_n_ground, -Y)
          3. Compor q_final = q_correction * q_current
          4. Recalcular corrected_pos: ajustar Y para que a face de
             contato toque exatamente o piso (Y = 0 no sistema Bullet)
          5. resetBaseVelocity([0,0,0], [0,0,0])
          6. resetBasePositionAndOrientation(corrected_pos, q_final)
        """
        pos, _ = pb.getBasePositionAndOrientation(
            bid, physicsClientId=self.client)

        face_idx = locked_face_idx
        if face_idx < 0:
            face_idx, _ = _detect_ground_face(dtype, orn)

        normals = _FACE_NORMALS_LOCAL.get(dtype)
        if normals is not None and 0 <= face_idx < len(normals):
            # --- Passo 2: quaternion corretivo no espaço mundo ---
            # Queremos que a normal local da face de chão aponte para -Y
            # após a rotação final.  Calculamos a rotação que leva a
            # normal mundo ATUAL para -Y e a compomos com a orientação
            # corrente: q_final = q_corr * q_curr.
            R_curr  = _quat_to_rot(orn)
            local_n = normals[face_idx].astype(np.float64)
            world_n = R_curr @ local_n                      # normal no mundo agora
            target  = np.array([0., -1., 0.], dtype=np.float64)  # chão = -Y

            # --- Passo 3: composição ---
            q_corr  = _rotation_between(world_n, target)   # (x,y,z,w)
            q_curr  = np.array(orn, dtype=np.float64)
            q_final = _quat_mul(q_corr, q_curr)
            q_final /= np.linalg.norm(q_final) + 1e-12

            # --- Passo 4: corrected_pos ---
            # Com q_final aplicado, a face de chão aponta exatamente para
            # -Y.  A distância do centro de massa ao plano de contato é o
            # quanto a normal local "alcança" do CoM até a superfície —
            # para dados convexos simétricos isso é DICE_TARGET_SIZE/2.
            # Ajustamos Y para que essa distância seja exatamente o
            # deslocamento do piso (Y=0 no frame Bullet, piso em Y=0).
            r_contact = DICE_TARGET_SIZE / 2.0
            pos_arr   = np.array(pos, dtype=np.float64)
            # Y correto: o CoM fica r_contact acima do piso (Y = 0)
            corrected_pos = np.array([pos_arr[0],
                                      r_contact,
                                      pos_arr[2]], dtype=np.float64)
        else:
            q_final       = np.array(orn, dtype=np.float64)
            corrected_pos = np.array(pos, dtype=np.float64)

        # --- Passos 5 e 6: snap invisível ---
        # O snap é invisível porque no momento do lock:
        #   • velocidade ≈ zero (LOCK_CONFIRM_FRAMES garantiu isso)
        #   • diferença angular é mínima (alignment >= LOCK_ALIGNMENT)
        #   • deslocamento de posição é sub-milimétrico na maioria dos casos
        pb.changeDynamics(bid, -1,
            angularDamping=0.01, linearDamping=0.01,
            physicsClientId=self.client)
        pb.resetBaseVelocity(bid, [0, 0, 0], [0, 0, 0],
                             physicsClientId=self.client)
        pb.resetBasePositionAndOrientation(
            bid, corrected_pos.tolist(), q_final.tolist(),
            physicsClientId=self.client)

        state = self._die_states[bid]
        state.phase = "locked"
        print(f"[guided] {dtype} #{bid} → locked ✓  "
              f"(face_chão={face_idx}  "
              f"pos_y={corrected_pos[1]:.4f})")

    # ------------------------------------------------------------------
    # Transforms para o renderer
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Detecção de repouso global
    # ------------------------------------------------------------------

    def all_sleeping(self) -> bool:
        """
        Retorna True quando todos os dados estão locked OU
        quando todos ficaram parados por ≥30 ticks.
        """
        if not self.dice_ids:
            return True

        all_locked = all(
            self._die_states[bid].phase == "locked"
            for bid in self.dice_ids
        )
        if all_locked:
            return True

        all_still = all(
            sum(v*v for v in lv) + sum(v*v for v in av) <= 0.01**2
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
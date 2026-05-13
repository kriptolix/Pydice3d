"""
spawner.py – Sistema de Spawn de Dados (PyBullet)

Responsabilidade: criar e lançar um conjunto de dados com:
  - Posições iniciais agrupadas (simulando a mão do jogador)
  - Arremesso direcional em leque via resetBaseVelocity do PyBullet
  - Separação mínima garantida ao nascer
  - Velocidades e torques iniciais variados (movimento natural)
  - Seed opcional para reprodutibilidade

Diferenças em relação à versão Verlet
──────────────────────────────────────
  - PhysicsWorld (PyBullet) no lugar de VerletPhysics
  - Velocidades em m/s (padrão SI do PyBullet) no lugar de unidades/passo
  - Lançamento via p.resetBaseVelocity — sem apply_launch_impulse_3d/forces.py
  - Torque inicial via resetBaseVelocity(angularVelocity=...) no lugar de
    apply_random_torque
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pybullet as p

from dice import Dice
from dice_state import DiceState, DiceStatus


# ────────────────────────────────────────────────────────────────────────────
# Parâmetros de spawn
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class SpawnConfig:
    """
    Configuração completa de um lançamento de dados.

    Parâmetros de posição
    ─────────────────────
    spawn_height      : altura Y de nascimento dos dados
    spawn_cluster_xz  : ponto (X, Z) central do grupo ao nascer
    cluster_radius    : raio máximo do grupo ao nascer
    min_separation    : distância mínima entre centros ao nascer

    Parâmetros de arremesso
    ────────────────────────
    throw_azimuth_deg : direção central do arremesso no plano XZ (graus).
                        0° = +X | 90° = +Z | 180° = −X | 270° = −Z.
    throw_spread_deg  : abertura do leque em torno do azimute central (graus).

    Parâmetros de velocidade (m/s — padrão SI do PyBullet)
    ───────────────────────────────────────────────────────
    speed_min / speed_max   : faixa de velocidade de lançamento
    elev_min / elev_max     : faixa de ângulo de elevação (radianos)
    torque_max              : magnitude máxima do torque inicial (rad/s)

    Aleatoriedade
    ─────────────
    seed          : seed para np.random.Generator (None = não-determinístico)
    max_attempts  : tentativas máximas de posicionamento por dado
    """
    spawn_height:      float = 3.0
    spawn_cluster_xz:  tuple = (0.0, 3.5)
    cluster_radius:    float = 1.5
    min_separation:    float = 2.2

    throw_azimuth_deg: float = 270.0
    throw_spread_deg:  float = 30.0

    speed_min:  float = 4.0              # m/s
    speed_max:  float = 7.0              # m/s
    elev_min:   float = math.radians(15)
    elev_max:   float = math.radians(40)
    torque_max: float = 8.0              # rad/s

    seed:         Optional[int] = None
    max_attempts: int           = 60

    def make_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)


# ────────────────────────────────────────────────────────────────────────────
# Resultado do spawn
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class SpawnResult:
    """
    Resultado de um lançamento: dados criados + estados iniciais.

    Attributes
    ----------
    dice_list : dados com corpos físicos registrados no PyBullet
    states    : estados de ciclo-de-vida correspondentes
    seed_used : seed efetivamente usado (útil para replay/log)
    """
    dice_list: list[Dice]
    states:    list[DiceState]
    seed_used: Optional[int]


# ────────────────────────────────────────────────────────────────────────────
# Posicionamento com separação mínima  (inalterado em relação à versão Verlet)
# ────────────────────────────────────────────────────────────────────────────

def _place_positions(
    n: int,
    center_xz: tuple,
    cluster_radius: float,
    min_sep: float,
    rng: np.random.Generator,
    max_attempts: int,
) -> list[np.ndarray]:
    cx, cz = center_xz
    positions: list[np.ndarray] = []

    for _ in range(n):
        placed = False
        for _ in range(max_attempts):
            while True:
                dx = rng.uniform(-cluster_radius, cluster_radius)
                dz = rng.uniform(-cluster_radius, cluster_radius)
                if dx*dx + dz*dz <= cluster_radius**2:
                    break
            candidate = np.array([cx + dx, cz + dz])
            if all(np.linalg.norm(candidate - pos) >= min_sep for pos in positions):
                positions.append(candidate)
                placed = True
                break

        if not placed:
            best = _least_crowded_in_disk(positions, cx, cz, cluster_radius, rng)
            positions.append(best)

    positions = _push_apart(positions, min_sep, cx, cz, cluster_radius * 1.5)
    return positions


def _least_crowded_in_disk(
    existing: list[np.ndarray],
    cx: float, cz: float,
    radius: float,
    rng: np.random.Generator,
    samples: int = 40,
) -> np.ndarray:
    best_pos  = np.array([cx, cz])
    best_dist = -1.0
    for _ in range(samples):
        while True:
            dx = rng.uniform(-radius, radius)
            dz = rng.uniform(-radius, radius)
            if dx*dx + dz*dz <= radius**2:
                break
        c = np.array([cx + dx, cz + dz])
        min_d = min((np.linalg.norm(c - p_) for p_ in existing), default=float('inf'))
        if min_d > best_dist:
            best_dist = min_d
            best_pos  = c
    return best_pos


def _push_apart(
    positions: list[np.ndarray],
    min_sep: float,
    cx: float, cz: float,
    max_radius: float,
    iterations: int = 8,
) -> list[np.ndarray]:
    pos    = [p_.copy() for p_ in positions]
    center = np.array([cx, cz])
    n      = len(pos)
    for _ in range(iterations):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                delta = pos[i] - pos[j]
                dist  = float(np.linalg.norm(delta))
                if dist < min_sep and dist > 1e-6:
                    direction = delta / dist
                    push      = direction * ((min_sep - dist) * 0.5 + 0.01)
                    pos[i]   += push
                    pos[j]   -= push
                    moved     = True
            from_center = pos[i] - center
            d = float(np.linalg.norm(from_center))
            if d > max_radius:
                pos[i] = center + from_center / d * max_radius
        if not moved:
            break
    return pos


# ────────────────────────────────────────────────────────────────────────────
# Lançamento via PyBullet resetBaseVelocity
# ────────────────────────────────────────────────────────────────────────────

def _apply_launch(state: DiceState, cfg: SpawnConfig, rng: np.random.Generator) -> None:
    """
    Define velocidade linear e angular iniciais via resetBaseVelocity.

    Velocidades em m/s / rad/s — unidades nativas do PyBullet.
    """
    half_spread = math.radians(cfg.throw_spread_deg) / 2.0
    azimuth     = math.radians(cfg.throw_azimuth_deg) + rng.uniform(-half_spread, half_spread)
    elevation   = rng.uniform(cfg.elev_min, cfg.elev_max)
    speed       = rng.uniform(cfg.speed_min, cfg.speed_max)

    cos_el = math.cos(elevation)
    vx = speed * cos_el * math.cos(azimuth)
    vy = speed * math.sin(elevation)
    vz = speed * cos_el * math.sin(azimuth)

    # Torque aleatório em todas as direções
    torque = rng.uniform(-cfg.torque_max, cfg.torque_max, size=3)

    p.resetBaseVelocity(
        state.dice.body_id,
        linearVelocity=(vx, vy, vz),
        angularVelocity=torque.tolist(),
    )
    state.status = DiceStatus.ROLLING


# ────────────────────────────────────────────────────────────────────────────
# API pública
# ────────────────────────────────────────────────────────────────────────────

def spawn_dice(
    spec:    dict[str, int],
    physics,                          # PhysicsWorld
    cfg:     Optional[SpawnConfig] = None,
) -> SpawnResult:
    """
    Cria e lança um conjunto de dados.

    Parâmetros
    ----------
    spec    : {tipo: quantidade}, ex: {"d6": 2, "d20": 1}
    physics : instância de PhysicsWorld
    cfg     : configuração de spawn (usa SpawnConfig() padrão se None)

    Retorna
    -------
    SpawnResult com dice_list, states e seed_used

    Exemplo
    -------
    result = spawn_dice({"d6": 3}, physics, SpawnConfig(seed=42))
    """
    if cfg is None:
        cfg = SpawnConfig()

    rng     = cfg.make_rng()
    n_total = sum(spec.values())

    positions_xz = _place_positions(
        n=n_total,
        center_xz=cfg.spawn_cluster_xz,
        cluster_radius=cfg.cluster_radius,
        min_sep=cfg.min_separation,
        rng=rng,
        max_attempts=cfg.max_attempts,
    )

    dice_list: list[Dice] = []
    i = 0
    for dtype, count in spec.items():
        for k in range(count):
            xz  = positions_xz[i]
            pos = (float(xz[0]), cfg.spawn_height, float(xz[1]))
            dice = Dice.create(
                dice_type=dtype,
                position=pos,
                physics=physics,
                name=f"{dtype}_{k+1}",
            )
            dice_list.append(dice)
            i += 1

    states: list[DiceState] = []
    for dice in dice_list:
        state = DiceState.create(dice)
        _apply_launch(state, cfg, rng)
        states.append(state)

    return SpawnResult(
        dice_list=dice_list,
        states=states,
        seed_used=cfg.seed,
    )
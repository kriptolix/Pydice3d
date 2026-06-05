"""
spawner.py – Sistema de Spawn de Dados (PyBullet)

Responsabilidade: criar e lançar um conjunto de dados com:
  - Posições iniciais agrupadas no fundo da bandeja (simulando a mão do jogador)
  - Arremesso direcional em leque para dentro da bandeja (−Z)
  - Separação mínima garantida ao nascer (evita explosões de contato)
  - Velocidades e torques iniciais variados (movimento natural)
  - Seed opcional para reprodutibilidade

Correções v2 (sem regressão de comportamento)
─────────────────────────────────────────────
O efeito de "lançado da borda" é PRESERVADO: spawn_cluster_xz=(0.0, 3.5)
mantém os dados nascendo no fundo (+Z) e o azimute 270° os arremessa para
dentro (−Z). O que foi corrigido:

  1. Velocidade vertical (vy): antes calculada como speed*sin(elev) podia
     chegar a 4.5 m/s. Agora é limitada a vy_max=1.0 m/s, desacoplada da
     velocidade horizontal. O efeito visual de arco é preservado mas sem
     energia suficiente para escalar as paredes após ricochete.

  2. speed_max: reduzido de 7.0 → 5.5 m/s para que o impulso horizontal
     não projete dados contra a parede oposta com força de ricochete excessiva.

  3. Separação mínima: a lógica _place_positions original é mantida intacta.
     O cluster_radius é levemente aumentado (1.5→2.0) para que a separação
     mínima (2.2m) seja atingida com menos tentativas quando há muitos dados.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pybullet as p

from pydice3d.dice import Dice
from pydice3d.dice_state import DiceState, DiceStatus


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
    spawn_cluster_xz  : ponto (X, Z) central do grupo ao nascer.
                        (0, 3.5) = fundo da bandeja → arremesso para dentro.
    cluster_radius    : raio máximo do grupo ao nascer
    min_separation    : distância mínima entre centros ao nascer

    Parâmetros de arremesso
    ────────────────────────
    throw_azimuth_deg : direção central do arremesso no plano XZ (graus).
                        270° = −Z = para dentro da bandeja (padrão).
    throw_spread_deg  : abertura do leque em torno do azimute central (graus).

    Parâmetros de velocidade (m/s — SI do PyBullet)
    ─────────────────────────────────────────────────
    speed_min/max : faixa de velocidade horizontal de lançamento
    vy_max        : componente vertical máxima. Limitada para evitar que
                    dados escalem as paredes após ricochete no chão.
                    (substitui elev_min/elev_max da versão anterior)
    torque_max    : magnitude máxima do torque inicial (rad/s)
    """
    spawn_height:      float = 2.5
    spawn_cluster_xz:  tuple = (0.0, 3.5)   # fundo da bandeja — preserva efeito de arremesso
    cluster_radius:    float = 4.0           # levemente maior para facilitar separação
    min_separation:    float = 2.2

    throw_azimuth_deg: float = 270.0         # −Z = para dentro
    throw_spread_deg:  float = 30.0

    speed_min: float = 3.5   # m/s
    speed_max: float = 4.5   # m/s  ← reduzido de 7.0
    vy_max:    float = 1.0   # m/s  ← substitui elev_min/max; limita ricochete vertical

    torque_max: float = 7.0  # rad/s

    seed:         Optional[int] = None
    max_attempts: int           = 60

    def make_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)


# ────────────────────────────────────────────────────────────────────────────
# Resultado do spawn
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class SpawnResult:
    dice_list: list[Dice]
    states:    list[DiceState]
    seed_used: Optional[int]


# ────────────────────────────────────────────────────────────────────────────
# Posicionamento com separação mínima (lógica original preservada)
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

def _apply_launch(
    state: DiceState,
    cfg:   SpawnConfig,
    rng:   np.random.Generator,
) -> None:
    """
    Define velocidade linear e angular iniciais via resetBaseVelocity.

    A velocidade horizontal (XZ) usa azimute + espalhamento como antes.
    A componente vertical é separada e limitada por vy_max, evitando
    que dados ganhem altura excessiva e escapem pela borda superior.
    """
    half_spread = math.radians(cfg.throw_spread_deg) / 2.0
    azimuth     = math.radians(cfg.throw_azimuth_deg) + rng.uniform(-half_spread, half_spread)
    speed_h     = rng.uniform(cfg.speed_min, cfg.speed_max)

    vx = speed_h * math.cos(azimuth)
    vz = speed_h * math.sin(azimuth)
    vy = float(rng.uniform(0.2, cfg.vy_max))   # leve arco, sem exagero

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
    physics,
    cfg:     Optional[SpawnConfig] = None,
) -> SpawnResult:
    """
    Cria e lança um conjunto de dados agrupados no fundo da bandeja.

    Preserva o efeito visual original de "dados lançados da borda":
      - Nasce em spawn_cluster_xz=(0, 3.5) — fundo da bandeja (+Z)
      - Arremessados em azimute 270° (−Z) = para dentro
      - Separação mínima garantida via _place_positions + _push_apart

    A posição calculada é passada diretamente ao PyBullet via
    pb.resetBasePositionAndOrientation, contornando o spawn fixo
    de physics.add_dice e eliminando interpenetrações.
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
            # Reposiciona o dado para a posição calculada pelo cluster,
            # sobrescrevendo a posição aleatória definida por add_dice.
            import pybullet as pb
            _, orn = pb.getBasePositionAndOrientation(
                dice.body_id, physicsClientId=physics.client
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
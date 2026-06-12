"""
spawner.py – Data Spawning System (PyBullet)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pybullet as pb

from pydice3d.dice import Dice
from pydice3d.dice_state import DiceState, DiceStatus


@dataclass
class SpawnConfig:
    """
    Complete setup of a data launch.    
    """
    spawn_height:      float = 1.8
    spawn_cluster_xz:  tuple = (0.0, 3.5)
    cluster_radius:    float = 4.0
    min_separation:    float = 2.2

    throw_azimuth_deg: float = 270.0         # −Z = inside
    throw_spread_deg:  float = 30.0

    speed_min: float = 3.5   # m/s
    speed_max: float = 4.5   # m/s
    vy_max:    float = 1.0   # m/s

    torque_max: float = 7.0  # rad/s

    seed:         Optional[int] = None
    max_attempts: int = 60

    def make_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)


@dataclass
class SpawnResult:
    dice_list: list[Dice]
    states:    list[DiceState]
    seed_used: Optional[int]


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
            best = _least_crowded_in_disk(
                positions, cx, cz, cluster_radius, rng)
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
    best_pos = np.array([cx, cz])
    best_dist = -1.0
    for _ in range(samples):
        while True:
            dx = rng.uniform(-radius, radius)
            dz = rng.uniform(-radius, radius)
            if dx*dx + dz*dz <= radius**2:
                break
        c = np.array([cx + dx, cz + dz])
        min_d = min((np.linalg.norm(c - p_)
                    for p_ in existing), default=float('inf'))
        if min_d > best_dist:
            best_dist = min_d
            best_pos = c
    return best_pos


def _push_apart(
    positions: list[np.ndarray],
    min_sep: float,
    cx: float, cz: float,
    max_radius: float,
    iterations: int = 8,
) -> list[np.ndarray]:
    pos = [p_.copy() for p_ in positions]
    center = np.array([cx, cz])
    n = len(pos)
    for _ in range(iterations):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                delta = pos[i] - pos[j]
                dist = float(np.linalg.norm(delta))
                if dist < min_sep and dist > 1e-6:
                    direction = delta / dist
                    push = direction * ((min_sep - dist) * 0.5 + 0.01)
                    pos[i] += push
                    pos[j] -= push
                    moved = True
            from_center = pos[i] - center
            d = float(np.linalg.norm(from_center))
            if d > max_radius:
                pos[i] = center + from_center / d * max_radius
        if not moved:
            break
    return pos


def _apply_launch(
    state: DiceState,
    cfg:   SpawnConfig,
    rng:   np.random.Generator,
) -> None:
    """
    Set the initial linear and angular velocities via resetBaseVelocity.
    """
    half_spread = math.radians(cfg.throw_spread_deg) / 2.0
    azimuth = math.radians(cfg.throw_azimuth_deg) + \
        rng.uniform(-half_spread, half_spread)
    speed_h = rng.uniform(cfg.speed_min, cfg.speed_max)

    vx = speed_h * math.cos(azimuth)
    vz = speed_h * math.sin(azimuth)
    vy = float(rng.uniform(0.2, cfg.vy_max))   # leve arco, sem exagero

    torque = rng.uniform(-cfg.torque_max, cfg.torque_max, size=3)

    pb.resetBaseVelocity(
        state.dice.body_id,
        linearVelocity=(vx, vy, vz),
        angularVelocity=torque.tolist(),
    )
    state.status = DiceStatus.ROLLING


def spawn_dice(
    spec:    dict[str, int],
    physics,
    cfg:     Optional[SpawnConfig] = None,
) -> SpawnResult:
    """
    Creates and launches a grouped set of data from the bottom of the tray. 

    Special rule: each d100 in the spec automatically adds 1 d10 of 
    paired units. The d10 partner is marked with the attribute 
    ``dice.d100_partner = True`` so that the UI/app can identify it and 
    combine the results (tens of d100 + units of d10).   
    """
    if cfg is None:
        cfg = SpawnConfig()

    n_d100 = spec.get("d100", 0)
    expanded_spec: dict[str, int] = {}
    for dtype, count in spec.items():
        expanded_spec[dtype] = count
    if n_d100 > 0:
        expanded_spec["d10"] = expanded_spec.get("d10", 0) + n_d100

    rng = cfg.make_rng()
    n_total = sum(expanded_spec.values())

    positions_xz = _place_positions(
        n=n_total,
        center_xz=cfg.spawn_cluster_xz,
        cluster_radius=cfg.cluster_radius,
        min_sep=cfg.min_separation,
        rng=rng,
        max_attempts=cfg.max_attempts,
    )

    dice_list: list[Dice] = []
    # Tracks how many "extra" d10s (d100 partners) are still missing.
    d10_partners_remaining = n_d100

    i = 0
    for dtype, count in expanded_spec.items():
        for k in range(count):
            xz = positions_xz[i]
            pos = (float(xz[0]), cfg.spawn_height, float(xz[1]))
            dice = Dice.create(
                dice_type=dtype,
                position=pos,
                physics=physics,
                name=f"{dtype}_{k+1}",
            )

            if dtype == "d10" and d10_partners_remaining > 0:

                original_d10 = spec.get("d10", 0)
                if k >= original_d10:
                    dice.d100_partner = True
                    d10_partners_remaining -= 1
                else:
                    dice.d100_partner = False
            else:
                dice.d100_partner = False

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

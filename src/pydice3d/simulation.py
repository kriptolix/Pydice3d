"""
simulation.py – Data Simulation Orchestrator
"""

from __future__ import annotations

import math
from typing import Optional, Callable, TYPE_CHECKING

import numpy as np

from pydice3d.physics import PhysicsWorld
from pydice3d.dice_state import DiceState
from pydice3d.spawner import spawn_dice, SpawnConfig
from pydice3d.results import RollMonitor, RollResult
from pydice3d.camera import Camera
from pydice3d.audio import DiceAudioEngine
from pydice3d.scene import RenderScene

# Re-exported so that frontends don't need to import directly from results.
__all__ = ["DiceSimulation", "RollResult"]

if TYPE_CHECKING:
    from pydice3d.audio import CollisionEvent

# How many physics substeps are executed per step() call?
STEPS_PER_TICK: int = 4


class DiceSimulation:
    """
    Data simulation orchestrator.

    on_result: callable(RollResult) optional.
    If None, use the `result` property after `is_done == True`.    
    """

    def __init__(
        self,
        on_result:      Optional[Callable[[RollResult], None]] = None,
        steps_per_tick: int = STEPS_PER_TICK,
        spawn_cfg:      Optional[SpawnConfig] = None,
    ) -> None:
        self._physics = PhysicsWorld()
        self._states:       list[DiceState] = []
        self._monitor:      Optional[RollMonitor] = None
        self._on_result = on_result
        self._steps_per_tick = steps_per_tick
        self._spawn_cfg = spawn_cfg

        self.audio = DiceAudioEngine()

        self._camera = Camera(
            target=np.array([0.0, 0.0, 0.0], dtype=float),
            azimuth_deg=90.0,
            elevation_deg=89.0,
            radius=12.0,
            fov_y_deg=35.0,
            near=0.1,
            far=50.0,
        )

        # Viewport (pixels)
        self._vp_w: int = 660
        self._vp_h: int = 460

        self._simulating: bool = False

        self._scene:  RenderScene | None = None
        self._theme:  str = "light"

    def resize(self, width: int, height: int) -> None:
        """
        Notifies you of the viewport size simulation.
        """
        self._vp_w = max(width, 1)
        self._vp_h = max(height, 1)

        eye_height = float(self._camera.eye_position()[1])
        half_h = math.tan(math.radians(
            self._camera.fov_y_deg / 2)) * eye_height
        aspect = self._vp_w / self._vp_h
        half_w = half_h * aspect

        self._physics.resize_tray(half_w * 0.95, half_h * 0.95)

    def set_camera(
        self,
        eye:     Optional[np.ndarray] = None,
        target:  Optional[np.ndarray] = None,
        fov_deg: Optional[float] = None,
        near:    Optional[float] = None,
        far:     Optional[float] = None,
    ) -> None:
        """
        Adjusts camera parameters without requiring all of them.
        """
        if eye is not None or target is not None:
            new_eye = np.asarray(
                eye,    dtype=float) if eye is not None else self._camera.eye_position()
            new_target = np.asarray(target, dtype=float) if target is not None else np.asarray(
                self._camera.target, dtype=float)
            self._camera = Camera.from_eye_target(
                eye=new_eye,
                target=new_target,
                fov_y_deg=fov_deg if fov_deg is not None else self._camera.fov_y_deg,
                near=near if near is not None else self._camera.near,
                far=far if far is not None else self._camera.far,
            )
            # from_eye_target can produce elevation=90° if eye is
            # directly above the target — clamp to avoid singularity

            self._camera.elevation_deg = float(
                np.clip(self._camera.elevation_deg,
                        self._camera._ELEV_MIN,
                        self._camera._ELEV_MAX)
            )
        else:
            if fov_deg is not None:
                self._camera.fov_y_deg = float(fov_deg)
            if near is not None:
                self._camera.near = float(near)
            if far is not None:
                self._camera.far = float(far)

    def roll(
        self,
        spec:      dict[str, int],
        cfg:       Optional[SpawnConfig] = None,
        on_result: Optional[Callable[[RollResult], None]] = None,
        theme:     Optional[str] = None,
    ) -> None:
        """
        Starts a new roll, discarding any previous rolls.        
        """

        self._simulating = False
        self._physics.remove_all_dice()
        self._states.clear()
        self._monitor = None
        self._scene = None

        if theme is not None:
            self._theme = theme

        effective_cfg = cfg or self._spawn_cfg or SpawnConfig()
        result = spawn_dice(
            spec=spec, physics=self._physics, cfg=effective_cfg)
        self._states = result.states

        if self._theme or theme is not None:
            self._scene = RenderScene.from_states(self._states, self._theme)

        callback = on_result or self._on_result
        self._monitor = RollMonitor(self._states, on_complete=callback)
        self._simulating = True

    def step(self) -> None:
        """
        Advances the simulation by one tick (steps_per_tick physics substeps).        
        """
        if not self._simulating or not self._states:
            return

        collision_events = []
        for _ in range(self._steps_per_tick):
            pre_vel = self._physics._snapshot_velocities()
            self._physics.step()
            for s in self._states:
                s.update_status()
            collision_events.extend(
                self._physics.poll_collision_events(pre_vel)
            )

        # Audio: triggers only the event with the highest impulse per pair
        # (prevents multiple triggers of the same impact in consecutive substeps)
        best: dict[tuple[int, int], CollisionEvent] = {}
        for evt in collision_events:
            pair = (min(evt.body_a, evt.body_b), max(evt.body_a, evt.body_b))
            if pair not in best or evt.impulse > best[pair].impulse:
                best[pair] = evt
        for evt in best.values():
            self.audio.on_collision(evt)

        self.audio.on_rolling(self._states)
        self.audio.tick()

        if self._scene is not None:
            self._scene.update(self._states, alpha=1.0)

        if self._monitor:
            self._monitor.tick()

        if self._physics.all_sleeping():
            self._simulating = False
            self.audio.on_roll_complete()

    def stop(self) -> None:

        self._simulating = False
        self.audio.stop_all()

    def reset(self) -> None:

        self._simulating = False
        self._physics.remove_all_dice()
        self._states.clear()
        self._monitor = None
        self._scene = None
        self.audio.stop_all()

    @property
    def is_rolling(self) -> bool:
        return self._simulating

    @property
    def is_done(self) -> bool:
        return self._monitor is not None and self._monitor.completed

    @property
    def result(self) -> Optional[RollResult]:
        return self._monitor.result if self._monitor else None

    @property
    def partial_result(self) -> Optional[RollResult]:
        return self._monitor.partial_result() if self._monitor else None

    @property
    def progress(self) -> float:
        return self._monitor.progress if self._monitor else 0.0

    @property
    def states(self) -> list[DiceState]:
        """List of DiceState data that is active (readable)."""
        return self._states

    @property
    def scene(self) -> "RenderScene | None":
        return self._scene

    @property
    def dice_types(self) -> list[str]:
        """
        List of active data types, in the same order as scene.dice_renders.
        Useful for allocating GPU resources (VAOs, wireframes) without navigating through states.
        """
        return [s.dice.dice_type for s in self._states]

    @property
    def theme(self) -> str:
        return self._theme

    @theme.setter
    def theme(self, value: str) -> None:

        self._theme = value
        if self._scene is not None and self._states:
            self._scene = RenderScene.from_states(self._states, self._theme)

    @property
    def physics(self) -> PhysicsWorld:
        return self._physics

    @property
    def audio_enabled(self) -> bool:
        return self.audio.enabled

    @audio_enabled.setter
    def audio_enabled(self, value: bool) -> None:

        self.audio.enabled = bool(value)
        if not self.audio.enabled:
            self.audio.stop_all()

    @property
    def audio_volume(self) -> float:
        """Global audio volume [0.0, 1.0]."""
        return self.audio.master_volume

    @audio_volume.setter
    def audio_volume(self, value: float) -> None:

        self.audio.master_volume = float(np.clip(value, 0.0, 1.0))

    @property
    def camera(self) -> Camera:
        return self._camera

    def view_matrix(self) -> np.ndarray:
        return self._camera.view_matrix()

    def projection_matrix(self) -> np.ndarray:
        return self._camera.projection_matrix(self._vp_w, self._vp_h)

    def view_projection(self) -> np.ndarray:
        return self._camera.view_projection(self._vp_w, self._vp_h)

    def camera_position(self) -> np.ndarray:
        return self._camera.position

    def __del__(self) -> None:
        # PhysicsWorld already performs pb.disconnect in the __del__ itself,
        # but we guarantee a clean reset if the object is collected early.
        try:
            self.reset()
        except Exception:
            pass

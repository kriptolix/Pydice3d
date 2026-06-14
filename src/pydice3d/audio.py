"""
audio.py – Motor de Áudio para Rolagem de Dados. Recebe eventos de colisão vindos do 
PhysicsWorld e reproduz samples com volume e pitch variados via sounddevice + PortAudio.
"""

from __future__ import annotations

import random
import threading
import wave
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False


_AUDIO_DIR = Path(__file__).parent / "assets" / "audio"

IMPACT_THRESHOLD: float = 0.3
IMPACT_REFERENCE: float = 10.0

PITCH_MIN: float = 0.92
PITCH_MAX: float = 1.08

ROLLING_SPEED_THRESHOLD: float = 1.0

# Reference speed for maximum rolling volume
ROLLING_SPEED_REFERENCE: float = 15.0

ROLLING_VOL_MIN: float = 0.05

COOLDOWN_FLOOR: float = 0.08
COOLDOWN_WALL:  float = 0.04
COOLDOWN_DICE:  float = 0.06

# Assumed simulation rate for converting seconds into ticks
_SIM_HZ: float = 60.0

MAX_CONCURRENT_SOUNDS: int = 8


class Surface(Enum):
    FLOOR = auto()
    WALL = auto()
    DICE = auto()


@dataclass(frozen=True)
class CollisionEvent:

    body_a:  int
    body_b:  int
    surface: Surface
    impulse: float


@dataclass
class WavBuffer:

    data:        np.ndarray   # float32
    sample_rate: int
    n_channels:  int

    @classmethod
    def load(cls, path: Path) -> Optional["WavBuffer"]:

        try:
            with wave.open(str(path), "rb") as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()   # bytes por amostra
                sample_rate = wf.getframerate()
                raw = wf.readframes(wf.getnframes())

            # Decodifica para int dependendo da bit depth
            if sample_width == 1:
                pcm = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                data = (pcm - 128.0) / 128.0          # uint8 → [-1, 1]
            elif sample_width == 2:
                pcm = np.frombuffer(raw, dtype=np.int16)
                data = pcm.astype(np.float32) / 32768.0
            elif sample_width == 3:
                # 24-bit: não há dtype nativo — expande para int32
                raw_arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
                pcm = (raw_arr[:, 0].astype(np.int32)
                       | (raw_arr[:, 1].astype(np.int32) << 8)
                       | (raw_arr[:, 2].astype(np.int32) << 16))
                # Extensão de sinal
                pcm[pcm >= 0x800000] -= 0x1000000
                data = pcm.astype(np.float32) / 8388608.0
            elif sample_width == 4:
                pcm = np.frombuffer(raw, dtype=np.int32)
                data = pcm.astype(np.float32) / 2147483648.0
            else:
                return None

            data = np.clip(data, -1.0, 1.0)

            if n_channels == 2:
                data = data.reshape(-1, 2)

            return cls(data.copy(), sample_rate, n_channels)

        except Exception as e:
            print(f"[audio] error loading {path}: {e}")
            return None

    def with_pitch(self, pitch: float) -> "WavBuffer":
        """
        Retorna cópia com pitch alterado por resampling linear.
        """
        if abs(pitch - 1.0) < 1e-4:
            return self

        # Always operates in mono-flat to simplify resampling;
        # Reconstructs stereo at the end if necessary.
        mono = self.data.ravel()  # view flat
        n_src = len(mono)
        n_dst = max(1, int(round(n_src / pitch)))

        idx = np.linspace(0, n_src - 1, n_dst, dtype=np.float64)
        lo = idx.astype(np.int64)
        hi = np.clip(lo + 1, 0, n_src - 1)
        frac = (idx - lo).astype(np.float32)

        resampled = mono[lo] * (1.0 - frac) + mono[hi] * frac   # float32

        if self.n_channels == 2:
            # Ensures a pair of samples for stereo reconstruction.
            if len(resampled) % 2 != 0:
                resampled = resampled[:-1]
            resampled = resampled.reshape(-1, 2)

        return WavBuffer(resampled, self.sample_rate, self.n_channels)

    def with_volume(self, volume: float) -> "WavBuffer":
        """Returns a scaled copy. volume ∈ [0.0, 1.0]."""
        v = float(np.clip(volume, 0.0, 1.0))
        return WavBuffer((self.data * v).astype(np.float32),
                         self.sample_rate, self.n_channels)

    def as_output_array(self) -> np.ndarray:
        """
        Returns a C-contiguous float array of 32 units ready for sounddevice.play().        
        """
        arr = np.ascontiguousarray(self.data, dtype=np.float32)
        if self.n_channels == 1 and arr.ndim == 2:
            arr = arr.ravel()
        return arr


class _SoundSlot:
    """
    Stream sound device pre-opened and in standby.
    """

    def __init__(self, sample_rate: int = 44100, channels: int = 1) -> None:
        self._lock = threading.Lock()
        self._buf:    Optional[np.ndarray] = None   # float32
        self._pos:    int = 0
        self._active: bool = False

        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            blocksize=256,
            callback=self._callback,
        )
        self._stream.start()

    def play(self, buf: np.ndarray) -> None:
        """Line up `buf` (float32) for immediate reproduction."""
        with self._lock:
            self._buf = np.ascontiguousarray(buf, dtype=np.float32)
            self._pos = 0
            self._active = True

    @property
    def is_free(self) -> bool:
        return not self._active

    def stop(self) -> None:
        with self._lock:
            self._buf = None
            self._pos = 0
            self._active = False

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        with self._lock:
            if not self._active or self._buf is None:
                outdata[:] = 0
                return

            buf = self._buf
            pos = self._pos
            n_src = len(buf)
            need = frames

            written = 0
            while need > 0:
                avail = n_src - pos
                chunk = min(avail, need)
                outdata[written:written + chunk] = buf[pos:pos + chunk].reshape(
                    chunk, -1) if outdata.ndim == 2 else buf[pos:pos + chunk]
                pos += chunk
                written += chunk
                need -= chunk
                if pos >= n_src:
                    # Buffer exhausted — clear the rest and mark a free slot.
                    outdata[written:] = 0
                    self._buf = None
                    self._pos = 0
                    self._active = False
                    return

            self._pos = pos


class DiceAudioEngine:
    """
    Audio engine for data scrolling using sounddevice. 
    """

    def __init__(
        self,
        audio_dir:     Path = _AUDIO_DIR,
        enabled:       bool = True,
        master_volume: float = 1.0,
    ) -> None:

        self.enabled = enabled and _SD_AVAILABLE
        self.master_volume = float(np.clip(master_volume, 0.0, 1.0))

        self._samples: dict[str, Optional[WavBuffer]] = {}
        self._load_assets(audio_dir)

        self._slots:       list[_SoundSlot] = []
        self._slots_ready: bool = False
        self._init_slots()

        # Cooldown per pair of bodies: (smaller_id, larger_id) → remaining ticks
        self._cooldown: dict[tuple[int, int], int] = {}
        self._cooldown_ticks_floor = max(1, int(COOLDOWN_FLOOR * _SIM_HZ))
        self._cooldown_ticks_wall = max(1, int(COOLDOWN_WALL * _SIM_HZ))
        self._cooldown_ticks_dice = max(1, int(COOLDOWN_DICE * _SIM_HZ))

        # Rolling loop
        self._rolling_stream: Optional[sd.OutputStream] = None
        self._rolling_buf:    Optional[np.ndarray] = None
        self._rolling_pos:    int = 0
        self._rolling_lock:   threading.Lock = threading.Lock()

    def _load_assets(self, audio_dir: Path) -> None:
        for name in ("hit_floor_soft", "hit_floor_hard",
                     "hit_wall", "hit_dice", "rolling_loop"):
            path = audio_dir / f"{name}.wav"
            buf = WavBuffer.load(path)
            if buf is None:
                print(
                    f"[audio] warning: '{name}.wav' not found in {audio_dir}")
            self._samples[name] = buf

    def _init_slots(self) -> None:

        if self._slots_ready or not self.enabled:
            return
        try:
            ref = next((b for b in self._samples.values()
                       if b is not None), None)
            sr = ref.sample_rate if ref else 44100
            ch = ref.n_channels if ref else 1
            self._slots = [_SoundSlot(sr, ch)
                           for _ in range(MAX_CONCURRENT_SOUNDS)]
            self._slots_ready = True
        except Exception as e:
            print(f"[audio] Error creating stream pool.: {e}")
            self.enabled = False

    def _ensure_slots(self) -> None:

        if not self._slots_ready:
            self._init_slots()

    def on_collision(self, event: CollisionEvent) -> None:
        """
        Processes a CollisionEvent coming from PhysicsWorld.
        """
        if not self.enabled:
            return
        if event.impulse < IMPACT_THRESHOLD:
            return

        self._ensure_slots()

        pair = (min(event.body_a, event.body_b),
                max(event.body_a, event.body_b))
        if self._cooldown.get(pair, 0) > 0:
            return

        ticks = {
            Surface.FLOOR: self._cooldown_ticks_floor,
            Surface.WALL:  self._cooldown_ticks_wall,
            Surface.DICE:  self._cooldown_ticks_dice,
        }.get(event.surface, self._cooldown_ticks_floor)
        self._cooldown[pair] = ticks

        buf = self._samples.get(self._sample_for(event))
        if buf is None:
            return

        volume = float(np.clip(event.impulse / IMPACT_REFERENCE, 0.1, 1.0))
        volume *= self.master_volume
        pitch = random.uniform(PITCH_MIN, PITCH_MAX)

        slot = next((s for s in self._slots if s.is_free), None)
        if slot is None:
            return

        processed = buf.with_pitch(pitch).with_volume(volume)
        slot.play(processed.as_output_array())

    def on_rolling(self, states: list) -> None:

        if not self.enabled:
            return

        buf = self._samples.get("rolling_loop")
        if buf is None:
            return

        speeds = [float(np.linalg.norm(s.angular_velocity))
                  for s in states if not s.is_resting]

        if not speeds or max(speeds) < ROLLING_SPEED_THRESHOLD:
            self._stop_rolling()
            return

        avg_speed = sum(speeds) / len(speeds)
        vol = float(np.clip(avg_speed / ROLLING_SPEED_REFERENCE,
                            ROLLING_VOL_MIN, 1.0)) * self.master_volume
        self._ensure_rolling_stream(buf, vol)

    def on_roll_complete(self) -> None:
        self._stop_rolling()

    def tick(self) -> None:
        """Advances cooldown counters. Call once per tick."""
        expired = [p for p, n in self._cooldown.items() if n <= 1]
        for p in expired:
            del self._cooldown[p]
        for p in list(self._cooldown):
            self._cooldown[p] -= 1

    def stop_all(self) -> None:
        self._stop_rolling()
        for slot in self._slots:
            slot.stop()

    def _sample_for(self, event: CollisionEvent) -> str:
        if event.surface == Surface.DICE:
            return "hit_dice"
        if event.surface == Surface.WALL:
            return "hit_wall"
        return "hit_floor_soft" if event.impulse < 2.0 else "hit_floor_hard"

    def _ensure_rolling_stream(self, buf: WavBuffer, volume: float) -> None:
        new_arr = buf.with_volume(volume).as_output_array()
        with self._rolling_lock:
            self._rolling_buf = new_arr

        if self._rolling_stream is not None and self._rolling_stream.active:
            return

        n_ch = buf.n_channels

        def _cb(outdata: np.ndarray, frames: int, _t, _s) -> None:
            with self._rolling_lock:
                src = self._rolling_buf
            if src is None:
                outdata[:] = 0
                return
            n_src = len(src)
            pos, need, written = self._rolling_pos, frames, 0
            while need > 0:
                chunk = min(n_src - pos, need)
                if outdata.ndim == 2:
                    outdata[written:written + chunk] = src[pos:pos +
                                                           chunk].reshape(chunk, -1)
                else:
                    outdata[written:written + chunk] = src[pos:pos + chunk]
                pos = (pos + chunk) % n_src
                written += chunk
                need -= chunk
            self._rolling_pos = pos

        try:
            self._rolling_stream = sd.OutputStream(
                samplerate=buf.sample_rate, channels=n_ch,
                dtype="float32", blocksize=512, callback=_cb,
            )
            self._rolling_stream.start()
        except Exception as e:
            print(f"[audio] Error starting rolling stream.: {e}")
            self._rolling_stream = None

    def _stop_rolling(self) -> None:
        if self._rolling_stream is not None:
            try:
                self._rolling_stream.stop()
                self._rolling_stream.close()
            except Exception:
                pass
            self._rolling_stream = None
        with self._rolling_lock:
            self._rolling_buf = None
        self._rolling_pos = 0

    def __del__(self) -> None:
        try:
            self._stop_rolling()
            for slot in self._slots:
                slot.close()
        except Exception:
            pass

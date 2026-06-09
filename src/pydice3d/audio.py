"""
audio.py – Motor de Áudio para Rolagem de Dados

Camada de áudio independente de OpenGL e GTK.
Recebe eventos de colisão vindos do PhysicsWorld e reproduz
samples com volume e pitch variados via sounddevice + PortAudio.

Por que sounddevice em vez de simpleaudio
──────────────────────────────────────────
simpleaudio depende de ALSA, que foi removido de distros Linux modernas
(Fedora, Ubuntu 22+) em favor de PipeWire/PulseAudio. sounddevice usa
PortAudio, que abstrai o backend de áudio do SO e funciona em:
  Linux   : PipeWire, PulseAudio, JACK, ALSA (qualquer um disponível)
  macOS   : CoreAudio
  Windows : WASAPI / DirectSound

Instalação
──────────
    pip install sounddevice

Arquitetura
───────────
PhysicsWorld.poll_collision_events()
    └─► CollisionEvent(body_a, body_b, surface, impulse)
            └─► DiceAudioEngine.on_collision(event)
                    ├─► descarta se impulse < threshold
                    ├─► cooldown por par de corpos
                    ├─► mapeia surface → sample
                    ├─► mapeia impulse → volume
                    ├─► pitch shift por resampling linear
                    └─► sounddevice.play() em thread separada

DiceAudioEngine.on_rolling(states)
    └─► velocidade angular média dos dados em movimento
    └─► loop contínuo via callback de stream

Pitch shift
───────────
sounddevice trabalha com float32 nativamente. O resampling linear
interpola o array PCM para um comprimento diferente antes de enviar
ao dispositivo, mantendo a taxa de saída constante:
  pitch > 1.0 → array menor  → soa mais agudo
  pitch < 1.0 → array maior  → soa mais grave

Estrutura de assets esperada
─────────────────────────────
assets/audio/
  hit_floor_soft.wav      impacto suave no piso
  hit_floor_hard.wav      impacto forte no piso
  hit_wall.wav            batida na parede
  hit_dice.wav            dado contra dado
  rolling_loop.wav        som contínuo de rolar (loop)

WAV: qualquer sample rate, mono ou estéreo, qualquer bit depth.
O motor converte internamente para float32 e para o sample rate
do dispositivo padrão.

O motor funciona silenciosamente se sounddevice não estiver instalado
ou se algum arquivo de audio estiver ausente.
"""

from __future__ import annotations

import random
import threading
import wave
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False


# ────────────────────────────────────────────────────────────────────────────
# Caminho dos assets
# ────────────────────────────────────────────────────────────────────────────

_AUDIO_DIR = Path(__file__).parent / "assets" / "sounds"


# ────────────────────────────────────────────────────────────────────────────
# Tipos de superfície
# ────────────────────────────────────────────────────────────────────────────

class Surface(Enum):
    FLOOR = auto()   # piso da bandeja
    WALL  = auto()   # parede lateral
    DICE  = auto()   # dado contra dado


# ────────────────────────────────────────────────────────────────────────────
# Evento de colisão  (produzido por PhysicsWorld.poll_collision_events)
# ────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CollisionEvent:
    """
    Representa uma nova colisão detectada pelo motor de física.

    Campos
    ──────
    body_a  : body ID do primeiro objeto (sempre um dado)
    body_b  : body ID do segundo objeto (dado, piso ou parede)
    surface : tipo de superfície colidida
    impulse : magnitude do impulso normal (N·s) — proxy de intensidade
    """
    body_a:  int
    body_b:  int
    surface: Surface
    impulse: float


# ────────────────────────────────────────────────────────────────────────────
# WavBuffer — PCM float32 normalizado
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class WavBuffer:
    """
    PCM decodificado de um WAV, armazenado como float32 em [-1.0, 1.0].

    sounddevice trabalha nativamente com float32, então mantemos os dados
    nesse formato desde o carregamento — sem conversão a cada playback.

    data        : shape (n_samples,) para mono, (n_samples, 2) para estéreo
    sample_rate : taxa original do arquivo (usada para resampling de pitch)
    n_channels  : 1 ou 2
    """
    data:        np.ndarray   # float32
    sample_rate: int
    n_channels:  int

    @classmethod
    def load(cls, path: Path) -> Optional["WavBuffer"]:
        """
        Carrega WAV e converte para float32.
        Suporta 8, 16, 24 e 32-bit PCM.
        Retorna None se o arquivo não existir ou for inválido.
        """
        try:
            with wave.open(str(path), "rb") as wf:
                n_channels   = wf.getnchannels()
                sample_width = wf.getsampwidth()   # bytes por amostra
                sample_rate  = wf.getframerate()
                raw          = wf.readframes(wf.getnframes())

            # Decodifica para int dependendo da bit depth
            if sample_width == 1:
                pcm = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                data = (pcm - 128.0) / 128.0          # uint8 → [-1, 1]
            elif sample_width == 2:
                pcm  = np.frombuffer(raw, dtype=np.int16)
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
                pcm  = np.frombuffer(raw, dtype=np.int32)
                data = pcm.astype(np.float32) / 2147483648.0
            else:
                return None

            data = np.clip(data, -1.0, 1.0)

            if n_channels == 2:
                data = data.reshape(-1, 2)

            return cls(data.copy(), sample_rate, n_channels)

        except Exception as e:
            print(f"[audio] erro ao carregar {path}: {e}")
            return None

    # ── Transformações ────────────────────────────────────────────────────────

    def with_pitch(self, pitch: float) -> "WavBuffer":
        """
        Retorna cópia com pitch alterado por resampling linear.

          pitch > 1.0 → array menor  → soa mais agudo
          pitch < 1.0 → array maior  → soa mais grave

        O sample_rate do objeto resultante é idêntico ao original —
        sounddevice sempre reproduz na mesma taxa; o que muda é o
        conteúdo interpolado.
        """
        if abs(pitch - 1.0) < 1e-4:
            return self

        # Opera sempre em mono-flat para simplificar o resampling;
        # reconstrói estéreo ao final se necessário.
        mono = self.data.ravel()  # view flat (não copia)
        n_src = len(mono)
        n_dst = max(1, int(round(n_src / pitch)))

        idx  = np.linspace(0, n_src - 1, n_dst, dtype=np.float64)
        lo   = idx.astype(np.int64)
        hi   = np.clip(lo + 1, 0, n_src - 1)
        frac = (idx - lo).astype(np.float32)

        resampled = mono[lo] * (1.0 - frac) + mono[hi] * frac   # float32

        if self.n_channels == 2:
            # Garante par de amostras para reconstruir estéreo
            if len(resampled) % 2 != 0:
                resampled = resampled[:-1]
            resampled = resampled.reshape(-1, 2)

        return WavBuffer(resampled, self.sample_rate, self.n_channels)

    def with_volume(self, volume: float) -> "WavBuffer":
        """Retorna cópia com amplitude escalada. volume ∈ [0.0, 1.0]."""
        v = float(np.clip(volume, 0.0, 1.0))
        return WavBuffer((self.data * v).astype(np.float32),
                         self.sample_rate, self.n_channels)

    def as_output_array(self) -> np.ndarray:
        """
        Retorna array float32 C-contíguo pronto para sounddevice.play().
        sounddevice exige shape (n,) para mono ou (n, 2) para estéreo.
        """
        arr = np.ascontiguousarray(self.data, dtype=np.float32)
        if self.n_channels == 1 and arr.ndim == 2:
            arr = arr.ravel()
        return arr


# ────────────────────────────────────────────────────────────────────────────
# Parâmetros de áudio
# ────────────────────────────────────────────────────────────────────────────

# Impulso mínimo (N·s) para disparar um som de colisão
IMPACT_THRESHOLD: float = 0.3

# Impulso de referência para volume máximo
IMPACT_REFERENCE: float = 10.0

# Faixa de pitch aleatório para colisões
PITCH_MIN: float = 0.92
PITCH_MAX: float = 1.08

# Velocidade angular (rad/s) mínima para iniciar o loop de rolling
ROLLING_SPEED_THRESHOLD: float = 1.0

# Velocidade de referência para volume máximo do rolling
ROLLING_SPEED_REFERENCE: float = 15.0

# Volume mínimo do loop de rolling (evita silêncio abrupto na desaceleração)
ROLLING_VOL_MIN: float = 0.05

# Cooldown mínimo entre sons do mesmo par de corpos (segundos).
# PyBullet gera vários pontos de contato por colisão no mesmo tick;
# o cooldown garante que só um som seja disparado por impacto real.
COOLDOWN_SAME_PAIR: float = 0.08

# Taxa de simulação assumida para converter segundos em ticks
_SIM_HZ: float = 60.0

# Máximo de sons simultâneos — evita sobrecarga em cascatas de colisão
MAX_CONCURRENT_SOUNDS: int = 8


# ────────────────────────────────────────────────────────────────────────────
# DiceAudioEngine
# ────────────────────────────────────────────────────────────────────────────

class DiceAudioEngine:
    """
    Motor de áudio para rolagem de dados usando sounddevice.

    Parâmetros
    ----------
    audio_dir     : diretório com os arquivos WAV.
                    Padrão: assets/audio/ relativo a este módulo.
    enabled       : False desabilita todo processamento (headless/testes).
    master_volume : escala global de volume [0.0, 1.0].
    """

    def __init__(
        self,
        audio_dir:     Path  = _AUDIO_DIR,
        enabled:       bool  = True,
        master_volume: float = 1.0,
    ) -> None:
        self.enabled       = enabled and _SD_AVAILABLE
        self.master_volume = float(np.clip(master_volume, 0.0, 1.0))

        # Buffers WAV em memória — None se arquivo ausente
        self._samples: dict[str, Optional[WavBuffer]] = {}
        self._load_assets(audio_dir)

        # Sons de impacto em andamento: lista de threads não-bloqueantes
        self._active_threads: list[threading.Thread] = []

        # Cooldown por par de corpos: (id_menor, id_maior) → ticks restantes
        self._cooldown: dict[tuple[int, int], int] = {}
        self._cooldown_ticks = max(1, int(COOLDOWN_SAME_PAIR * _SIM_HZ))

        # Estado do loop de rolling
        self._rolling_stream: Optional[sd.OutputStream] = None
        self._rolling_buf:    Optional[np.ndarray] = None   # float32 atual
        self._rolling_pos:    int  = 0      # posição no buffer do loop
        self._rolling_lock:   threading.Lock = threading.Lock()

    # ── Carregamento ──────────────────────────────────────────────────────────

    def _load_assets(self, audio_dir: Path) -> None:
        names = [
            "hit_floor_soft",
            "hit_floor_hard",
            "hit_wall",
            "hit_dice",
            "rolling_loop",
        ]
        for name in names:
            path = audio_dir / f"{name}.wav"
            buf  = WavBuffer.load(path)
            if buf is None:
                print(f"[audio] aviso: '{name}.wav' não encontrado em {audio_dir}")
            self._samples[name] = buf

    # ── API principal ─────────────────────────────────────────────────────────

    def on_collision(self, event: CollisionEvent) -> None:
        """
        Processa um CollisionEvent vindo do PhysicsWorld.

        Chamado para cada evento de poll_collision_events() no tick de física.
        Disparos rápidos (<= cooldown) do mesmo par de corpos são descartados.
        """
        if not self.enabled:
            return
        if event.impulse < IMPACT_THRESHOLD:
            return

        # Limita sons simultâneos
        self._active_threads = [t for t in self._active_threads if t.is_alive()]
        if len(self._active_threads) >= MAX_CONCURRENT_SOUNDS:
            return

        # Cooldown por par de corpos
        pair = (min(event.body_a, event.body_b),
                max(event.body_a, event.body_b))
        if self._cooldown.get(pair, 0) > 0:
            return
        self._cooldown[pair] = self._cooldown_ticks

        # Seleciona sample e calcula parâmetros
        buf = self._samples.get(self._sample_for(event))
        if buf is None:
            return

        volume = float(np.clip(event.impulse / IMPACT_REFERENCE, 0.1, 1.0))
        volume *= self.master_volume
        pitch  = random.uniform(PITCH_MIN, PITCH_MAX)

        # Playback em thread separada — não bloqueia o loop de física/render
        t = threading.Thread(
            target=self._play_oneshot,
            args=(buf, volume, pitch),
            daemon=True,
        )
        t.start()
        self._active_threads.append(t)

    def on_rolling(self, states: list) -> None:
        """
        Atualiza o loop contínuo de rolling.

        Inicia o stream se dados estão em movimento, para se pararam.
        O volume é ajustado proporcionalmente à velocidade angular média.
        """
        if not self.enabled:
            return

        buf = self._samples.get("rolling_loop")
        if buf is None:
            return

        # Velocidade angular média dos dados ainda em movimento
        speeds = [
            float(np.linalg.norm(s.angular_velocity))
            for s in states if not s.is_resting
        ]

        if not speeds or max(speeds) < ROLLING_SPEED_THRESHOLD:
            self._stop_rolling()
            return

        avg_speed = sum(speeds) / len(speeds)
        vol = float(np.clip(avg_speed / ROLLING_SPEED_REFERENCE,
                            ROLLING_VOL_MIN, 1.0)) * self.master_volume

        self._ensure_rolling_stream(buf, vol)

    def on_roll_complete(self) -> None:
        """Para o loop de rolling quando todos os dados param."""
        self._stop_rolling()

    def tick(self) -> None:
        """
        Avança contadores de cooldown. Chamar uma vez por tick de simulação.
        """
        expired = [p for p, n in self._cooldown.items() if n <= 1]
        for p in expired:
            del self._cooldown[p]
        for p in list(self._cooldown):
            self._cooldown[p] -= 1

    def stop_all(self) -> None:
        """Para todos os sons imediatamente."""
        self._stop_rolling()
        # Sons de impacto são não-bloqueantes e curtos; deixamos terminar
        # naturalmente, mas podemos interromper o dispositivo padrão se preciso.
        try:
            if _SD_AVAILABLE:
                sd.stop()
        except Exception:
            pass
        self._active_threads.clear()

    # ── Internos — sons de impacto ────────────────────────────────────────────

    def _sample_for(self, event: CollisionEvent) -> str:
        if event.surface == Surface.DICE:
            return "hit_dice"
        if event.surface == Surface.WALL:
            return "hit_wall"
        return "hit_floor_soft" if event.impulse < 2.0 else "hit_floor_hard"

    def _play_oneshot(self, buf: WavBuffer, volume: float, pitch: float) -> None:
        """Executa em thread daemon — aplica pitch/volume e toca até o fim."""
        try:
            processed = buf.with_pitch(pitch).with_volume(volume)
            arr       = processed.as_output_array()
            sd.play(arr, samplerate=processed.sample_rate, blocking=True)
        except Exception as e:
            print(f"[audio] erro no playback: {e}")

    # ── Internos — loop de rolling ────────────────────────────────────────────

    def _ensure_rolling_stream(self, buf: WavBuffer, volume: float) -> None:
        """
        Garante que o stream de rolling está ativo com o volume correto.

        Usa um OutputStream com callback para fazer loop infinito sem gaps.
        O volume é ajustado atomicamente via _rolling_buf sem reiniciar o stream.
        """
        # Atualiza buffer com novo volume (troca atômica lida pelo callback)
        new_arr = buf.with_volume(volume).as_output_array()
        with self._rolling_lock:
            self._rolling_buf = new_arr

        if self._rolling_stream is not None and self._rolling_stream.active:
            return  # stream já rodando; o callback usa o buf atualizado

        # Cria novo stream com callback de loop
        n_ch = buf.n_channels

        def _callback(outdata: np.ndarray, frames: int, time_info, status) -> None:
            with self._rolling_lock:
                src = self._rolling_buf
            if src is None:
                outdata[:] = 0
                return

            n_src  = len(src)
            pos    = self._rolling_pos
            needed = frames

            # Preenche outdata fazendo loop no buffer de rolling
            out_flat = outdata.ravel() if n_ch == 1 else outdata
            written  = 0
            while needed > 0:
                available = n_src - pos
                chunk     = min(available, needed)
                if n_ch == 1:
                    out_flat[written:written + chunk] = src[pos:pos + chunk]
                else:
                    outdata[written:written + chunk] = src[pos:pos + chunk]
                pos     = (pos + chunk) % n_src
                written += chunk
                needed  -= chunk

            self._rolling_pos = pos

        try:
            self._rolling_stream = sd.OutputStream(
                samplerate=buf.sample_rate,
                channels=n_ch,
                dtype="float32",
                callback=_callback,
            )
            self._rolling_stream.start()
        except Exception as e:
            print(f"[audio] erro ao iniciar stream de rolling: {e}")
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
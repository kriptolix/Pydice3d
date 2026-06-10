"""
audio_debug.py — Diagnóstico isolado do sistema de áudio (sounddevice)

Execute na raiz do projeto:
    python audio_debug.py
"""

import sys
import time
from pathlib import Path

print("=== 1. sounddevice ===")
try:
    import sounddevice as sd
    print(f"  OK: sounddevice {sd.__version__} importado")
    devs = sd.query_devices()
    default_out = sd.query_devices(kind="output")
    print(f"  Dispositivo de saída padrão: {default_out['name']}")
    print(f"  Sample rate padrão: {default_out['default_samplerate']} Hz")
except ImportError as e:
    print(f"  ERRO: {e}")
    print("  → pip install sounddevice")
    sys.exit(1)
except Exception as e:
    print(f"  AVISO: {e}")

print("\n=== 2. numpy ===")
try:
    import numpy as np
    print(f"  OK: numpy {np.__version__}")
except ImportError as e:
    print(f"  ERRO: {e}")
    sys.exit(1)

print("\n=== 3. Arquivos WAV ===")
candidates = [
    Path(__file__).parent / "src" / "pydice3d" / "assets" / "sounds",
    Path(__file__).parent / "pydice3d" / "assets" / "sounds",
    Path(__file__).parent / "assets" / "sounds",
]

audio_dir = None
for c in candidates:
    if c.exists():
        audio_dir = c
        print(f"  OK: {audio_dir}")
        break

if audio_dir is None:
    print("  ERRO: diretório de audio não encontrado. Tentados:")
    for c in candidates:
        print(f"    {c}")
    sys.exit(1)

import wave
expected = ["hit_floor_soft.wav", "hit_floor_hard.wav",
            "hit_wall.wav", "hit_dice.wav", "rolling_loop.wav"]
wavs_found = []
for name in expected:
    path = audio_dir / name
    if path.exists():
        with wave.open(str(path), "rb") as wf:
            info = f"{wf.getnchannels()}ch {wf.getsampwidth()*8}bit {wf.getframerate()}Hz"
        print(f"  OK  {name}  ({info})")
        wavs_found.append(path)
    else:
        print(f"  AUSENTE  {name}")

if not wavs_found:
    print("\n  Nenhum WAV encontrado.")
    sys.exit(1)

print("\n=== 4. Playback direto (sem engine) ===")
test_path = wavs_found[0]
try:
    with wave.open(str(test_path), "rb") as wf:
        sw  = wf.getsampwidth()
        sr  = wf.getframerate()
        ch  = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    pcm  = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch == 2:
        pcm = pcm.reshape(-1, 2)
    print(f"  Reproduzindo {test_path.name} diretamente...")
    sd.play(pcm, samplerate=sr, blocking=True)
    print("  OK — ouviu o som?")
except Exception as e:
    print(f"  ERRO: {e}")
    print("  → Verifique dispositivo de áudio / permissões do PipeWire")
    sys.exit(1)

print("\n=== 5. WavBuffer — pitch e volume ===")
sys.path.insert(0, str(Path(__file__).parent / "src"))
try:
    from pydice3d.audio import WavBuffer
    buf = WavBuffer.load(test_path)
    if buf is None:
        print("  ERRO: WavBuffer.load() retornou None")
    else:
        print(f"  OK: {len(buf.data)} amostras, {buf.sample_rate} Hz, {buf.n_channels}ch")

        print("  Pitch normal (1.0), volume 0.5 ...")
        b = buf.with_volume(0.5)
        sd.play(b.as_output_array(), samplerate=b.sample_rate, blocking=True)
        time.sleep(0.1)

        print("  Pitch agudo (1.3), volume 1.0 ...")
        b = buf.with_pitch(1.3).with_volume(1.0)
        sd.play(b.as_output_array(), samplerate=b.sample_rate, blocking=True)
        time.sleep(0.1)

        print("  Pitch grave (0.7), volume 1.0 ...")
        b = buf.with_pitch(0.7).with_volume(1.0)
        sd.play(b.as_output_array(), samplerate=b.sample_rate, blocking=True)
        print("  OK")
except Exception as e:
    print(f"  ERRO: {e}")
    import traceback; traceback.print_exc()

print("\n=== 6. DiceAudioEngine — CollisionEvents ===")
try:
    from pydice3d.audio import DiceAudioEngine, CollisionEvent, Surface

    engine = DiceAudioEngine(audio_dir=audio_dir)
    print(f"  enabled={engine.enabled}  master_volume={engine.master_volume}")
    print("  Samples carregados:")
    for name, buf in engine._samples.items():
        print(f"    {name}: {'OK' if buf else 'AUSENTE'}")

    print("\n  FLOOR hard (impulse=6.0) ...")
    engine.on_collision(CollisionEvent(1, 0, Surface.FLOOR, 6.0))
    time.sleep(1.0)

    print("  FLOOR soft (impulse=0.8) ...")
    engine.on_collision(CollisionEvent(2, 0, Surface.FLOOR, 0.8))
    time.sleep(1.0)

    print("  WALL (impulse=3.0) ...")
    engine.on_collision(CollisionEvent(1, 3, Surface.WALL, 3.0))
    time.sleep(1.0)

    print("  DICE (impulse=2.0) ...")
    engine.on_collision(CollisionEvent(1, 2, Surface.DICE, 2.0))
    time.sleep(1.0)
    print("  OK")

except Exception as e:
    print(f"  ERRO: {e}")
    import traceback; traceback.print_exc()

print("\n=== 7. Loop de rolling ===")
try:
    from pydice3d.audio import DiceAudioEngine

    class FakeState:
        is_resting = False
        angular_velocity = np.array([8.0, 5.0, 3.0])

    engine2 = DiceAudioEngine(audio_dir=audio_dir)
    print("  Iniciando loop por 2s ...")
    for _ in range(60):
        engine2.on_rolling([FakeState()])
        time.sleep(1/30)
    print("  Parando loop ...")
    engine2.on_roll_complete()
    time.sleep(0.3)
    print("  OK")
except Exception as e:
    print(f"  ERRO: {e}")
    import traceback; traceback.print_exc()

print("\n=== 8. PhysicsWorld.poll_collision_events ===")
try:
    from pydice3d.physics import PhysicsWorld
    pw = PhysicsWorld()
    if hasattr(pw, "poll_collision_events"):
        evts = pw.poll_collision_events()
        print(f"  OK: retornou {len(evts)} eventos (sem dados, esperado 0)")
    else:
        print("  ERRO: physics.py atualizado não foi aplicado")
except Exception as e:
    print(f"  ERRO: {e}")

print("\n=== Diagnóstico concluído ===")
import numpy as np
import sounddevice as sd

# ============================================================
# CONFIGURAÇÃO
# ============================================================

import sys

MATERIAL_A = sys.argv[1] if len(sys.argv) > 1 else "dice"
MATERIAL_B = sys.argv[2] if len(sys.argv) > 2 else "dice"
VELOCITY  = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0

DICE_MASS = 0.020       # 20g

SAMPLE_RATE = 44100
DURATION = 0.04         # 40 ms

# ============================================================
# PRESETS
# ============================================================

PRESETS = {
    ("dice", "dice"): {
        "freqs": [
        3592,
        3602,
        3613
    ],
    "decay": 0.018,
    "noise": 0.91,
    "tone": 0.09
    },

    ("dice", "hard"): {
        "freqs": [250, 500, 900],
        "decay": 0.018,
        "noise": 0.85,
        "tone": 0.15,
    },

    ("dice", "soft"): {
        "freqs": [
        1972,
        1996,
        2089
    ],
    "decay": 0.010,
    "noise": 0.72,
    "tone": 0.28
    },
}


# ============================================================
# GERADOR
# ============================================================

def generate_impact_sound(
    velocity,
    material_a,
    material_b,
    duration=DURATION,
):
    key = tuple(sorted((material_a, material_b)))

    cfg = PRESETS[key]

    n = int(duration * SAMPLE_RATE)

    t = np.arange(n) / SAMPLE_RATE

    # energia aproximada
    energy = 0.5 * DICE_MASS * velocity**2

    amplitude = min(
        1.0,
        0.15 + energy * 25.0
    )

    envelope = np.exp(
        -t / cfg["decay"]
    )

    noise = np.random.randn(n)
    noise *= envelope

    resonances = np.zeros(n)

    for freq in cfg["freqs"]:

        detune = np.random.uniform(
            -0.08,
            0.08
        )

        f = freq * (1.0 + detune)

        resonances += np.sin(
            2 * np.pi * f * t
        )

    resonances *= envelope

    sound = (
        cfg["noise"] * noise +
        cfg["tone"] * resonances
    )

    sound *= amplitude

    peak = np.max(np.abs(sound))

    if peak > 0:
        sound /= peak

    # evita clipping
    sound *= 0.8

    return sound.astype(np.float32)


# ============================================================
# LOOP
# ============================================================

def make_loop(sound, gap_ms=150):
    gap_samples = int(
        SAMPLE_RATE *
        gap_ms /
        1000
    )

    silence = np.zeros(
        gap_samples,
        dtype=np.float32
    )

    return np.concatenate([
        sound,
        silence
    ])


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    audio = generate_impact_sound(
        velocity=VELOCITY,
        material_a=MATERIAL_A,
        material_b=MATERIAL_B
    )

    loop_audio = make_loop(
        audio,
        gap_ms=150
    )

    print()
    print("===================================")
    print("VELOCITY :", VELOCITY)
    print("MATERIAL :", MATERIAL_A, "x", MATERIAL_B)
    print("Pressione CTRL+C para parar.")
    print("===================================")
    print()

    try:
        sd.play(
            loop_audio,
            SAMPLE_RATE,
            loop=True
        )

        while True:
            sd.sleep(1000)

    except KeyboardInterrupt:
        sd.stop()

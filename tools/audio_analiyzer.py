import sys
import json

import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert


def analyze_wav(filename):

    sample_rate, audio = wavfile.read(filename)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float64)

    audio /= np.max(np.abs(audio))

    duration = len(audio) / sample_rate

    # =====================================================
    # FREQUÊNCIAS DOMINANTES
    # =====================================================

    window = np.hanning(len(audio))

    spectrum = np.abs(
        np.fft.rfft(audio * window)
    )

    freqs = np.fft.rfftfreq(
        len(audio),
        d=1.0 / sample_rate
    )

    spectrum[0] = 0

    peak_indices = np.argpartition(
        spectrum,
        -3
    )[-3:]

    peak_freqs = freqs[peak_indices]

    peak_freqs = sorted(
        peak_freqs.astype(int)
    )

    # =====================================================
    # DECAY
    # =====================================================

    envelope = np.abs(
        hilbert(audio)
    )

    peak = envelope.max()

    threshold = peak * 0.1

    below = np.where(
        envelope < threshold
    )[0]

    if len(below):
        decay = below[0] / sample_rate
    else:
        decay = duration

    # =====================================================
    # NOISE / TONE
    # =====================================================

    total_energy = np.sum(spectrum)

    peak_energy = np.sum(
        spectrum[peak_indices]
    )

    tone = peak_energy / total_energy

    tone = np.clip(
        tone * 5.0,
        0.0,
        1.0
    )

    noise = 1.0 - tone

    result = {
    "freqs": [int(x) for x in peak_freqs],
    "decay": float(round(decay, 3)),
    "noise": float(round(noise, 2)),
    "tone": float(round(tone, 2)),
    }

    return result


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "uso: python analyze.py arquivo.wav"
        )
        sys.exit(1)

    result = analyze_wav(
        sys.argv[1]
    )

    print(
        json.dumps(
            result,
            indent=4
        )
    )

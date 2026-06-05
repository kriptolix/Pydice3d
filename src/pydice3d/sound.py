import simpleaudio as sa

class SoundBank:
    def __init__(self, files):
        self.sounds = {
            name: sa.WaveObject.from_wave_file(path)
            for name, path in files.items()
        }

    def play(self, name):
        self.sounds[name].play()
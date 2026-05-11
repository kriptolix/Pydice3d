# sound.py
#
# Requer:
#   pip install pygobject
#
# Linux:
#   sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-gstreamer-1.0 \
#                    gstreamer1.0-plugins-base \
#                    gstreamer1.0-plugins-good


import gi

gi.require_version("Gst", "1.0")

from gi.repository import Gst


class SoundPlayer:
    def __init__(self, channels=16):
        Gst.init(None)

        self.players = []

        for _ in range(channels):
            player = Gst.ElementFactory.make("playbin", None)
            self.players.append(player)

        self.index = 0

    def play(self, filepath, volume=1.0):
        player = self.players[self.index]

        self.index = (self.index + 1) % len(self.players)

        uri = Gst.filename_to_uri(filepath)

        player.set_state(Gst.State.NULL)
        player.set_property("uri", uri)
        player.set_property("volume", volume)
        player.set_state(Gst.State.PLAYING)

    def stop_all(self):
        for player in self.players:
            player.set_state(Gst.State.NULL)

'''## Uso
sound = SoundPlayer(channels=32)

sound.play("sounds/roll.wav")
sound.play("sounds/collision.wav")
sound.play("sounds/collision.wav")'''
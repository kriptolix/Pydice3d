"""
sound.py — Sistema de som para rolagem de dados via GStreamer.

Uso
---
    from sound import SoundManager

    snd = SoundManager()
    snd.play_roll()      # toca som de arremesso (início da rolagem)
    snd.play_hit()       # toca som de impacto (dado batendo na bandeja)
    snd.play_settle()    # toca som de repouso (dados pararam)

    # Para silenciar tudo:
    snd.muted = True

Arquivos esperados (OGG Vorbis recomendado — nativo no GStreamer):
    assets/sounds/roll.ogg    — arremesso / chacoalhar
    assets/sounds/hit.ogg     — impacto
    assets/sounds/settle.ogg  — dados parando

Se um arquivo não existir, o som correspondente é silenciosamente ignorado.
OGG pode ser substituído por WAV, FLAC ou MP3 — basta mudar os caminhos.

Dependências
------------
    gi (já presente no ambiente GTK4)
    GStreamer + gst-plugins-base:
        sudo dnf install gstreamer1-plugins-base   # Fedora/RHEL
        sudo apt install gstreamer1.0-plugins-base # Debian/Ubuntu
"""

import os

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

Gst.init(None)


# ---------------------------------------------------------------------------
# Caminhos padrão — ajuste conforme sua estrutura de pastas
# ---------------------------------------------------------------------------
DEFAULT_SOUNDS = {
    "roll":   "assets/sounds/roll.ogg",
    "hit":    "assets/sounds/hit.ogg",
    "settle": "assets/sounds/settle.ogg",
}


class SoundManager:
    """
    Gerenciador de sons baseado em GStreamer (playbin).

    Cada evento usa um pipeline independente para permitir
    sobreposição de sons (ex: múltiplos impactos simultâneos).
    """

    def __init__(self, sounds: dict[str, str] | None = None):
        """
        sounds : dict opcional sobrescrevendo DEFAULT_SOUNDS.
                 Chaves: "roll", "hit", "settle".
                 Valores: caminho para o arquivo de áudio.
        """
        self.muted = False
        self._sounds = {**DEFAULT_SOUNDS, **(sounds or {})}

        # Converte para URI file:// absoluta e valida existência
        self._uris: dict[str, str | None] = {}
        for key, path in self._sounds.items():
            abs_path = os.path.abspath(path)
            if os.path.isfile(abs_path):
                self._uris[key] = f"file://{abs_path}"
                print(f"[sound] {key}: {abs_path}")
            else:
                self._uris[key] = None
                print(f"[sound] {key}: arquivo não encontrado — {abs_path}")

        # Pool de pipelines reutilizáveis para "hit" (pode ocorrer várias vezes
        # em rápida sucessão). Tamanho 4 cobre 4 dados simultâneos.
        self._hit_pool: list = [self._make_pipeline() for _ in range(4)]
        self._hit_pool_idx = 0

    # ------------------------------------------------------------------
    def _make_pipeline(self):
        """Cria um pipeline playbin pronto para uso."""
        pipeline = Gst.ElementFactory.make("playbin", None)
        if pipeline is None:
            print("[sound] AVISO: GStreamer 'playbin' não disponível. "
                  "Instale gstreamer1-plugins-base.")
        return pipeline

    def _play_uri(self, uri: str | None, pipeline=None):
        """Dispara a reprodução de uma URI em um pipeline."""
        if self.muted or uri is None:
            return
        if pipeline is None:
            pipeline = self._make_pipeline()
        if pipeline is None:
            return

        # Para e reseta antes de reusar
        pipeline.set_state(Gst.State.NULL)
        pipeline.set_property("uri", uri)
        pipeline.set_state(Gst.State.PLAYING)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def play_roll(self):
        """
        Som de arremesso — tocado uma vez no início de cada rolagem.
        Representa o chacoalhar / lançamento dos dados.
        """
        self._play_uri(self._uris.get("roll"))

    def play_hit(self):
        """
        Som de impacto — pode ser chamado múltiplas vezes por rolagem
        (uma por dado, ou sincronizado com eventos de colisão no futuro).
        Usa pool rotativo para suportar sobreposição.
        """
        pipeline = self._hit_pool[self._hit_pool_idx % len(self._hit_pool)]
        self._hit_pool_idx += 1
        self._play_uri(self._uris.get("hit"), pipeline)

    def play_settle(self):
        """
        Som de repouso — tocado uma vez quando todos os dados pararam.
        """
        self._play_uri(self._uris.get("settle"))

    def stop_all(self):
        """Para todos os sons em reprodução."""
        for p in self._hit_pool:
            if p:
                p.set_state(Gst.State.NULL)

    def __del__(self):
        self.stop_all()
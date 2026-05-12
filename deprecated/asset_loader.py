"""
asset_loader.py — Descoberta e carregamento de assets dos dados.

Responsabilidades:
  - Mapeamento tipo → caminhos de OBJ e texturas
  - Validação de existência de arquivos
  - Cálculo de escala visual a partir do bounding box do OBJ
"""

import os
import numpy as np


# Diretório raiz do projeto (dois níveis acima deste arquivo:
#   rendering/asset_loader.py  →  rendering/  →  projeto/)
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__))
)


def _abs(rel_path: str) -> str:
    """Converte caminho relativo à raiz do projeto em caminho absoluto."""
    return os.path.join(_PROJECT_ROOT, rel_path)


# ---------------------------------------------------------------------------
# Definição dos assets por tipo de dado
# ---------------------------------------------------------------------------

_DICE_ASSETS: dict[str, dict[str, str]] = {
    "d4":  {"obj": "assets/d4/d4.obj",   "tex": "assets/d4"},
    "d6":  {"obj": "assets/d6/d6.obj",   "tex": "assets/d6"},
    "d8":  {"obj": "assets/d8/d8.obj",   "tex": "assets/d8"},
    "d10": {"obj": "assets/d10/d10.obj", "tex": "assets/d10"},
    "d12": {"obj": "assets/d12/d12.obj", "tex": "assets/d12"},
    "d20": {"obj": "assets/d20/d20.obj", "tex": "assets/d20"},
}

TEX_BASE_FILENAME   = "DefaultMaterial_Base_color.png"
TEX_NORMAL_FILENAME = "DefaultMaterial_Normal_DirectX.png"

FLOOR_TEX_BASE   = "assets/floor/wood_base_color.png"
FLOOR_TEX_NORMAL = "assets/floor/wood_normal_directx.png"


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def get_obj_path(dice_type: str) -> str | None:
    """Retorna o caminho absoluto do OBJ para o tipo, ou None se não existir."""
    entry = _DICE_ASSETS.get(dice_type)
    if entry is None:
        return None
    path = _abs(entry["obj"])
    if not os.path.isfile(path):
        print(f"[asset_loader] OBJ não encontrado: {path}")
        return None
    return path


def get_texture_paths(dice_type: str) -> dict[str, str | None]:
    """
    Retorna {"base": <path|None>, "normal": <path|None>} para o tipo.

    Resolve os caminhos relativamente à raiz do projeto, eliminando
    a dependência do cwd que causava o bug de textura no glarena original.
    """
    entry = _DICE_ASSETS.get(dice_type)
    if entry is None:
        return {"base": None, "normal": None}

    tex_dir = _abs(entry["tex"])

    def _find(filename):
        p = os.path.join(tex_dir, filename)
        if os.path.isfile(p):
            return p
        print(f"[asset_loader] Textura não encontrada: {p}")
        return None

    return {
        "base":   _find(TEX_BASE_FILENAME),
        "normal": _find(TEX_NORMAL_FILENAME),
    }


def get_floor_texture_paths() -> dict[str, str | None]:
    """Retorna {"base": <path|None>, "normal": <path|None>} para o piso."""
    def _find(rel):
        p = _abs(rel)
        if os.path.isfile(p):
            return p
        print(f"[asset_loader] Textura de piso não encontrada: {p}")
        return None

    return {
        "base":   _find(FLOOR_TEX_BASE),
        "normal": _find(FLOOR_TEX_NORMAL),
    }

VISUAL_SIZE = 0.8 / 39.0

# Multiplicador fino por tipo — ajuste conforme necessário
_SIZE_MULTIPLIER = {
    "d4":  1.5,
    "d6":  1.0,
    "d8":  1.5,
    "d10": 1.5,
    "d12": 1.0,   # reduza até ficar do tamanho certo
    "d20": 1.0,
}

def compute_dice_scale(positions_array: np.ndarray,
                       target_size: float,
                       dice_type: str = "") -> float:    
    """
    Calcula o fator de escala para que o dado ocupe target_size metros.

    positions_array: array de posições dos vértices do OBJ (N×3 ou flat).
    """
    multiplier = _SIZE_MULTIPLIER.get(dice_type, 1.0)
    return (target_size / VISUAL_SIZE) * multiplier
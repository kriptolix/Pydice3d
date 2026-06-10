"""
atlas_debug.py — Diagnóstico visual do atlas MSDF

Execute na raiz do projeto:
    python atlas_debug.py

Gera atlas_debug.png com os rects dos dígitos 0-9, +, - sobrepostos
na imagem do atlas, para verificar visualmente se os UVs estão corretos.
Também imprime todos os codepoints presentes no JSON.
"""

import json, sys, pathlib
import numpy as np

# ── Localiza os arquivos ──────────────────────────────────────────────────────
candidates = [
    pathlib.Path("src/pydice3d/assets/atlas"),
    pathlib.Path("pydice3d/assets/atlas"),
    pathlib.Path("assets/atlas"),
]
atlas_dir = next((c for c in candidates if c.exists()), None)
if atlas_dir is None:
    print("ERRO: diretório de atlas não encontrado")
    sys.exit(1)

npy_path  = atlas_dir / "atlas.npy"
json_path = atlas_dir / "atlas.json"

img  = np.load(str(npy_path))          # (H, W, 4) uint8
data = json.load(open(json_path))

atlas_w = float(data["atlas"]["width"])
atlas_h = float(data["atlas"]["height"])
H, W    = img.shape[:2]

print(f"Atlas: {W}×{H} px  (json diz {atlas_w:.0f}×{atlas_h:.0f})")
print(f"yOrigin: {data['atlas'].get('yOrigin')}")

# ── Lista todos os codepoints presentes ──────────────────────────────────────
glyphs = data["glyphs"]
print(f"\nTotal de glifos no JSON: {len(glyphs)}")
print("Codepoints com atlasBounds:")
has_bounds = [(g["unicode"], chr(g["unicode"])) for g in glyphs if "atlasBounds" in g]
for cp, ch in sorted(has_bounds):
    print(f"  {cp:4d}  '{ch}'")

# ── Verifica dígitos e símbolos de interesse ──────────────────────────────────
targets = {48:'0', 49:'1', 50:'2', 51:'3', 52:'4',
           53:'5', 54:'6', 55:'7', 56:'8', 57:'9',
           43:'+', 45:'-'}
by_unicode = {g["unicode"]: g for g in glyphs if "atlasBounds" in g}

print("\nRects dos dígitos e símbolos (pixels absolutos, yOrigin=bottom):")
for cp, ch in sorted(targets.items()):
    if cp in by_unicode:
        b = by_unicode[cp]["atlasBounds"]
        u0 = b["left"]/atlas_w;  u1 = b["right"]/atlas_w
        v0 = b["bottom"]/atlas_h; v1 = b["top"]/atlas_h
        print(f"  '{ch}' (U+{cp:04X}): pixels=({b['left']:.1f},{b['bottom']:.1f},"
              f"{b['right']:.1f},{b['top']:.1f})  uv=({u0:.3f},{v0:.3f},{u1:.3f},{v1:.3f})")
    else:
        print(f"  '{ch}' (U+{cp:04X}): NÃO ENCONTRADO no atlas")

# ── Gera imagem de debug com rects sobrepostos ────────────────────────────────
try:
    from PIL import Image, ImageDraw, ImageFont

    # yOrigin=bottom: o numpy tem Y=0 no topo, atlas tem Y=0 na base
    # precisa flip vertical para exibir corretamente
    img_rgb = img[:, :, :3].copy()
    img_pil = Image.fromarray(np.flipud(img_rgb))
    draw    = ImageDraw.Draw(img_pil)

    colors = {
        'digit': (0, 255, 0),    # verde para dígitos
        'symbol': (255, 128, 0), # laranja para + e -
    }

    for cp, ch in sorted(targets.items()):
        if cp not in by_unicode:
            continue
        b     = by_unicode[cp]["atlasBounds"]
        color = colors['symbol'] if ch in ('+', '-') else colors['digit']

        # Converte pixels do atlas (yOrigin=bottom) para pixels da imagem (yOrigin=top)
        x0 = b["left"]
        x1 = b["right"]
        y0_img = H - b["top"]    # flip Y
        y1_img = H - b["bottom"]

        draw.rectangle([x0, y0_img, x1, y1_img], outline=color, width=2)
        draw.text((x0 + 2, y0_img + 2), ch, fill=color)

    out_path = "atlas_debug.png"
    img_pil.save(out_path)
    print(f"\nImagem de debug salva em: {out_path}")
    print("Verde = dígitos, Laranja = +/-")
    print("Verifique se os retângulos cobrem os glifos corretos.")

except ImportError:
    print("\n(Pillow não instalado — imagem de debug não gerada)")
    print("Para gerar: pip install Pillow")
import os
import numpy as np
from PIL import Image

def convert_png_to_npy(input_file):
    if not input_file.lower().endswith(".png"):
        raise ValueError("File must be PNG")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not founded: {input_file}")

    img = Image.open(input_file).convert("RGBA")
    arr = np.array(img, dtype=np.uint8)

    output_file = os.path.splitext(input_file)[0] + ".npy"

    np.save(output_file, arr)

    print(f"Salvo: {output_file} shape={arr.shape}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Use: python convert.py imagem.png")
        sys.exit(1)

    convert_png_to_npy(sys.argv[1])

"""
shaders.py — Shaders GLSL e utilitários de upload para a GPU.

Responsabilidades:
  - Código-fonte dos shaders PBR-lite e Phong simples
  - Compilação e linkagem de programas OpenGL
  - Upload de geometria (VAO/VBO)
  - Carregamento de texturas (Pillow → OpenGL)
"""

from OpenGL.GL import *
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Shader PBR-lite: Base Color + Normal map
#
# Atributos:  loc 0 aPos, loc 1 aNormal, loc 2 aUV, loc 3 aTangent
# Uniforms:   uTexBase, uTexNormal, uHasBase, uHasNormal,
#             uMVP, uModelView, uNormalMat, uColor, uAlpha, uLightPos
# ---------------------------------------------------------------------------

_VERT_PBR = """
#version 330 core

layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;
layout(location=3) in vec3 aTangent;

uniform mat4 uMVP;
uniform mat4 uModelView;
uniform mat3 uNormalMat;

out vec3 vFragPos;
out vec2 vUV;
out mat3 vTBN;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vFragPos    = vec3(uModelView * vec4(aPos, 1.0));
    vUV         = aUV;

    vec3 N = normalize(uNormalMat * aNormal);
    vec3 T = normalize(uNormalMat * aTangent);
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T);
    vTBN = mat3(T, B, N);
}
"""

_FRAG_PBR = """
#version 330 core

in vec3 vFragPos;
in vec2 vUV;
in mat3 vTBN;

uniform sampler2D uTexBase;
uniform sampler2D uTexNormal;
uniform int       uHasBase;
uniform int       uHasNormal;

uniform vec3  uColor;
uniform float uAlpha;
uniform vec3  uLightPos;

out vec4 FragColor;

void main() {
    vec3 baseColor = (uHasBase == 1)
        ? pow(texture(uTexBase, vUV).rgb, vec3(2.2))
        : uColor;

    vec3 N;
    if (uHasNormal == 1) {
        vec3 nMap = texture(uTexNormal, vUV).rgb * 2.0 - 1.0;
        nMap.y = -nMap.y;   // DirectX → OpenGL
        N = normalize(vTBN * nMap);
    } else {
        N = normalize(vTBN[2]);
    }

    vec3  L    = normalize(uLightPos - vFragPos);
    vec3  V    = normalize(-vFragPos);
    vec3  H    = normalize(L + V);
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 24.0) * 0.25;

    vec3 col = baseColor * (0.18 + diff * 0.80) + vec3(spec);
    col = pow(col, vec3(1.0 / 2.2));
    FragColor = vec4(col, uAlpha);
}
"""

# ---------------------------------------------------------------------------
# Shader Phong simples (piso sem UV)
# ---------------------------------------------------------------------------

_VERT_SIMPLE = """
#version 330 core

layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;

uniform mat4 uMVP;
uniform mat4 uModelView;
uniform mat3 uNormalMat;

out vec3 vNormal;
out vec3 vFragPos;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vFragPos    = vec3(uModelView * vec4(aPos, 1.0));
    vNormal     = normalize(uNormalMat * aNormal);
}
"""

_FRAG_SIMPLE = """
#version 330 core

in vec3 vNormal;
in vec3 vFragPos;

uniform vec3  uColor;
uniform float uAlpha;
uniform vec3  uLightPos;

out vec4 FragColor;

void main() {
    vec3  L   = normalize(uLightPos - vFragPos);
    vec3  H   = normalize(L + normalize(-vFragPos));
    float dif = max(dot(vNormal, L), 0.0);
    float spc = pow(max(dot(vNormal, H), 0.0), 48.0) * 0.3;
    vec3  col = uColor * (0.20 + dif * 0.75) + vec3(spc);
    FragColor = vec4(col, uAlpha);
}
"""


# ---------------------------------------------------------------------------
# Compilação e link
# ---------------------------------------------------------------------------

def _compile_shader(src: str, kind: int) -> int:
    sh = glCreateShader(kind)
    glShaderSource(sh, src)
    glCompileShader(sh)
    if not glGetShaderiv(sh, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(sh).decode())
    return sh


def _link_program(vs_src: str, fs_src: str) -> int:
    vs = _compile_shader(vs_src, GL_VERTEX_SHADER)
    fs = _compile_shader(fs_src, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(prog).decode())
    glDeleteShader(vs)
    glDeleteShader(fs)
    return prog


def make_program() -> int:
    """Programa PBR-lite (dados com textura)."""
    return _link_program(_VERT_PBR, _FRAG_PBR)


def make_simple_program() -> int:
    """Programa Phong simples (piso / objetos sem UV)."""
    return _link_program(_VERT_SIMPLE, _FRAG_SIMPLE)


# ---------------------------------------------------------------------------
# Upload de geometria
# ---------------------------------------------------------------------------

def upload_mesh(pos_flat, nor_flat, uv_flat=None, tan_flat=None) -> tuple[int, int]:
    """
    Cria VAO + VBOs e retorna (vao, vertex_count).

    Atributos registrados:
      0 — posições (obrigatório)
      1 — normais  (obrigatório)
      2 — UVs      (opcional)
      3 — tangentes(opcional)
    """
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    def _vbo(data, loc, size):
        buf = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, buf)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glVertexAttribPointer(loc, size, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(loc)
        return buf

    _vbo(pos_flat, 0, 3)
    _vbo(nor_flat, 1, 3)
    if uv_flat  is not None and len(uv_flat)  > 0:
        _vbo(uv_flat,  2, 2)
    if tan_flat is not None and len(tan_flat) > 0:
        _vbo(tan_flat, 3, 3)

    glBindVertexArray(0)
    return vao, len(pos_flat) // 3


# ---------------------------------------------------------------------------
# Carregamento de textura
# ---------------------------------------------------------------------------

def load_texture(path: str, srgb: bool = False) -> int:
    """
    Carrega PNG/JPG como textura OpenGL.

    srgb=True  → GL_SRGB8_ALPHA8  (Base Color)
    srgb=False → GL_RGBA8          (Normal, Roughness, etc.)
    """
    

    img  = Image.open(path).convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
    data = np.array(img, dtype=np.uint8)

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(
        GL_TEXTURE_2D, 0,
        GL_SRGB8_ALPHA8 if srgb else GL_RGBA8,
        img.width, img.height, 0,
        GL_RGBA, GL_UNSIGNED_BYTE, data
    )
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex

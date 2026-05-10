from OpenGL.GL import *
import numpy as np

# ---------------------------------------------------------------------------
# Shader PBR-lite: Base Color + Normal map
#
# Atributos de vértice:
#   loc 0 — aPos     vec3
#   loc 1 — aNormal  vec3
#   loc 2 — aUV      vec2
#   loc 3 — aTangent vec3
#
# Uniforms de textura:
#   uTexBase   sampler2D  — Base Color (sRGB)
#   uTexNormal sampler2D  — Normal map DirectX (Y invertido vs OpenGL)
#   uHasBase   int        — 1 se a textura de base está carregada
#   uHasNormal int        — 1 se o normal map está carregado
# ---------------------------------------------------------------------------

VERT_SRC = """
#version 330 core

layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;
layout(location=3) in vec3 aTangent;

uniform mat4 uMVP;
uniform mat4 uModelView;
uniform mat3 uNormalMat;

out vec3 vFragPos;   // em view-space
out vec2 vUV;
out mat3 vTBN;       // tangent-to-view matrix

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vFragPos    = vec3(uModelView * vec4(aPos, 1.0));
    vUV         = aUV;

    // Constrói TBN em view-space
    vec3 N = normalize(uNormalMat * aNormal);
    vec3 T = normalize(uNormalMat * aTangent);
    T = normalize(T - dot(T, N) * N);   // re-ortogonaliza
    vec3 B = cross(N, T);

    vTBN = mat3(T, B, N);
}
"""

FRAG_SRC = """
#version 330 core

in vec3 vFragPos;
in vec2 vUV;
in mat3 vTBN;

uniform sampler2D uTexBase;
uniform sampler2D uTexNormal;
uniform int       uHasBase;
uniform int       uHasNormal;

uniform vec3  uColor;      // cor fallback quando não há textura
uniform float uAlpha;
uniform vec3  uLightPos;   // em view-space

out vec4 FragColor;

void main() {
    // ---- Cor base ----
    vec3 baseColor = (uHasBase == 1)
        ? pow(texture(uTexBase, vUV).rgb, vec3(2.2))   // sRGB → linear
        : uColor;

    // ---- Normal ----
    vec3 N;
    if (uHasNormal == 1) {
        vec3 nMap = texture(uTexNormal, vUV).rgb * 2.0 - 1.0;
        nMap.y = -nMap.y;   // DirectX → OpenGL: inverte canal Y
        N = normalize(vTBN * nMap);
    } else {
        N = normalize(vTBN[2]);   // coluna Z = normal geométrica
    }

    // ---- Iluminação Blinn-Phong ----
    vec3  L       = normalize(uLightPos - vFragPos);
    vec3  V       = normalize(-vFragPos);
    vec3  H       = normalize(L + V);

    float ambient = 0.18;
    float diff    = max(dot(N, L), 0.0);
    float spec    = pow(max(dot(N, H), 0.0), 64.0) * 0.4;

    vec3 col = baseColor * (ambient + diff * 0.80) + vec3(spec);

    // Correção gamma na saída
    col = pow(col, vec3(1.0 / 2.2));

    FragColor = vec4(col, uAlpha);
}
"""

# ---------------------------------------------------------------------------
# Shader simples para objetos sem textura (piso da bandeja)
# ---------------------------------------------------------------------------

VERT_SIMPLE = """
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

FRAG_SIMPLE = """
#version 330 core
in vec3 vNormal;
in vec3 vFragPos;

uniform vec3  uColor;
uniform float uAlpha;
uniform vec3  uLightPos;

out vec4 FragColor;

void main() {
    vec3 L    = normalize(uLightPos - vFragPos);
    vec3 H    = normalize(L + normalize(-vFragPos));
    float amb = 0.20;
    float dif = max(dot(vNormal, L), 0.0);
    float spc = pow(max(dot(vNormal, H), 0.0), 48.0) * 0.3;
    vec3  col = uColor * (amb + dif * 0.75) + vec3(spc);
    FragColor = vec4(col, uAlpha);
}
"""


# ---------------------------------------------------------------------------
# Helpers de compilação / link
# ---------------------------------------------------------------------------

def _compile(src, kind):
    sh = glCreateShader(kind)
    glShaderSource(sh, src)
    glCompileShader(sh)
    if not glGetShaderiv(sh, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(sh).decode())
    return sh


def _link(vs_src, fs_src):
    vs = _compile(vs_src, GL_VERTEX_SHADER)
    fs = _compile(fs_src, GL_FRAGMENT_SHADER)
    p = glCreateProgram()
    glAttachShader(p, vs)
    glAttachShader(p, fs)
    glLinkProgram(p)
    if not glGetProgramiv(p, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(p).decode())
    glDeleteShader(vs)
    glDeleteShader(fs)
    return p


def make_program():
    """Programa PBR-lite (dados com textura)."""
    return _link(VERT_SRC, FRAG_SRC)


def make_simple_program():
    """Programa Phong simples (piso, objetos sem UV)."""
    return _link(VERT_SIMPLE, FRAG_SIMPLE)


# ---------------------------------------------------------------------------
# Upload de geometria
# ---------------------------------------------------------------------------

def upload_mesh(pos_flat, nor_flat, uv_flat=None, tan_flat=None):
    """
    Cria VAO + VBOs e retorna (vao, vertex_count).

    Se uv_flat e tan_flat forem fornecidos, registra os atributos 2 e 3.
    Caso contrário, apenas 0 (pos) e 1 (nor) são registrados — compatível
    com o shader simples do piso.
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

    if uv_flat is not None and len(uv_flat) > 0:
        _vbo(uv_flat,  2, 2)
    if tan_flat is not None and len(tan_flat) > 0:
        _vbo(tan_flat, 3, 3)

    glBindVertexArray(0)
    return vao, len(pos_flat) // 3


# ---------------------------------------------------------------------------
# Carregamento de textura
# ---------------------------------------------------------------------------

def load_texture(path, srgb=False):
    """
    Carrega uma imagem PNG/JPG como textura OpenGL e retorna o texture ID.

    srgb=True  → usa GL_SRGB8_ALPHA8 (correto para Base Color)
    srgb=False → usa GL_RGBA8 (correto para Normal, Roughness, etc.)

    Requer Pillow: pip install Pillow
    """
    from PIL import Image
    import numpy as np

    img = Image.open(path).convert("RGBA")
    img = img.transpose(Image.FLIP_TOP_BOTTOM)   # OpenGL origem = canto inferior esq.
    data = np.array(img, dtype=np.uint8)

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)

    internal = GL_SRGB8_ALPHA8 if srgb else GL_RGBA8

    glTexImage2D(
        GL_TEXTURE_2D, 0, internal,
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
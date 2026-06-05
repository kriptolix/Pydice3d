"""
shaders.py – Código GLSL e Utilitários de Compilação

Responsabilidade: definir os shaders GLSL e compilá-los em objetos OpenGL.
Não cria janelas nem contextos — requer contexto OpenGL ativo.

Shaders implementados
──────────────────────
DICE_VERT / DICE_FRAG : shader principal para dados com Blinn-Phong + atlas de glifos
GROUND_VERT / GROUND_FRAG : shader para o plano do chão (grade simples)

Sistema de glifos — Texture Atlas
───────────────────────────────────
Cada face recebe:
  - UV  : coordenadas [-1,1]² centradas na face, passadas como atributo de vértice
  - u_face_glyphs[F] : índice do glifo (int) para cada face F

O fragment shader converte as coordenadas UV da face para coordenadas na
atlas usando u_glyph_uvs[glyph_id] = vec4(u0, v0, u1, v1) e amostra
u_glyph_atlas (sampler2D, unidade de textura 0).

Índices de glifos
─────────────────
  0–9   : dígitos simples 0–9
  10–20 : números de dois dígitos 10–20
  21–30 : dezenas do d100 (00,10,20,…,90)  →  21 + tens//10
  31    : símbolo "+"  (dado fudge positivo)
  32    : símbolo "−"  (dado fudge negativo)
  33    : face vazia   (dado fudge neutro)
  255   : sem glifo (face não numerada)

Uniforms do shader de dados
────────────────────────────
  mat4      u_model                   : matriz de modelo
  mat4      u_view_proj               : VP = P × V
  mat3      u_normal_mat              : matriz de normais
  vec3      u_light_dir               : direção da luz (mundo, normalizada)
  vec3      u_light_color             : cor da luz
  vec3      u_ambient                 : cor ambiente
  vec3      u_dice_color              : cor base do dado
  vec3      u_glyph_color             : cor do glifo
  float     u_shininess               : expoente especular
  bool      u_highlight               : destaca o dado (resultado pronto)
  vec3      u_cam_pos                 : posição da câmera (espaço do mundo)
  int       u_face_glyphs[MAX_FACES]  : índice do glifo por face
  sampler2D u_glyph_atlas             : textura da atlas (unidade 0)
  vec4      u_glyph_uvs[MAX_GLYPHS]  : (u0,v0,u1,v1) de cada glifo na atlas

Atributos de vértice
─────────────────────
  layout 0: vec3  a_position
  layout 1: vec3  a_normal
  layout 2: vec2  a_uv        — UV local da face em [-1,1]²
  layout 3: float a_face_idx  — índice da face (flat int via float)
"""

from __future__ import annotations
from OpenGL import GL


# ────────────────────────────────────────────────────────────────────────────
# Constantes de índice de glifo (usadas em Python e espelhadas no GLSL)
# ────────────────────────────────────────────────────────────────────────────

GLYPH_NONE   = 255   # sem glifo
GLYPH_PLUS   = 31    # "+"  dado fudge
GLYPH_MINUS  = 32    # "−"  dado fudge
GLYPH_BLANK  = 33    # face vazia dado fudge
MAX_FACES    = 24    # máximo de faces suportadas pelo array de uniforms
MAX_GLYPHS   = 34    # índices 0–33 (GLYPH_BLANK inclusive); 255=GLYPH_NONE fica fora

# d100 dezenas: 00→21, 10→22, ..., 90→30
def glyph_d100(tens: int) -> int:
    """tens ∈ {0,10,20,...,90} → índice de glifo"""
    return 21 + (tens // 10)


# ────────────────────────────────────────────────────────────────────────────
# Código GLSL
# ────────────────────────────────────────────────────────────────────────────

DICE_VERT = """
#version 330 core

layout(location = 0) in vec3  a_position;
layout(location = 1) in vec3  a_normal;
layout(location = 2) in vec2  a_uv;
layout(location = 3) in float a_face_idx;

uniform mat4 u_model;
uniform mat4 u_view_proj;
uniform mat3 u_normal_mat;

out vec3  v_normal_world;
out vec3  v_frag_pos;
out vec2  v_uv;
flat out int v_face_idx;

void main() {
    vec4 world_pos  = u_model * vec4(a_position, 1.0);
    v_frag_pos      = world_pos.xyz;
    v_normal_world  = normalize(u_normal_mat * a_normal);
    v_uv            = a_uv;
    v_face_idx      = int(a_face_idx + 0.5);
    gl_Position     = u_view_proj * world_pos;
}
"""

# ── Fragment shader — atlas de glifos ───────────────────────────────────────
# v_uv está em [-1,1]² centrado na face.
# u_glyph_uvs[id] = vec4(u0, v0, u1, v1) — retângulo do glifo na atlas.
#
# Glifos simples (0–9, +, −): um único sample da atlas centralizado.
# Glifos compostos (10–20, d100 21–30): dois samples lado a lado,
#   cada dígito ocupa metade da largura com um offset lateral de ±PAIR_OFFSET.
# GLYPH_BLANK (33): nenhum sample (face vazia do dado fudge).

DICE_FRAG = """
#version 330 core

in  vec3  v_normal_world;
in  vec3  v_frag_pos;
in  vec2  v_uv;
flat in int v_face_idx;

out vec4 frag_color;

uniform vec3      u_light_dir;
uniform vec3      u_light_color;
uniform vec3      u_dice_color;
uniform vec3      u_glyph_color;
uniform float     u_shininess;
uniform bool      u_highlight;       // só ativo em modo debug
uniform vec3      u_cam_pos;
uniform int       u_face_glyphs[24];
uniform sampler2D u_glyph_atlas;
uniform sampler2D u_glyph_normal;    // normal map da atlas (mesmas UVs)
uniform vec4      u_glyph_uvs[10];
uniform vec4      u_glyph_uv_plus;
uniform vec4      u_glyph_uv_minus;

// ── Amostragem da atlas ──────────────────────────────────────────────────────

#define SINGLE_SCALE 0.62
#define PAIR_Y       0.62
#define PAIR_X       0.40
#define PAIR_OFF     0.43

// Retorna vec2(atlas_u, atlas_v) para um dígito, ou vec2(-1) se fora do domínio.
vec2 digit_atlas_uv(vec2 uv, int digit, float offset_x, float x_scale, float y_scale) {
    vec2 local = vec2((uv.x - offset_x) / x_scale, uv.y / y_scale);
    if (abs(local.x) > 1.0 || abs(local.y) > 1.0) return vec2(-1.0);
    vec4  rect = u_glyph_uvs[digit];
    vec2  n    = local * 0.5 + 0.5;
    return vec2(mix(rect.x, rect.z, n.x), mix(rect.w, rect.y, n.y));
}

vec2 symbol_atlas_uv(vec2 uv, vec4 rect, float scale) {
    vec2 local = uv / scale;
    if (abs(local.x) > 1.0 || abs(local.y) > 1.0) return vec2(-1.0);
    vec2 n = local * 0.5 + 0.5;
    return vec2(mix(rect.x, rect.z, n.x), mix(rect.w, rect.y, n.y));
}

// Retorna (mask, atlas_uv) para o glifo. atlas_uv é (-1,-1) se sem glifo.
// Usamos duas saídas empacotadas em vec3: .x = mask, .yz = atlas_uv.
vec3 glyph_sample(vec2 uv, int glyph_id) {
    vec2 auv = vec2(-1.0);

    if (glyph_id >= 0 && glyph_id <= 9) {
        auv = digit_atlas_uv(uv, glyph_id, 0.0, SINGLE_SCALE, SINGLE_SCALE);
    } else if (glyph_id >= 10 && glyph_id <= 20) {
        int   tens  = glyph_id / 10;
        int   units = glyph_id - tens * 10;
        vec2  lv    = digit_atlas_uv(uv, tens,  -PAIR_OFF, PAIR_X, PAIR_Y);
        vec2  rv    = digit_atlas_uv(uv, units,  PAIR_OFF, PAIR_X, PAIR_Y);
        // Escolhe o lado com maior cobertura
        float lm = (lv.x >= 0.0) ? texture(u_glyph_atlas, lv).r : 0.0;
        float rm = (rv.x >= 0.0) ? texture(u_glyph_atlas, rv).r : 0.0;
        auv = (lm >= rm) ? lv : rv;
    } else if (glyph_id >= 21 && glyph_id <= 30) {
        int   td = glyph_id - 21;
        vec2  lv = digit_atlas_uv(uv, td, -PAIR_OFF, PAIR_X, PAIR_Y);
        vec2  rv = digit_atlas_uv(uv,  0,  PAIR_OFF, PAIR_X, PAIR_Y);
        float lm = (lv.x >= 0.0) ? texture(u_glyph_atlas, lv).r : 0.0;
        float rm = (rv.x >= 0.0) ? texture(u_glyph_atlas, rv).r : 0.0;
        auv = (lm >= rm) ? lv : rv;
    } else if (glyph_id == 31) {
        auv = symbol_atlas_uv(uv, u_glyph_uv_plus,  SINGLE_SCALE);
    } else if (glyph_id == 32) {
        auv = symbol_atlas_uv(uv, u_glyph_uv_minus, SINGLE_SCALE * 0.7);
    }
    // 33 = blank, 255 = none → auv permanece (-1,-1)

    float mask = (auv.x >= 0.0) ? texture(u_glyph_atlas, auv).r : 0.0;
    return vec3(mask, auv);
}

// ── Iluminação ───────────────────────────────────────────────────────────────

void main() {
    vec3 N = normalize(v_normal_world);
    vec3 L = normalize(u_light_dir);
    vec3 V = normalize(u_cam_pos - v_frag_pos);
    vec3 H = normalize(L + V);

    // ── Cor base (highlight só em debug) ──────────────────────────────────
    vec3 base_color = u_highlight
        ? mix(u_dice_color, vec3(1.0, 0.85, 0.1), 0.45)
        : u_dice_color;

    // ── Glifo: máscara + normal map ───────────────────────────────────────
    int glyph_id = (v_face_idx >= 0 && v_face_idx < 24)
                   ? u_face_glyphs[v_face_idx] : 255;

    float glyph_mask_val = 0.0;
    vec3  N_final        = N;

    if (glyph_id != 255) {
        float edge_mask = 1.0 - smoothstep(0.78, 0.82, length(v_uv));
        vec3  gs        = glyph_sample(v_uv, glyph_id);
        float raw       = gs.x * edge_mask;
        float fw        = fwidth(raw);
        glyph_mask_val  = smoothstep(0.45 - fw, 0.45 + fw, raw);

        vec2 auv = gs.yz;
        if (auv.x >= 0.0 && glyph_mask_val > 0.01) {
            vec3 nm_raw = texture(u_glyph_normal, auv).rgb * 2.0 - 1.0;
            vec3 dp1  = dFdx(v_frag_pos);
            vec3 dp2  = dFdy(v_frag_pos);
            vec2 duv1 = dFdx(v_uv);
            vec2 duv2 = dFdy(v_uv);
            vec3 T    = normalize( duv2.y * dp1 - duv1.y * dp2);
            vec3 B    = normalize(-duv2.x * dp1 + duv1.x * dp2);
            vec3 N_bump = normalize(mat3(T, B, N) * nm_raw);
            N_final = normalize(mix(N, N_bump, glyph_mask_val * 0.55));
        }
    }

    float NdotL = max(dot(N_final, L), 0.0);
    float NdotV = max(dot(N_final, V), 0.0);
    float NdotH = max(dot(N_final, H), 0.0);

    // ── Diffuse Lambert simples — preserva cor saturada ───────────────────
    // (smooth-step aqui lavava porque enchia as sombras com a cor base)
    float diff = NdotL;

    // ── Ambient hemisférico neutro — escala de cinza para não tingir ──────
    // Sky/ground em cinza neutro: não empurra cor, só define contraste.
    float hemi    = N_final.y * 0.5 + 0.5;
    float amb_val = mix(0.12, 0.28, hemi);
    vec3  ambient = vec3(amb_val);

    // ── Rim light fraco e neutro — só volume, sem tingimento ─────────────
    float rim   = pow(1.0 - NdotV, 4.0);
    float rim_v = rim * 0.10;   // intensidade reduzida

    // ── Especular plástico: separado da cor base ──────────────────────────
    // Plástico tem duas camadas:
    //   1. Camada difusa colorida (base_color)
    //   2. Camada especular dielétrica BRANCA (independente da cor)
    // Lóbulo largo  (glossy coat) : baixo expoente, baixa intensidade
    // Lóbulo estreito (hot spot)  : alto expoente, intensidade visível
    float spec_broad  = pow(NdotH, u_shininess * 0.25) * 0.12;
    float spec_sharp  = pow(NdotH, u_shininess * 2.5)  * 0.70;
    float spec_total  = (spec_broad + spec_sharp) * step(0.001, NdotL);

    // ── Composição ────────────────────────────────────────────────────────
    vec3 lit_color = (ambient + diff * u_light_color + rim_v) * base_color
                   + spec_total * u_light_color;   // especular branco separado

    // ── Glifo sobre o dado ────────────────────────────────────────────────
    if (glyph_mask_val > 0.0) {
        vec3 glyph_lit = (ambient + diff * u_light_color) * u_glyph_color
                       + spec_total * u_light_color * 0.4;  // glifo pega menos brilho
        lit_color = mix(lit_color, glyph_lit, glyph_mask_val);
    }

    // ── Gamma correction ──────────────────────────────────────────────────
    // Só aplicar se o framebuffer for linear (sem sRGB automático do GTK).
    // Versão conservadora: gamma 1.8 em vez de 2.2 — menos branqueamento.
    lit_color = pow(clamp(lit_color, 0.0, 1.0), vec3(1.0 / 1.8));

    frag_color = vec4(lit_color, 1.0);
}
"""

GROUND_VERT = """
#version 330 core

layout(location = 0) in vec3 a_position;

uniform mat4 u_view_proj;

out vec2 v_uv;

void main() {
    v_uv        = a_position.xz * 0.5;
    gl_Position = u_view_proj * vec4(a_position, 1.0);
}
"""

GROUND_FRAG = """
#version 330 core

in  vec2 v_uv;
out vec4 frag_color;

vec4 grid(vec2 uv, float spacing) {
    vec2 wrapped = abs(fract(uv / spacing) - 0.5);
    vec2 dv      = fwidth(uv / spacing);
    vec2 line    = smoothstep(vec2(0.0), dv * 1.5, wrapped);
    float val    = 1.0 - min(line.x, line.y);
    return vec4(0.35, 0.35, 0.35, val * 0.7);
}

void main() {
    vec4 g1 = grid(v_uv, 1.0);
    vec4 g2 = grid(v_uv, 5.0);
    vec4 base = vec4(0.12, 0.12, 0.14, 1.0);
    frag_color = mix(base, vec4(0.5, 0.5, 0.5, 1.0), g2.a * 0.5 + g1.a * 0.25);
}
"""

WIRE_VERT = """
#version 330 core
layout(location = 0) in vec3 a_position;
uniform mat4 u_mvp;
void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
}
"""

WIRE_FRAG = """
#version 330 core
out vec4 frag_color;
uniform vec3 u_color;
void main() {
    frag_color = vec4(u_color, 1.0);
}
"""


# ────────────────────────────────────────────────────────────────────────────
# Compilação
# ────────────────────────────────────────────────────────────────────────────

class ShaderError(RuntimeError):
    pass


def _compile_shader(source: str, shader_type: int) -> int:
    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        log = GL.glGetShaderInfoLog(shader).decode()
        kind = "vertex" if shader_type == GL.GL_VERTEX_SHADER else "fragment"
        GL.glDeleteShader(shader)
        raise ShaderError(f"Falha ao compilar shader {kind}:\n{log}")
    return shader


def _link_program(vert: int, frag: int) -> int:
    program = GL.glCreateProgram()
    GL.glAttachShader(program, vert)
    GL.glAttachShader(program, frag)
    GL.glLinkProgram(program)
    GL.glDeleteShader(vert)
    GL.glDeleteShader(frag)
    if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
        log = GL.glGetProgramInfoLog(program).decode()
        GL.glDeleteProgram(program)
        raise ShaderError(f"Falha ao linkar programa:\n{log}")
    return program


def build_program(vert_src: str, frag_src: str) -> int:
    vert = _compile_shader(vert_src, GL.GL_VERTEX_SHADER)
    frag = _compile_shader(frag_src, GL.GL_FRAGMENT_SHADER)
    return _link_program(vert, frag)


def build_dice_program() -> int:
    return build_program(DICE_VERT, DICE_FRAG)


def build_ground_program() -> int:
    return build_program(GROUND_VERT, GROUND_FRAG)


# ────────────────────────────────────────────────────────────────────────────
# Helpers de uniforms
# ────────────────────────────────────────────────────────────────────────────

def set_uniform_mat4(program: int, name: str, mat) -> None:
    import numpy as np
    loc = GL.glGetUniformLocation(program, name)
    if loc != -1:
        GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mat.T.astype(np.float32))


def set_uniform_mat3(program: int, name: str, mat) -> None:
    import numpy as np
    loc = GL.glGetUniformLocation(program, name)
    if loc != -1:
        GL.glUniformMatrix3fv(loc, 1, GL.GL_FALSE, mat.T.astype(np.float32))


def set_uniform_vec3(program: int, name: str, v) -> None:
    loc = GL.glGetUniformLocation(program, name)
    if loc != -1:
        GL.glUniform3f(loc, float(v[0]), float(v[1]), float(v[2]))


def set_uniform_float(program: int, name: str, val: float) -> None:
    loc = GL.glGetUniformLocation(program, name)
    if loc != -1:
        GL.glUniform1f(loc, float(val))


def set_uniform_bool(program: int, name: str, val: bool) -> None:
    loc = GL.glGetUniformLocation(program, name)
    if loc != -1:
        GL.glUniform1i(loc, int(val))


def set_uniform_int_array(program: int, name: str, values: list[int]) -> None:
    """Envia array de ints para uniform int[]."""
    import ctypes, numpy as np
    n = len(values)
    for i, v in enumerate(values):
        loc = GL.glGetUniformLocation(program, f"{name}[{i}]")
        if loc != -1:
            GL.glUniform1i(loc, int(v))


def set_uniform_int(program: int, name: str, val: int) -> None:
    loc = GL.glGetUniformLocation(program, name)
    if loc != -1:
        GL.glUniform1i(loc, int(val))


def set_uniform_vec4(program: int, name: str, v) -> None:
    loc = GL.glGetUniformLocation(program, name)
    if loc != -1:
        GL.glUniform4f(loc, float(v[0]), float(v[1]), float(v[2]), float(v[3]))


def set_uniform_vec4_array(program: int, name: str, data: "np.ndarray") -> None:
    """
    Envia array de vec4 para uniform vec4[].
    data: float32 (N, 4)
    """
    import numpy as np
    arr = data.astype(np.float32)
    for i, row in enumerate(arr):
        loc = GL.glGetUniformLocation(program, f"{name}[{i}]")
        if loc != -1:
            GL.glUniform4f(loc, float(row[0]), float(row[1]), float(row[2]), float(row[3]))


# ────────────────────────────────────────────────────────────────────────────
# Tabela de UV da atlas
# ────────────────────────────────────────────────────────────────────────────

def build_glyph_uv_table(atlas_json: dict) -> "np.ndarray":
    """
    Retorna float32 (10, 4) com os rects (u0,v0,u1,v1) dos dígitos 0–9.
    Índice i == dígito i.
    """
    import numpy as np
    table = np.zeros((10, 4), dtype=np.float32)
    glyphs = atlas_json.get("glyphs", {})
    for i in range(10):
        key = str(i)
        if key in glyphs:
            g = glyphs[key]
            table[i] = [g["u0"], g["v0"], g["u1"], g["v1"]]
    return table


def build_symbol_uvs(atlas_json: dict) -> "tuple[np.ndarray, np.ndarray]":
    """
    Retorna (plus_rect, minus_rect) como float32 arrays de shape (4,).
    Usados para os uniforms u_glyph_uv_plus e u_glyph_uv_minus.
    """
    import numpy as np
    glyphs = atlas_json.get("glyphs", {})
    def _rect(key):
        if key in glyphs:
            g = glyphs[key]
            return np.array([g["u0"], g["v0"], g["u1"], g["v1"]], dtype=np.float32)
        return np.zeros(4, dtype=np.float32)
    return _rect("+"), _rect("-")
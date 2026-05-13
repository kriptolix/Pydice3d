"""
shaders.py – Código GLSL e Utilitários de Compilação

Responsabilidade: definir os shaders GLSL e compilá-los em objetos OpenGL.
Não cria janelas nem contextos — requer contexto OpenGL ativo.

Shaders implementados
──────────────────────
DICE_VERT / DICE_FRAG : shader principal para dados com Blinn-Phong
GROUND_VERT / GROUND_FRAG : shader para o plano do chão (grade simples)

Iluminação — Blinn-Phong
─────────────────────────
    Componentes : ambiente + difuso + especular
    Luz         : direcional (sem posição, sem atenuação)
    Normal      : transformada pela matriz normal = transpose(inverse(M))
                  Simplificado: como os dados têm escala uniforme,
                  usamos diretamente a parte rotação de M (sem distorção).

Uniforms do shader de dados
────────────────────────────
    mat4  u_model      : matriz de modelo (posição + rotação do dado)
    mat4  u_view_proj  : VP = P × V (projeção × view)
    mat3  u_normal_mat : matriz de normais = transpose(inverse(mat3(u_model)))
    vec3  u_light_dir  : direção da luz (espaço do mundo, normalizada)
    vec3  u_light_color: cor da luz
    vec3  u_ambient    : cor ambiente
    vec3  u_dice_color : cor base do dado
    float u_shininess  : expoente especular
    bool  u_highlight  : destaca o dado (resultado pronto)
"""

from __future__ import annotations

from OpenGL import GL


# ────────────────────────────────────────────────────────────────────────────
# Código GLSL
# ────────────────────────────────────────────────────────────────────────────

DICE_VERT = """
#version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

uniform mat4 u_model;
uniform mat4 u_view_proj;
uniform mat3 u_normal_mat;

out vec3 v_normal_world;
out vec3 v_frag_pos;

void main() {
    vec4 world_pos  = u_model * vec4(a_position, 1.0);
    v_frag_pos      = world_pos.xyz;
    v_normal_world  = normalize(u_normal_mat * a_normal);
    gl_Position     = u_view_proj * world_pos;
}
"""

DICE_FRAG = """
#version 330 core

in  vec3 v_normal_world;
in  vec3 v_frag_pos;
out vec4 frag_color;

uniform vec3  u_light_dir;      // espaço do mundo, normalizado, aponta PARA a luz
uniform vec3  u_light_color;
uniform vec3  u_ambient;
uniform vec3  u_dice_color;
uniform float u_shininess;
uniform bool  u_highlight;
uniform vec3  u_cam_pos;        // posição da câmera (para especular)

void main() {
    vec3 N = normalize(v_normal_world);
    vec3 L = normalize(u_light_dir);
    vec3 V = normalize(u_cam_pos - v_frag_pos);
    vec3 H = normalize(L + V);  // half-vector (Blinn-Phong)

    // Difuso
    float diff = max(dot(N, L), 0.0);

    // Especular (Blinn-Phong)
    float spec = 0.0;
    if (diff > 0.0)
        spec = pow(max(dot(N, H), 0.0), u_shininess);

    vec3 base_color = u_highlight
        ? mix(u_dice_color, vec3(1.0, 0.85, 0.1), 0.5)   // dourado ao parar
        : u_dice_color;

    vec3 color = u_ambient * base_color
               + diff * u_light_color * base_color
               + spec * u_light_color * 0.5;

    frag_color = vec4(color, 1.0);
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

// Grade simples via fwidth anti-aliased
vec4 grid(vec2 uv, float spacing) {
    vec2 wrapped = abs(fract(uv / spacing) - 0.5);
    vec2 dv      = fwidth(uv / spacing);
    vec2 line    = smoothstep(vec2(0.0), dv * 1.5, wrapped);
    float val    = 1.0 - min(line.x, line.y);
    return vec4(0.35, 0.35, 0.35, val * 0.7);
}

void main() {
    vec4 g1 = grid(v_uv, 1.0);     // grade fina
    vec4 g2 = grid(v_uv, 5.0);     // grade grossa
    vec4 base = vec4(0.12, 0.12, 0.14, 1.0);
    frag_color = mix(base, vec4(0.5, 0.5, 0.5, 1.0), g2.a * 0.5 + g1.a * 0.25);
}
"""


# ────────────────────────────────────────────────────────────────────────────
# Compilação e linkagem
# ────────────────────────────────────────────────────────────────────────────

class ShaderError(RuntimeError):
    """Exceção lançada quando compilação ou linkagem de shader falha."""


def _compile_shader(source: str, shader_type: int) -> int:
    """
    Compila um shader GLSL.

    Parâmetros
    ----------
    source      : código GLSL como string
    shader_type : GL_VERTEX_SHADER ou GL_FRAGMENT_SHADER

    Retorna o shader object (int). Lança ShaderError se falhar.
    """
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
    """
    Linka um programa GLSL a partir de shaders já compilados.
    Deleta os shaders após a linkagem (boa prática).

    Retorna o program object (int). Lança ShaderError se falhar.
    """
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
    """
    Compila e linka um programa GLSL completo.

    Parâmetros
    ----------
    vert_src : código GLSL do vertex shader
    frag_src : código GLSL do fragment shader

    Retorna
    -------
    int — program object OpenGL
    """
    vert = _compile_shader(vert_src, GL.GL_VERTEX_SHADER)
    frag = _compile_shader(frag_src, GL.GL_FRAGMENT_SHADER)
    return _link_program(vert, frag)


def build_dice_program() -> int:
    """Compila e linka o programa shader para os dados."""
    return build_program(DICE_VERT, DICE_FRAG)


def build_ground_program() -> int:
    """Compila e linka o programa shader para o chão."""
    return build_program(GROUND_VERT, GROUND_FRAG)


# ────────────────────────────────────────────────────────────────────────────
# Helpers de uniforms
# ────────────────────────────────────────────────────────────────────────────

def set_uniform_mat4(program: int, name: str, mat: "np.ndarray") -> None:
    """Envia matriz 4×4 para uniform em column-major (padrão OpenGL/GLSL).
    NumPy usa row-major, então transpomos antes de enviar com GL_FALSE."""
    import numpy as np
    loc = GL.glGetUniformLocation(program, name)
    if loc != -1:
        GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mat.T.astype(np.float32))


def set_uniform_mat3(program: int, name: str, mat: "np.ndarray") -> None:
    """Envia matriz 3×3 para uniform em column-major."""
    import numpy as np
    loc = GL.glGetUniformLocation(program, name)
    if loc != -1:
        GL.glUniformMatrix3fv(loc, 1, GL.GL_FALSE, mat.T.astype(np.float32))


def set_uniform_vec3(program: int, name: str, v: "np.ndarray | tuple") -> None:
    """Envia vec3 para uniform."""
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
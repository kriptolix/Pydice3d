"""
dice_reader.py — Leitura do resultado de dados parados.

Lógica por tipo:
  D4   → face virada para BAIXO (normal mais próxima de Y-)
         O D4 é lido pela face da base, não do topo.
  D6   → face virada para CIMA (normal mais próxima de Y+)
  D8   → face virada para CIMA (normal mais próxima de Y+)
  D10  → vértice mais alto (maior Y no mundo) — o trapezoedro
         não tem face horizontal; cada vértice de pico identifica
         univocamente uma face.
  D12  → face virada para CIMA (normal mais próxima de Y+)
  D20  → face virada para CIMA (normal mais próxima de Y+)

Mapeamento face → valor:
  Cada tipo define FACE_NORMALS_<tipo> com as normais das faces
  no espaço LOCAL do modelo, na mesma ordem que os valores em
  FACE_VALUES_<tipo>.

  Para o D10, VERTEX_VALUES_D10 mapeia índice do vértice de pico
  (dos 5 vértices superiores, índices 0-4) ao valor da face.

  ⚠️  Estes mapeamentos assumem que o OBJ está orientado com a
      face "1" apontando para +Y em repouso (convenção padrão de
      fabricantes de dados). Se o seu OBJ tiver orientação diferente,
      ajuste as listas abaixo — os valores são intercambiáveis.
"""

import math
import numpy as np
import pybullet as pb


# ---------------------------------------------------------------------------
# Normais de face no espaço LOCAL — ordem corresponde a FACE_VALUES
# ---------------------------------------------------------------------------

# D4 — tetraedro regular, 4 faces triangulares
# Normais calculadas para tetraedro com vértices em (±1,±1,±1)
# (mesma base de _tetrahedron_verts em physics.py)
_s = 1.0 / math.sqrt(3)
FACE_NORMALS_D4 = np.array([
    [ _s,  _s,  _s],   # face oposta ao vértice (-1,-1,-1)
    [ _s, -_s, -_s],   # face oposta ao vértice (-1, 1, 1)
    [-_s,  _s, -_s],   # face oposta ao vértice ( 1,-1, 1)
    [-_s, -_s,  _s],   # face oposta ao vértice ( 1, 1,-1)
], dtype=np.float32)

# Valores lidos na face de BAIXO do D4 (convencão: face 1 fica na base ao rolar 1)
FACE_VALUES_D4 = [1, 2, 3, 4]

# D6 — cubo, 6 faces
FACE_NORMALS_D6 = np.array([
    [ 1,  0,  0],
    [-1,  0,  0],
    [ 0,  1,  0],   # topo quando valor=6
    [ 0, -1,  0],   # base quando valor=1
    [ 0,  0,  1],
    [ 0,  0, -1],
], dtype=np.float32)
FACE_VALUES_D6 = [4, 3, 6, 1, 2, 5]

# D8 — octaedro regular, 8 faces
_o = 1.0 / math.sqrt(3)
FACE_NORMALS_D8 = np.array([
    [ _o,  _o,  _o],
    [ _o,  _o, -_o],
    [ _o, -_o,  _o],
    [ _o, -_o, -_o],
    [-_o,  _o,  _o],
    [-_o,  _o, -_o],
    [-_o, -_o,  _o],
    [-_o, -_o, -_o],
], dtype=np.float32)
FACE_VALUES_D8 = [1, 2, 3, 4, 5, 6, 7, 8]

# D10 — trapezoedro pentagonal
# Usa lógica de vértice mais alto, não normal de face.
# Os 5 vértices superiores (índices 0-4 em _trapezoid_d10_verts)
# identificam as faces. Mapeamento índice → valor:
VERTEX_VALUES_D10 = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9,   # vértices de y_top
                     5: 2, 6: 4, 7: 6, 8: 8, 9: 0}    # vértices de y_bot (0=10)

# D12 — dodecaedro regular, 12 faces pentagonais
# Normais das 12 faces do dodecaedro (direções das faces para fora)
_phi = (1 + math.sqrt(5)) / 2
_d12_raw = [
    [ 0,  1,  _phi], [ 0,  1, -_phi],
    [ 0, -1,  _phi], [ 0, -1, -_phi],
    [ 1,  _phi,  0], [ 1, -_phi,  0],
    [-1,  _phi,  0], [-1, -_phi,  0],
    [ _phi,  0,  1], [ _phi,  0, -1],
    [-_phi,  0,  1], [-_phi,  0, -1],
]
_d12_norm = math.sqrt(1 + _phi * _phi)
FACE_NORMALS_D12 = np.array(
    [[x / _d12_norm, y / _d12_norm, z / _d12_norm] for x, y, z in _d12_raw],
    dtype=np.float32
)
FACE_VALUES_D12 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# D20 — icosaedro regular, 20 faces triangulares
# Normais = vetores do centro para o centróide de cada face
_phi20 = (1 + math.sqrt(5)) / 2
_ico_verts_raw = []
for _s1 in (+1, -1):
    for _s2 in (+1, -1):
        _ico_verts_raw += [
            [0,       _s1,       _s2 * _phi20],
            [_s1,     _s2 * _phi20, 0        ],
            [_s1 * _phi20, 0,    _s2        ],
        ]
_ico_norm_val = math.sqrt(1 + _phi20 * _phi20)
_ico_verts = np.array(_ico_verts_raw, dtype=np.float64) / _ico_norm_val

# Faces do icosaedro (20 triângulos) — índices padrão
_ICO_FACES = [
    (0,  4,  1), (0,  9,  4), (9,  5,  4), (4,  5,  8), (4,  8,  1),
    (8, 10,  1), (8,  3, 10), (5,  3,  8), (5,  2,  3), (2,  7,  3),
    (7, 10,  3), (7,  6, 10), (7, 11,  6), (11,  0,  6), (0,  1,  6),
    (6,  1, 10), (9,  0, 11), (9, 11,  2), (9,  2,  5), (7,  2, 11),
]
FACE_NORMALS_D20 = np.array([
    (_ico_verts[a] + _ico_verts[b] + _ico_verts[c]) / 3.0
    for a, b, c in _ICO_FACES
], dtype=np.float32)
# Normaliza centróides
for _i in range(len(FACE_NORMALS_D20)):
    _l = np.linalg.norm(FACE_NORMALS_D20[_i])
    if _l > 1e-8:
        FACE_NORMALS_D20[_i] /= _l

FACE_VALUES_D20 = list(range(1, 21))


# ---------------------------------------------------------------------------
# Tabela de despacho
# ---------------------------------------------------------------------------
_FACE_DATA = {
    "d4":  (FACE_NORMALS_D4,  FACE_VALUES_D4),
    "d6":  (FACE_NORMALS_D6,  FACE_VALUES_D6),
    "d8":  (FACE_NORMALS_D8,  FACE_VALUES_D8),
    "d12": (FACE_NORMALS_D12, FACE_VALUES_D12),
    "d20": (FACE_NORMALS_D20, FACE_VALUES_D20),
}


# ---------------------------------------------------------------------------
# Funções públicas
# ---------------------------------------------------------------------------

def _rotate_normal(local_normal: np.ndarray, quaternion) -> np.ndarray:
    """Aplica a rotação do quaternion Bullet a um vetor local → world-space."""
    rm = pb.getMatrixFromQuaternion(quaternion)
    R = np.array(rm, dtype=np.float64).reshape(3, 3)
    return R @ local_normal


def read_die(dice_type: str, body_id: int, physics_client: int) -> int | None:
    """
    Lê o resultado de um dado parado.

    Parâmetros
    ----------
    dice_type     : "d4" | "d6" | "d8" | "d10" | "d12" | "d20"
    body_id       : ID do corpo rígido no PyBullet
    physics_client: ID do cliente PyBullet

    Retorna
    -------
    Valor inteiro da face ou None se o tipo for desconhecido.
    """
    _, orn = pb.getBasePositionAndOrientation(body_id,
                                              physicsClientId=physics_client)

    # ---- D10: lógica especial via vértice mais alto ----
    if dice_type == "d10":
        return _read_d10(orn, physics_client)

    data = _FACE_DATA.get(dice_type)
    if data is None:
        print(f"[dice_reader] Tipo desconhecido: {dice_type!r}")
        return None

    face_normals, face_values = data

    # D4 usa face de baixo (Y-); demais usam face de cima (Y+)
    target = np.array([0.0, -1.0, 0.0]) if dice_type == "d4" \
             else np.array([0.0,  1.0, 0.0])

    best_dot   = -2.0
    best_value = face_values[0]

    for local_n, value in zip(face_normals, face_values):
        world_n = _rotate_normal(local_n, orn)
        dot = float(np.dot(world_n, target))
        if dot > best_dot:
            best_dot   = dot
            best_value = value

    return best_value


def _read_d10(orn, physics_client: int) -> int:
    """
    D10: encontra qual dos 10 vértices do trapezoedro está mais alto (Y+)
    no referencial do mundo e retorna o valor mapeado a ele.
    """
    from originals.physics import _trapezoid_d10_verts   # importação local para evitar ciclo
    local_verts = np.array(_trapezoid_d10_verts(r=1.0), dtype=np.float64)

    rm = pb.getMatrixFromQuaternion(orn)
    R  = np.array(rm, dtype=np.float64).reshape(3, 3)

    world_verts = (R @ local_verts.T).T   # shape (10, 3)
    highest_idx = int(np.argmax(world_verts[:, 1]))
    return VERTEX_VALUES_D10.get(highest_idx, 0)


def read_all_dice(dice_type: str,
                  dice_ids: list[int],
                  physics_client: int) -> list[int]:
    """
    Lê o resultado de todos os dados de uma rolagem.

    Retorna lista de valores na mesma ordem de dice_ids.
    Imprime resultado no terminal.
    """
    results = []
    for i, bid in enumerate(dice_ids):
        val = read_die(dice_type, bid, physics_client)
        results.append(val)

    total = sum(r for r in results if r is not None)
    labels = " + ".join(str(r) for r in results)
    if len(results) == 1:
        print(f"[Resultado] {dice_type.upper()}: {results[0]}")
    else:
        print(f"[Resultado] {len(results)}× {dice_type.upper()}: "
              f"{labels} = {total}")

    return results
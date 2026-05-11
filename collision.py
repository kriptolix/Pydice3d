import math

# ---------------------------------------------------------------------------
# Geradores de vértices para shapes de colisão
# ---------------------------------------------------------------------------

def _icosahedron_verts(r=1.0):
    """12 vértices de icosaedro regular (D20)."""
    phi = (1 + math.sqrt(5)) / 2
    pts = []
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            pts.append([0,        s1 * 1,    s2 * phi])
            pts.append([s1 * 1,   s2 * phi,  0       ])
            pts.append([s1 * phi, 0,          s2 * 1  ])
    norm = math.sqrt(1 + phi * phi)
    return [[x / norm * r, y / norm * r, z / norm * r] for x, y, z in pts]


def _octahedron_verts(r=1.0):
    """6 vértices de octaedro regular (D8)."""
    return [
        [ r,  0,  0], [-r,  0,  0],
        [ 0,  r,  0], [ 0, -r,  0],
        [ 0,  0,  r], [ 0,  0, -r],
    ]


def _tetrahedron_verts(r=1.0):
    """4 vértices de tetraedro regular (D4).
    Normalizado para que cada vértice fique a distância r do centro."""
    s = r / math.sqrt(3)
    return [
        [ s,  s,  s],
        [ s, -s, -s],
        [-s,  s, -s],
        [-s, -s,  s],
    ]


def _dodecahedron_verts(r=1.0):
    """20 vértices de dodecaedro regular (D12)."""
    phi = (1 + math.sqrt(5)) / 2
    pts = []
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            for s3 in (+1, -1):
                pts.append([s1,        s2,        s3       ])
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            pts.append([0,         s1 * phi,  s2 / phi])
            pts.append([s1 / phi,  0,         s2 * phi])
            pts.append([s1 * phi,  s2 / phi,  0       ])
    norm = math.sqrt(3)
    return [[x / norm * r, y / norm * r, z / norm * r] for x, y, z in pts]


def _trapezoid_d10_verts(r=1.0):
    """10 vértices do trapezoedro pentagonal (D10).
    Altura aumentada para evitar repouso instável em arestas."""
    verts = []
    r_xy, y_top = 0.8 * r, 0.45 * r   # era 0.22 — mais altura = mais estável
    for i in range(5):
        a = 2 * math.pi * i / 5
        verts.append([r_xy * math.cos(a), y_top, r_xy * math.sin(a)])
    r_xy, y_bot = 0.8 * r, -0.45 * r
    for i in range(5):
        a = 2 * math.pi * i / 5 + math.pi / 5
        verts.append([r_xy * math.cos(a), y_bot, r_xy * math.sin(a)])
    return verts


# Mapa: tipo → função que gera vértices do convex hull de colisão
_COLLISION_VERTS = {
    "d4":  _tetrahedron_verts,
    "d8":  _octahedron_verts,
    "d10": _trapezoid_d10_verts,
    "d12": _dodecahedron_verts,
    "d20": _icosahedron_verts,
}
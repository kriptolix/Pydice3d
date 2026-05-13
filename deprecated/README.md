# Rolador de Dados 3D — Estrutura do Projeto

## Estrutura de Arquivos

```
dice_sim/
├── ui/
│   └── main.py                  # Janela GTK4, controles, ponto de entrada
├── core/
│   ├── physics.py               # Simulação PyBullet (bandeja, dados, stepping)
│   └── dice_reader.py           # Leitura de resultado dos dados parados
└── rendering/
    ├── glarena.py               # GTK4 GLArea: ciclo OpenGL + loop de simulação
    ├── geometrics.py            # Construção de meshes, OBJ, UV, tangentes
    ├── matrices.py              # Projeção, look-at, Bullet → mat4
    ├── shaders.py               # Shaders GLSL, upload de VAO/VBO, texturas
    └── asset_loader.py          # Caminhos de assets, escala do dado
```

## Como executar

```bash
cd dice_sim
python -m ui.main        # ou: PYTHONPATH=. python ui/main.py
```

---

## Bug de textura — causa e correção

### Sintoma
`assets/d6/DefaultMaterial_Base_color.png` não era encontrado pelo
`glarena.py` original, mesmo o arquivo existindo no disco.

### Causa
O código original construía o caminho como:

```python
p = os.path.join(folder, filename)   # "assets/d6/DefaultMaterial_Base_color.png"
os.path.isfile(p)
```

Esse é um **caminho relativo**, resolvido pelo Python em relação ao
**diretório de trabalho atual** (`cwd`) do processo — ou seja, o
diretório a partir do qual você executou `python main.py`.

Se você rodava o programa de qualquer lugar que não fosse a raiz do
projeto (por exemplo: `python originals/main.py` de dentro de outro
diretório), o `cwd` não coincidia com a raiz do projeto e o arquivo
nunca era encontrado.

### Correção (`rendering/asset_loader.py`)
Todos os caminhos de assets são agora resolvidos **relativamente ao
arquivo `asset_loader.py`** usando `__file__`, que é sempre conhecido
independentemente do `cwd`:

```python
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

def _abs(rel_path: str) -> str:
    return os.path.join(_PROJECT_ROOT, rel_path)
```

Isso garante que `assets/d6/DefaultMaterial_Base_color.png` seja
sempre resolvido como `<raiz_do_projeto>/assets/d6/...`,
independentemente de onde o processo é iniciado.

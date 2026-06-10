> This file contains my personal notes about this project and is write in portuguese, my mother language. It's not intended to be used by others.

## Dependencies
pillow
PyOpenGL
PyOpenGL-accelerate
numpy
scipy 
pybullet 
pygobject (para interface gtk)
simpleaudio

## Problemas 

* Cria uma forma de escolher entre light e dark themes
* testar redimensionamento de janela
* dados empilhados não terminam a rolagem 

* adicionar capacidades de carregamento de texturas
* audio com simpleaudio

build_unicode_index duplicada em shaders.py
A função é chamada em dois lugares dentro do mesmo arquivo (build_glyph_uv_table e build_symbol_uvs), mas recria o índice toda vez. Bastaria construir o índice uma vez fora e passar para ambas, ou cachear dentro do atlas_json.

_choose_glyph_color em render_data.py importa de renderer.py
Esta é a questão estrutural mais séria do codebase. render_data.py é uma camada de dados CPU pura — ela não deveria conhecer renderer.py. O import em runtime na linha 354 (from pydice3d.renderer import DICE_COLORS, DEFAULT_DICE_COLOR, DICE_THEMES) cria uma dependência circular latente: renderer importa de render_data, e render_data importa de volta de renderer.

A solução é mover DICE_THEMES (e DEFAULT_DICE_COLOR, se sobreviver) para um módulo neutro — o candidato natural é o próprio dice_mesh.py ou um theme.py novo — e ambos render_data e renderer importariam desse lugar. _choose_glyph_color ficaria em render_data.py mas sem precisar tocar em renderer.
glyph_d100 em shaders.py

É uma função de domínio puro (converte valor de face → índice de glifo). Não usa OpenGL, não usa GLSL. Está em shaders.py provavelmente porque foi adicionada por conveniência quando as constantes de glifo foram definidas lá, mas conceitualmente pertence a render_data.py junto de build_face_glyphs, que já a usa. shaders.py deveria ser apenas compilação GL + helpers de uniform.

FUDGE_GLYPHS em render_data.py
São constantes de mapeamento glifo, dependem de GLYPH_PLUS/MINUS/BLANK de shaders.py. Tudo bem importar constantes de shaders, mas FUDGE_GLYPHS é uma lista mutável (list[int]) exposta no módulo — deveria ser uma tupla constante para consistência com o restante do código.

simulation.py importa CollisionEvent sem declará-lo
Na linha do best: dict[tuple[int, int], CollisionEvent] dentro de step(), CollisionEvent é usado como type hint mas não está nos imports do arquivo (from pydice3d.audio import DiceAudioEngine não reexporta CollisionEvent). Isso não quebra em runtime (o hint está dentro de uma função) mas quebra qualquer checagem estática.
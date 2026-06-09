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

## Decisões

em physics._make_collision_shape, r = DICE_TARGET_SIZE, o valor antigo era a metade mas dava muitos problemas de colisão. Futuramente cada dado deve ter um valor separado, assim o ajuste mais fino é possível

## Construção de detecção de colisões para tocar sons.

O motor de física detecta:

início da colisão;
intensidade do impacto;
tipo de superfície.

Então ele envia um evento para o sistema de áudio.

Fluxo conceitual
Física → Evento de colisão → Sistema de áudio → Som
O que detectar

Para cada frame:

1. Detectar colisão nova

Somente quando um contato começa.

Evite tocar som:

enquanto objetos continuam encostados;
em micro vibrações.

2. Calcular intensidade do impacto

Use:

velocidade relativa;
impulso da colisão;
energia transferida.

Exemplo:

impact = relative_velocity.length()

ou idealmente:

impact = collision_impulse

3. Aplicar threshold

Ignorar colisões pequenas:

if impact < 0.3:
    return

4. Converter impacto em áudio

Mapeie impacto para:

volume;
pitch;
escolha de sample.

Exemplo:

volume = clamp(impact / 10.0, 0.1, 1.0)

pitch = random(0.95, 1.05)

Pequena variação de pitch evita repetição artificial.

Separar tipos de som
Rolling

Som contínuo:

enquanto o dado desliza/gira;
volume depende da velocidade angular.
Collision

Som curto:

trigger instantâneo;
baseado no impacto.

. Pitch randomizado (essencial)
pitch = random(0.92, 1.08)


2. Volume baseado em impacto
volume = clamp(impact / 10.0, 0.1, 1.0)
3. EQ simples (muito importante)
impactos leves → mais médios/agudos
impactos fortes → mais graves
4. Layering (muito eficaz)

Em vez de muitos arquivos:

“hit base” (corpo do som)
“clack” (transiente curto)
“rumble” (grave opcional)

Combinados dinamicamente.

5. Micro delays (desalinhamento humano)
delay = random(0.0, 0.02)

Evita som “robotizado”.



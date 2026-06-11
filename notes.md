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

* testar redimensionamento de janela
* df lidos com valores errados

* Simulation deve usar camera, não manter o estado de camera ele mesmo
* Não existe um loader, renderer nao deveria fazer load
* Gl.arena nao deveria usar render_data diretamente, deveria usar através de renderer
* dice_state não deve recuperar valor, isso deve ser feito em results
* create_dice_set provavelmente deveria sair de dice e ir pra spawner

## Melhorias

* adicionar capacidades de carregamento de texturas




# pydice3d

> Physics-based 3D polyhedral dice simulation library for Python

pydice3d é uma biblioteca focada na simulação física e renderização de dados poliédricos utilizados em jogos de RPG e aplicações similares. O projeto foi desenvolvido para ser independente de frameworks de interface gráfica, permitindo integração com GTK, Qt, Pygame ou qualquer outra solução capaz de fornecer um contexto OpenGL.

O repositório inclui uma aplicação GTK utilizada para testes, desenvolvimento e demonstração das funcionalidades da biblioteca, mas a interface gráfica não faz parte do núcleo do projeto.

## Características

* Simulação física baseada em PyBullet
* Renderização OpenGL
* Suporte a dados poliédricos (d4, d6, d8, d10, d12, d20 e fudge dice)
* Carregamento de malhas OBJ
* Suporte a texturas
* Reprodução de efeitos sonoros
* Arquitetura independente de toolkit gráfico
* Integração simples com aplicações existentes

## Arquitetura

O projeto é dividido em camadas independentes:

```text
+-----------------------+
| GTK / Qt / Pygame     |
+-----------------------+
| OpenGL Renderer       |
+-----------------------+
| Scene Management      |
+-----------------------+
| PyBullet Physics      |
+-----------------------+
```

A biblioteca concentra toda a lógica de simulação, gerenciamento de cena, renderização e áudio.

A camada de interface gráfica é responsável apenas por:

* Criar a janela
* Fornecer o contexto OpenGL
* Processar eventos de entrada

## Aplicação de Demonstração

O repositório inclui uma interface GTK utilizada para:

* Testes manuais
* Desenvolvimento de funcionalidades
* Ajustes de física
* Validação visual

Ela não é necessária para utilizar a biblioteca.

## Assets

O projeto utiliza modelos, texturas e efeitos sonoros de terceiros.

As informações de autoria e licenciamento encontram-se em:

```text
THIRD_PARTY_LICENSES.md
```

Sempre que exigido pelas respectivas licenças, os créditos são mantidos e distribuídos juntamente com o projeto.

## Objetivos

* Fornecer uma biblioteca reutilizável para aplicações de RPG
* Permitir integração com diferentes toolkits gráficos
* Manter separação clara entre física, renderização e interface
* Facilitar a criação de aplicações de rolagem de dados visualmente ricas

## Status

O projeto encontra-se em desenvolvimento ativo mas pode estar em estado de inconsistência.

Contribuições, sugestões e relatos de problemas são bem-vindos.

## Licença

O código-fonte é distribuído sob a licença definida no arquivo LICENSE.

Os assets de terceiros permanecem sob suas respectivas licenças.

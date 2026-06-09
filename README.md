# pydice3d

> Physics-based 3D polyhedral dice simulation library for Python

Pydice3d is a library focused on the physical simulation and rendering of polyhedral dice commonly used in tabletop RPGs and similar applications. The project was designed to be independent of any graphical user interface framework, allowing integration with GTK, Qt, Pygame, or any other solution capable of providing an OpenGL context.

## Features

* Physics simulation powered by PyBullet
* OpenGL rendering
* Support for polyhedral dice (d4, d6, d8, d10, d12, d20, d100, and fudge dice)
* GUI toolkit-independent architecture
* Simple integration with existing applications

## Roadmap

* OBJ mesh loading
* Texture support
* Sound effect playback

## Architecture

The library contains all simulation, scene management, rendering, and audio logic.

The graphical user interface layer is only responsible for:

* Creating the application window
* Providing an OpenGL context
* Processing input events

## Demonstration Application

The repository includes a GTK-based interface used for:

* Manual testing
* Feature development
* Physics tuning
* Visual validation

It is not required to use the library.

## Status

The project is currently under active development and is in an alpha stage.

The architecture, APIs, and internal organization are evolving rapidly and may change significantly between releases. 

Bug reports, feature suggestions, feedback end code contributions and pull requests will only be accepted once the project reaches a more stable state and the public APIs have settled. This policy is intended to avoid wasting contributor effort on code that may become obsolete as the project evolves.

## License

The source code is distributed under the AGPL license.




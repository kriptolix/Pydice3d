# pydice3d
 
> Physics-based 3D polyhedral dice simulation library for Python
 
Pydice3d simulates and renders polyhedral dice as used in tabletop RPGs. It handles physics, geometry, rendering, and audio, exposing a single entry point — `DiceSimulation` — that any application can drive with a few lines of code.
 
The library has no dependency on any GUI toolkit. Anything that can provide an OpenGL 3.3 context (GTK, Qt, Pygame, SDL, etc.) can host it.
 
---
 
## Supported dice
 
d4, d6, d8, d10, d12, d20, d100 (percentile, paired d10), and Fudge/FATE dice (df).
 
---
 
## Quick start
 
### Headless (no rendering)
 
```python
from pydice3d.simulation import DiceSimulation
 
sim = DiceSimulation()
sim.roll({"d6": 2, "d20": 1})
 
while not sim.is_done:
    sim.step()
 
print(sim.result.as_dict())  # {"d6": [3, 5], "d20": [17]}
```
 
### With OpenGL rendering
 
```python
from pydice3d.simulation import DiceSimulation, RollResult
from pydice3d.renderer import Renderer
 
sim = DiceSimulation(on_result=lambda r: print(r.summary()))
sim.resize(viewport_w, viewport_h)
sim.roll({"d6": 3}, theme="dark")
 
renderer = Renderer(sim.scene, sim.dice_types)
 
# inside the render loop:
sim.step()                                             # advance physics, update scene
renderer.draw(sim.scene, sim.view_projection(),
              sim.camera_position(), width, height)
```
 
---
 
## Architecture
 
The library is split into clearly separated layers. Public API surface is intentionally small. For more details, check the architecture file.
 
```
pydice3d/
├── simulation.py    # Entry point. Orchestrates everything.
├── physics.py       # PyBullet world, bodies, collision events
├── spawner.py       # Spawn and launch dice into the scene
├── dice.py          # Dice entity: physics body + mesh
├── dice_state.py    # Per-die lifecycle (SPAWNED → ROLLING → SETTLING → RESTING)
├── dice_mesh.py     # Polyhedron geometry (vertices, faces, normals, face values)
├── results.py       # Roll result aggregation and completion monitoring
├── scene.py         # CPU-side render data, model matrices, themes
├── renderer.py      # OpenGL renderer (VAO/VBO, shaders, MSDF glyph atlas)
├── shaders.py       # GLSL source and shader utilities
├── camera.py        # Orbital camera, view/projection matrices
├── math_utils.py    # Quaternion and vector math
└── audio.py         # Collision and rolling audio engine
```

### Data flow per frame
 
```
PhysicsWorld.step()
    └─ DiceState.update_status()      # lifecycle transitions
    └─ RollMonitor.tick()             # detect completion
    └─ RenderScene.update()           # write model matrices from orientations
         └─ Renderer.draw()           # GPU draw calls
```
 
### Entry point contract
 
`DiceSimulation` is the only class a frontend needs to import from the library.
`RollResult` is reexported from `simulation` so frontends don't need to reach into `results`.
 
---
 
## Demonstration application
 
The repository includes a GTK4-based demo (`exemples/gtk`) used for manual testing, physics tuning, and visual validation. It is not required for using the library and is not part of the public API.
 
---
 
## Status

Active development. The project is feature-complete for the initial release but it needs to undergo more comprehensive testing. The architecture and APIs have reached a point of stability. Future updates are expected to focus primarily on bug fixes, testing, performance improvements, and incremental enhancements rather than disruptive changes.

Bug reports, suggestions, and pull requests are welcome.

 
---
 
## License
 
AGPL

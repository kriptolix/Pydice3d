"""
simulation.py – Orquestrador da Simulação de Dados

Camada intermediária entre o núcleo (física + estados) e qualquer interface
gráfica. Não importa OpenGL, GTK nem nenhum outro toolkit.

Responsabilidades
─────────────────
- Criar e destruir o PhysicsWorld
- Executar spawn_dice e manter a lista de DiceState
- Avançar a simulação frame a frame (step)
- Calcular matrizes de câmera/projeção
- Redimensionar a bandeja física quando o viewport muda
- Monitorar o término da rolagem via RollMonitor
- Expor o resultado final via `result`

O que NÃO faz
─────────────
- Nada de OpenGL (sem VAO, VBO, shaders, texturas)
- Nada de GTK/Qt/SDL (sem janelas, sinais, timers de toolkit)
- Nada de I/O (sem carregamento de assets)

Uso headless (testes, CLI, servidor)
──────────────────────────────────────
    from pydice3d.simulation import DiceSimulation

    sim = DiceSimulation()
    sim.roll({"d6": 2, "d20": 1})
    while not sim.is_done:
        sim.step()
    print(sim.result.as_dict())   # {"d6": [3, 5], "d20": [17]}

Uso com frontend OpenGL
────────────────────────
    sim = DiceSimulation()
    sim.resize(viewport_w, viewport_h)   # sincroniza bandeja e câmera
    sim.roll({"d6": 3})

    # a cada frame do loop de render:
    VP       = sim.view_projection()     # mat4 float32
    cam_pos  = sim.camera_position()     # vec3 float32
    states   = sim.states                # lista de DiceState
    sim.step()                           # avança física + monitora término
"""

from __future__ import annotations

import math
from typing import Optional, Callable

import numpy as np

from pydice3d.physics    import PhysicsWorld
from pydice3d.dice_state import DiceState
from pydice3d.spawner    import spawn_dice, SpawnConfig
from pydice3d.roll_result import RollMonitor, RollResult
from pydice3d.camera     import look_at, perspective
from pydice3d.audio      import DiceAudioEngine


# ────────────────────────────────────────────────────────────────────────────
# Parâmetros de câmera padrão
# ────────────────────────────────────────────────────────────────────────────

_DEFAULT_CAM_EYE    = np.array([0.0, 12.0,  0.0], dtype=np.float32)
_DEFAULT_CAM_CENTER = np.array([0.0,  0.0,  0.0], dtype=np.float32)
_DEFAULT_CAM_UP     = np.array([0.0,  0.0, -1.0], dtype=np.float32)
_DEFAULT_FOV_DEG    = 35.0
_DEFAULT_NEAR       = 0.1
_DEFAULT_FAR        = 50.0

# Quantos substeps de física são executados por chamada a step()
STEPS_PER_TICK: int = 4


# ────────────────────────────────────────────────────────────────────────────
# DiceSimulation
# ────────────────────────────────────────────────────────────────────────────

class DiceSimulation:
    """
    Orquestrador da simulação de dados.

    Parâmetros
    ----------
    on_result : callable(RollResult) opcional.
                Chamado exatamente uma vez quando todos os dados param.
                Se None, use a propriedade `result` após `is_done == True`.
    steps_per_tick : quantos steps de física por chamada a step().
                     Padrão: 4 — bom equilíbrio entre velocidade e suavidade.
    spawn_cfg : SpawnConfig personalizado. Usa padrão se None.
    """

    def __init__(
        self,
        on_result:      Optional[Callable[[RollResult], None]] = None,
        steps_per_tick: int = STEPS_PER_TICK,
        spawn_cfg:      Optional[SpawnConfig] = None,
    ) -> None:
        self._physics       = PhysicsWorld()
        self._states:       list[DiceState]      = []
        self._monitor:      Optional[RollMonitor] = None
        self._on_result     = on_result
        self._steps_per_tick = steps_per_tick
        self._spawn_cfg     = spawn_cfg

        # Motor de áudio — silencioso se simpleaudio não estiver instalado
        self.audio = DiceAudioEngine()

        # Câmera
        self._cam_eye    = _DEFAULT_CAM_EYE.copy()
        self._cam_center = _DEFAULT_CAM_CENTER.copy()
        self._cam_up     = _DEFAULT_CAM_UP.copy()
        self._fov_deg    = _DEFAULT_FOV_DEG
        self._near       = _DEFAULT_NEAR
        self._far        = _DEFAULT_FAR

        # Dimensões do viewport (pixels) — necessárias para a projeção
        self._vp_w: int = 660
        self._vp_h: int = 460

        self._simulating: bool = False

    # ── Configuração ─────────────────────────────────────────────────────────

    def resize(self, width: int, height: int) -> None:
        """
        Notifica a simulação do tamanho do viewport.

        Recalcula os limites da bandeja física para que ela preencha o
        frustum da câmera sem deixar espaço vazio ou cortar os dados.
        Deve ser chamado sempre que o widget de render muda de tamanho,
        antes ou depois de roll().
        """
        self._vp_w = max(width, 1)
        self._vp_h = max(height, 1)

        # Half-height da bandeja no plano Y=0 projetada pelo frustum
        half_h = math.tan(math.radians(self._fov_deg / 2)) * float(self._cam_eye[1])
        aspect = self._vp_w / self._vp_h
        half_w = half_h * aspect

        # 5 % de margem para que as paredes fiquem fora do frustum
        self._physics.resize_tray(half_w * 0.95, half_h * 0.95)

    def set_camera(
        self,
        eye:    Optional[np.ndarray] = None,
        center: Optional[np.ndarray] = None,
        up:     Optional[np.ndarray] = None,
        fov_deg: Optional[float] = None,
        near:   Optional[float] = None,
        far:    Optional[float] = None,
    ) -> None:
        """Ajusta qualquer parâmetro da câmera sem exigir todos."""
        if eye    is not None: self._cam_eye    = np.asarray(eye,    dtype=np.float32)
        if center is not None: self._cam_center = np.asarray(center, dtype=np.float32)
        if up     is not None: self._cam_up     = np.asarray(up,     dtype=np.float32)
        if fov_deg is not None: self._fov_deg   = float(fov_deg)
        if near   is not None: self._near       = float(near)
        if far    is not None: self._far        = float(far)

    # ── Controle da rolagem ──────────────────────────────────────────────────

    def roll(
        self,
        spec:      dict[str, int],
        cfg:       Optional[SpawnConfig] = None,
        on_result: Optional[Callable[[RollResult], None]] = None,
    ) -> None:
        """
        Inicia uma nova rolagem, descartando qualquer rolagem anterior.

        Parâmetros
        ----------
        spec      : dicionário {tipo: quantidade}, ex: {"d6": 2, "d20": 1}.
                    d100 adiciona automaticamente 1 d10 de unidades por dado.
        cfg       : SpawnConfig para esta rolagem (sobrescreve o padrão do __init__).
        on_result : callback para esta rolagem específica (sobrescreve o do __init__).
        """
        # Limpa estado anterior
        self._simulating = False
        self._physics.remove_all_dice()
        self._states.clear()
        self._monitor = None

        effective_cfg = cfg or self._spawn_cfg or SpawnConfig()
        result        = spawn_dice(spec=spec, physics=self._physics, cfg=effective_cfg)
        self._states  = result.states

        callback = on_result or self._on_result
        self._monitor = RollMonitor(self._states, on_complete=callback)
        self._simulating = True

    def step(self) -> None:
        """
        Avança a simulação por um tick (steps_per_tick substeps de física).

        Chame uma vez por frame do loop de render. Quando todos os dados
        param, `is_done` passa a True e o callback on_result é disparado.
        """
        if not self._simulating or not self._states:
            return

        for _ in range(self._steps_per_tick):
            self._physics.step()
            for s in self._states:
                s.update_status()

        # Áudio: colisões novas detectadas pelo motor de física
        for event in self._physics.poll_collision_events():
            self.audio.on_collision(event)

        # Áudio: loop contínuo de rolling
        self.audio.on_rolling(self._states)
        self.audio.tick()

        if self._monitor:
            self._monitor.tick()

        if self._physics.all_sleeping():
            self._simulating = False
            self.audio.on_roll_complete()

    def stop(self) -> None:
        """Interrompe a simulação sem limpar os dados (preserva poses finais)."""
        self._simulating = False
        self.audio.stop_all()

    def reset(self) -> None:
        """Remove todos os dados e reseta o estado completo."""
        self._simulating = False
        self._physics.remove_all_dice()
        self._states.clear()
        self._monitor = None
        self.audio.stop_all()

    # ── Estado e resultado ───────────────────────────────────────────────────

    @property
    def is_rolling(self) -> bool:
        """True enquanto a simulação está ativa (dados ainda em movimento)."""
        return self._simulating

    @property
    def is_done(self) -> bool:
        """True quando todos os dados pararam e o resultado está disponível."""
        return self._monitor is not None and self._monitor.completed

    @property
    def result(self) -> Optional[RollResult]:
        """
        Resultado final da rolagem. None se ainda não concluída.
        Acesse após is_done == True, ou use on_result para ser notificado.
        """
        return self._monitor.result if self._monitor else None

    @property
    def partial_result(self) -> Optional[RollResult]:
        """Resultado parcial com os dados que já pararam. Útil para HUD progressivo."""
        return self._monitor.partial_result() if self._monitor else None

    @property
    def progress(self) -> float:
        """Fração de dados parados [0.0, 1.0]."""
        return self._monitor.progress if self._monitor else 0.0

    @property
    def states(self) -> list[DiceState]:
        """Lista de DiceState dos dados ativos (leitura)."""
        return self._states

    @property
    def physics(self) -> PhysicsWorld:
        """Acesso direto ao PhysicsWorld (para usos avançados)."""
        return self._physics

    # ── Câmera ───────────────────────────────────────────────────────────────

    def view_matrix(self) -> np.ndarray:
        """Matriz view 4×4 float32 (look-at)."""
        return look_at(self._cam_eye, self._cam_center, self._cam_up)

    def projection_matrix(self) -> np.ndarray:
        """Matriz de projeção perspectiva 4×4 float32."""
        aspect = self._vp_w / max(self._vp_h, 1)
        return perspective(math.radians(self._fov_deg), aspect, self._near, self._far)

    def view_projection(self) -> np.ndarray:
        """Produto P×V como float32 — pronto para enviar ao shader."""
        return (self.projection_matrix() @ self.view_matrix()).astype(np.float32)

    def camera_position(self) -> np.ndarray:
        """Posição da câmera no espaço do mundo (vec3 float32)."""
        return self._cam_eye.copy()

    # ── Ciclo de vida do PhysicsWorld ────────────────────────────────────────

    def __del__(self) -> None:
        # PhysicsWorld já faz pb.disconnect no próprio __del__,
        # mas garantimos reset limpo se o objeto for coletado cedo.
        try:
            self.reset()
        except Exception:
            pass
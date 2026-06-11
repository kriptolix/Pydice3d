"""
roll_result.py – Sistema de Resultados de Rolagem  (Fase 9)

Responsabilidade: agregar os resultados individuais dos dados numa estrutura
semântica de rolagem, detectar quando a rolagem está completa e disparar
o callback de conclusão.

Por que um módulo separado?
────────────────────────────
  dice_state.py  → resultado de UM dado individual (face superior, valor)
  roll_result.py → resultado de UMA ROLAGEM (agregação de N dados, evento de
                   conclusão, estrutura de retorno da spec)
  app.py         → O QUE fazer quando a rolagem termina (atualizar UI, log, etc.)

Fluxo de uso
────────────
    # 1. Cria o monitor junto com os estados da rolagem
    monitor = RollMonitor(states, on_complete=meu_callback)

    # 2. A cada frame (no loop de física)
    monitor.tick()          # verifica se todos pararam; dispara callback

    # 3. Callback recebe RollResult
    def meu_callback(result: RollResult):
        print(result.as_dict())
        # → {"d6": [3, 5], "d20": [17]}

Estrutura de retorno (spec)
────────────────────────────
    {
      "d6":  [3, 5],
      "d20": [17],
      "d8":  [2, 8, 6]
    }
    - Chaves: tipo do dado ("d4", "d6", ...)
    - Valores: lista de inteiros na ordem em que os dados foram registrados
    - Apenas tipos presentes na rolagem aparecem no dicionário

Evento de conclusão
────────────────────
RollMonitor.tick() verifica a cada chamada se todos os dados em states
atingiram DiceStatus.RESTING. Quando isso acontece:
  1. Constrói o RollResult final
  2. Chama on_complete(result) exatamente UMA VEZ
  3. Marca-se como completed — chamadas subsequentes a tick() são no-ops
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pydice3d.dice_state import DiceState


# ────────────────────────────────────────────────────────────────────────────
# RollResult — estrutura de dados do resultado final
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class RollResult:
    """
    Resultado completo de uma rolagem de dados.

    Atributos
    ---------
    values_by_type : dict mapeando tipo → lista de valores
                     Ex: {"d6": [3, 5], "d20": [17]}
                     Ordem dentro de cada lista = ordem de criação dos dados.
    total          : soma de todos os valores (útil para RPG)
    dice_count     : número total de dados rolados
    all_resting    : True se todos os dados pararam (sempre True neste objeto)

    Criação
    -------
    Não instanciar diretamente — usar RollResult.from_states(states).
    """
    values_by_type: dict[str, list[int]]
    total:          int
    dice_count:     int
    all_resting:    bool = True

    @classmethod
    def from_states(cls, states: list["DiceState"]) -> "RollResult":
        """
        Constrói RollResult a partir de uma lista de DiceState.

        Regra especial do d100: cada d10 marcado como ``dice.d100_partner``
        é combinado com seu d100 correspondente para formar o valor final:
            resultado = dezenas_d100 + unidades_d10
        onde dezenas_d100 ∈ {0,10,…,90} e unidades_d10 ∈ {0,1,…,9}
        (o d10 mostra 10 como "0").
        O valor combinado entra na chave "d100"; o d10 parceiro não aparece
        separadamente em "d10".

        Parâmetros
        ----------
        states : lista de DiceState (pode conter dados ainda em movimento)

        Retorna
        -------
        RollResult com os valores disponíveis no momento da chamada.
        """
        # Separar estados em d100, d10-parceiro e resto
        d100_states    = [s for s in states
                          if s.dice.dice_type == "d100"]
        partner_states = [s for s in states
                          if s.dice.dice_type == "d10"
                          and getattr(s.dice, "d100_partner", False)]
        other_states   = [s for s in states
                          if s not in d100_states and s not in partner_states]

        values_by_type: dict[str, list[int]] = {}
        total      = 0
        dice_count = 0

        # Dados normais
        for state in other_states:
            dtype = state.dice.dice_type
            if dtype not in values_by_type:
                values_by_type[dtype] = []
            if state.result is not None:
                values_by_type[dtype].append(state.result)
                total      += state.result
                dice_count += 1

        # d100 + parceiros combinados
        if d100_states:
            if "d100" not in values_by_type:
                values_by_type["d100"] = []
            for idx, d100_state in enumerate(d100_states):
                partner = partner_states[idx] if idx < len(partner_states) else None
                if d100_state.result is not None and partner is not None and partner.result is not None:
                    units = partner.result
                    # d10 usa face_value 10 para representar "0"
                    if units == 10:
                        units = 0
                    tens = d100_state.result  # já é 0, 10, 20, ..., 90
                    combined = tens + units
                    # Convenção: 00+0 = 100
                    if combined == 0:
                        combined = 100
                    values_by_type["d100"].append(combined)
                    total      += combined
                    dice_count += 1

        all_resting = all(s.is_resting for s in states)

        return cls(
            values_by_type=values_by_type,
            total=total,
            dice_count=dice_count,
            all_resting=all_resting,
        )

    def as_dict(self) -> dict[str, list[int]]:
        """
        Retorna o formato canônico da spec:
            {"d6": [3, 5], "d20": [17]}

        Apenas tipos com pelo menos um valor são incluídos.
        """
        return {k: list(v) for k, v in self.values_by_type.items() if v}

    def values_for(self, dice_type: str) -> list[int]:
        """Retorna a lista de valores para um tipo de dado. [] se ausente."""
        return list(self.values_by_type.get(dice_type, []))

    def summary(self) -> str:
        """
        Resumo legível para UI / log.
        Ex: "d6: [3, 5]  d20: [17]  total=25"
        """
        parts = [
            f"{dtype}: {vals}"
            for dtype, vals in sorted(self.values_by_type.items())
            if vals
        ]
        return "  ".join(parts) + f"  total={self.total}"

    def __repr__(self) -> str:
        return f"RollResult({self.as_dict()}, total={self.total})"


# ────────────────────────────────────────────────────────────────────────────
# RollMonitor — detecta conclusão e dispara evento
# ────────────────────────────────────────────────────────────────────────────

class RollMonitor:
    """
    Monitora uma lista de DiceState e dispara on_complete quando todos
    atingirem DiceStatus.RESTING.

    Comportamento
    -------------
    - tick() deve ser chamado a cada frame do loop de física/UI.
    - on_complete é chamado exatamente UMA VEZ, mesmo que tick() continue
      sendo chamado depois.
    - completed fica True após o disparo — permite checagem externa.
    - partial_result() retorna os valores disponíveis a qualquer momento
      (útil para exibir resultado parcial enquanto alguns dados ainda rolam).

    Parâmetros
    ----------
    states      : lista de DiceState a monitorar (referência, não cópia)
    on_complete : callable(RollResult) chamado quando todos param
                  Pode ser None — nesse caso tick() só atualiza o estado interno.

    Exemplo
    -------
    monitor = RollMonitor(states, on_complete=lambda r: print(r.as_dict()))
    # No loop:
    while simulando:
        physics.update(dt)
        for state in states:
            state.update_orientation(dt)
            state.update_status()
        collision.resolve(states)
        monitor.tick()
    """

    def __init__(
        self,
        states:      list["DiceState"],
        on_complete: Optional[Callable[["RollResult"], None]] = None,
    ) -> None:
        self._states      = states
        self._on_complete = on_complete
        self._completed   = False
        self._result:     Optional[RollResult] = None

    # ── API pública ──────────────────────────────────────────────────

    def tick(self) -> bool:
        """
        Verifica se a rolagem terminou.

        Retorna True se completou neste tick (transição false→true).
        Retorna False se já estava completa ou ainda não terminou.

        O callback on_complete é chamado apenas na primeira transição.
        """
        if self._completed:
            return False

        if all(s.is_resting for s in self._states):
            self._result    = RollResult.from_states(self._states)
            self._completed = True
            if self._on_complete is not None:
                self._on_complete(self._result)
            return True

        return False

    @property
    def completed(self) -> bool:
        """True se todos os dados pararam e o resultado foi capturado."""
        return self._completed

    @property
    def result(self) -> Optional[RollResult]:
        """
        Resultado final da rolagem, ou None se ainda não terminou.
        Disponível após tick() retornar True.
        """
        return self._result

    def partial_result(self) -> RollResult:
        """
        Resultado parcial com os dados que já pararam.
        Pode ser chamado a qualquer momento — útil para HUD progressivo.
        """
        return RollResult.from_states(self._states)

    @property
    def resting_count(self) -> int:
        """Número de dados que já pararam."""
        return sum(1 for s in self._states if s.is_resting)

    @property
    def total_count(self) -> int:
        """Número total de dados monitorados."""
        return len(self._states)

    @property
    def progress(self) -> float:
        """Fração de dados parados [0.0, 1.0]."""
        if not self._states:
            return 1.0
        return self.resting_count / self.total_count

    def reset(self, states: Optional[list["DiceState"]] = None) -> None:
        """
        Reinicia o monitor para uma nova rolagem.

        Parâmetros
        ----------
        states : nova lista de estados (usa a lista atual se None)
        """
        if states is not None:
            self._states = states
        self._completed = False
        self._result    = None

    def __repr__(self) -> str:
        return (f"RollMonitor("
                f"{self.resting_count}/{self.total_count} resting, "
                f"completed={self._completed})")
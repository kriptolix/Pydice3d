"""
roll_result.py – Aggregates the individual data results into a scrolling semantic structure, 
detects when scrolling is complete, and triggers the completion callback.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING

from pydice3d.math_utils import quat_to_matrix as _quat_to_matrix

if TYPE_CHECKING:
    from pydice3d.dice_state import DiceState

# Helpers


def _top_face_index_standard(state: "DiceState") -> int:
    R = _quat_to_matrix(state.orientation_quat)
    normals_world = state.dice.mesh.normals @ R.T
    
    return int(np.argmax(normals_world[:, 1]))


def _top_face_index_d4(state: "DiceState") -> int:
    R = _quat_to_matrix(state.orientation_quat)
    normals_world = state.dice.mesh.normals @ R.T

    return int(np.argmin(normals_world[:, 1]))


def read_face_value(state: "DiceState") -> int:

    if state.dice.dice_type == "d4":
        fi = _top_face_index_d4(state)
    else:
        fi = _top_face_index_standard(state)

    return int(state.dice.mesh.face_values[fi])


@dataclass
class RollResult:

    values_by_type: dict[str, list[int]]
    total:          int
    dice_count:     int
    all_resting:    bool = True

    @classmethod
    def from_states(cls, states: list["DiceState"]) -> "RollResult":
        """
        Constructs RollResult from a list of DiceState.

        Special d100 rule: each d10 marked as ``dice.d100_partner``
        is combined with its corresponding d100 to form the final value:
        result = tens_d100 + units_d10

        where tens_d100 ∈ {0,10,…,90} and units_d10 ∈ {0,1,…,9}

        (the d10 shows 10 as "0").

        The combined value goes into the key "d100"; the partner d10 does not appear
        separately in "d10".        
        """

        d100_states = [s for s in states
                       if s.dice.dice_type == "d100"]
        partner_states = [s for s in states
                          if s.dice.dice_type == "d10"
                          and getattr(s.dice, "d100_partner", False)]
        other_states = [s for s in states
                        if s not in d100_states and s not in partner_states]

        values_by_type: dict[str, list[int]] = {}
        total = 0
        dice_count = 0

        for state in other_states:
            dtype = state.dice.dice_type

            if dtype not in values_by_type:
                values_by_type[dtype] = []

            if state.is_resting:
                value = read_face_value(state)
                values_by_type[dtype].append(value)
                total += value
                dice_count += 1

        if d100_states:
            if "d100" not in values_by_type:
                values_by_type["d100"] = []

            for idx, d100_state in enumerate(d100_states):
                partner = partner_states[idx] if idx < len(
                    partner_states) else None
                
                if d100_state.is_resting and partner is not None and partner.is_resting:
                    units = read_face_value(partner)

                    if units == 10:
                        units = 0

                    tens = read_face_value(d100_state)
                    combined = tens + units

                    if combined == 0:
                        combined = 100

                    values_by_type["d100"].append(combined)
                    total += combined
                    dice_count += 1

        all_resting = all(s.is_resting for s in states)

        return cls(
            values_by_type=values_by_type,
            total=total,
            dice_count=dice_count,
            all_resting=all_resting,
        )

    def as_dict(self) -> dict[str, list[int]]:
        result = {}

        for k, v in self.values_by_type.items():
            if not v:
                continue

            result[k] = list(v)

        return result

    def values_for(self, dice_type: str) -> list[int]:
        return list(self.values_by_type.get(dice_type, []))

    def summary(self) -> str:

        parts = [
            f"{dtype}: {vals}"
            for dtype, vals in sorted(self.values_by_type.items())
            if vals
        ]
        return "  ".join(parts) + f"  total={self.total}"

    def __repr__(self) -> str:
        return f"RollResult({self.as_dict()}, total={self.total})"


class RollMonitor:
    """
    Monitors a list of DiceStates and triggers on_complete when all
    reach DiceStatus.RESTING. 
    """

    def __init__(
        self,
        states:      list["DiceState"],
        on_complete: Optional[Callable[["RollResult"], None]] = None,
    ) -> None:
        self._states = states
        self._on_complete = on_complete
        self._completed = False
        self._result:     Optional[RollResult] = None

    def tick(self) -> bool:
        """
        Check if scrolling has finished.        
        """
        if self._completed:
            return False

        if all(s.is_resting for s in self._states):
            self._result = RollResult.from_states(self._states)
            self._completed = True

            if self._on_complete is not None:
                self._on_complete(self._result)

            return True

        return False

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def result(self) -> Optional[RollResult]:
        return self._result

    def partial_result(self) -> RollResult:
        return RollResult.from_states(self._states)

    @property
    def resting_count(self) -> int:
        resting_count = 0

        for state in self._states:
            if state.is_resting:
                resting_count += 1

        return resting_count

    @property
    def total_count(self) -> int:
        return len(self._states)

    @property
    def progress(self) -> float:
        """Fraction of data set [0.0, 1.0]."""
        if not self._states:
            return 1.0
        
        return self.resting_count / self.total_count

    def reset(self, states: Optional[list["DiceState"]] = None) -> None:

        if states is not None:
            self._states = states

        self._completed = False
        self._result = None

    def __repr__(self) -> str:
        return (f"RollMonitor("
                f"{self.resting_count}/{self.total_count} resting, "
                f"completed={self._completed})")

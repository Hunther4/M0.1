"""Finite State Machine (FSM) for TrainingEngine v2."""

from enum import Enum, auto


class EngineState(Enum):
    """Execution states of the TrainingEngine FSM."""

    INIT = auto()
    LOAD = auto()
    TRAIN = auto()
    VALIDATE = auto()
    SAVE = auto()
    EVALUATE = auto()
    EXPORT = auto()
    FINISHED = auto()
    ERROR = auto()


class StateMachine:
    """Finite State Machine transition controller."""

    def __init__(self, initial_state: EngineState = EngineState.INIT) -> None:
        self.current_state = initial_state
        self.history: list[EngineState] = [initial_state]

    def transition_to(self, new_state: EngineState) -> None:
        """Transition engine to a new state."""
        self.history.append(new_state)
        self.current_state = new_state

    def __repr__(self) -> str:
        return f"StateMachine(current={self.current_state.name})"

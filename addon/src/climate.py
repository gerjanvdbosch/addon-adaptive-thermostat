from enum import Enum, auto
from statemachine import StateMachine


class ClimateState(Enum):
    NIGHT = auto()
    MORNING_BOOST = auto()
    DAY_IDLE = auto()
    EVENING_BOOST = auto()
    EVENING_COAST = auto()
    PAUSED_DHW = auto()


class ClimateMachine(StateMachine):
    def __init__(self, context):
        super().__init__("CLIMATE", context)
        self.state = ClimateState.NIGHT
        self.target_temp = context.temp_night

    def on_enter(self, state):
        pass

    def process(self, plan):
        pass

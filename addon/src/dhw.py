from enum import Enum, auto
from statemachine import StateMachine


class DhwState(Enum):
    IDLE = auto()
    WAITING = auto()
    RUNNING = auto()
    DONE = auto()


class DhwMachine(StateMachine):
    def __init__(self, context):
        super().__init__("DHW", context)
        self.state = DhwState.IDLE
        # self.run_start_time = None

    def on_enter(self, state):
        if state == DhwState.RUNNING:
            # Start DHW verwarming
            pass

        elif state == DhwState.DONE:
            # Stop DHW verwarming
            pass

    def process(self, plan):
        # now = self.context.now

        if self.state == DhwState.IDLE:
            if plan.start_dhw:
                self.transition(DhwState.RUNNING, plan.reason)
                # self.start_time = self.context.now

        elif self.state == DhwState.WAITING:
            # In afwachting van starttijd
            pass

        elif self.state == DhwState.RUNNING:
            # Controleer of we klaar zijn (op temp)
            pass

import logging

from datetime import datetime

logger = logging.getLogger(__name__)


class StateMachine:
    def __init__(self, name, context):
        self.name = name
        self.context = context
        self.state = None
        self.last_transition = context.now

    def transition(self, new_state, reason=""):
        if self.state != new_state:
            duration = datetime.now() - self.last_transition
            logger.info(
                f"[{self.name}] {self.state} -> {new_state} (na {duration}). Reden: {reason}"
            )
            self.state = new_state
            self.last_transition = self.context.now
            self.on_enter(new_state)

    def on_enter(self, state):
        pass

    def process(self, plan):
        raise NotImplementedError

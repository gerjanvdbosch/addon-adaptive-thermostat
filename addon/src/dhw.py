import logging

from planner import Plan
from enum import Enum, auto
from statemachine import StateMachine

logger = logging.getLogger(__name__)


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
        # self.run_start_time = None

    def on_enter(self, state):
        if state == "RUNNING":
            logger.info("[DHW] AAN: Start verwarmen.")
            # self.ctx.ha.set_switch(self.ctx.cfg.entity_dhw, True)
        elif state in ["IDLE", "DONE"]:
            logger.info("[DHW] UIT: Stoppen.")
            # self.ctx.ha.set_switch(self.ctx.cfg.entity_dhw, False)

    def process(self, plan: Plan):
        now = self.ctx.now

        # Bereken hoe lang we al draaien (in minuten)
        # Als we niet draaien, is dit getal niet relevant
        runtime_min = 0
        if self.state == "RUNNING":
            runtime_min = (now - self.last_transition).total_seconds() / 60

        # --- TRANSITIE LOGICA ---

        if self.state == "IDLE":
            if plan.start_dhw:
                self.transition("RUNNING", plan.reason)

        elif self.state == "RUNNING":
            # 1. Is het doel bereikt? (Maximale tijd)
            if runtime_min >= self.ctx.cfg.dhw_duration_hours * 60:
                # self.ctx.dhw_done_today = True
                self.transition("DONE", "Programma voltooid")
                return

            # 2. Bescherming: Minimaal 15 minuten draaien
            # Zelfs als de zon NU wegvalt en het plan zegt "stoppen",
            # negeren we dat als we pas net bezig zijn.
            if runtime_min < 15:
                # We doen niets, we blijven RUNNING.
                return

            # 3. Abort: Mag ik stoppen?
            # Het plan zegt: start_dhw = False (omdat de zon weg is)
            # if not plan.start_dhw:
            # OPTIONEEL: Hysterese toevoegen.
            # Stop alleen als we écht geen zon meer verwachten (LOW_LIGHT)
            # Of stop direct omdat we de stroom nu uit het net trekken.
            # self.transition("IDLE", f"Afgebroken: {plan.reason}")

        elif self.state == "DONE":
            pass
            # Reset om 06:00
            # if now.hour == 6 and not self.ctx.dhw_done_today:
            # self.transition("IDLE", "Nieuwe dag")

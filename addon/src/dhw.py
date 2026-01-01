import logging
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DhwStatus(Enum):
    CRITICAL = "Noodlaad"  # Nu aan (te koud / deadline bereikt)
    PLANNED = "Gepland"  # Wachten op gepland tijdstip
    SATISFIED = "Voldaan"  # Warm genoeg
    OFF = "Uit"  # Expliciet uit (wachten)


@dataclass
class DhwContext:
    status: DhwStatus
    target_temp: float
    reason: str
    energy_needed: float = 0.0
    deadline: datetime = None


class DhwAI:
    def __init__(self, ha, opts):
        self.ha = ha
        self.min_temp = float(opts.get("dhw_min_temp", 45.0))
        self.target_temp = float(opts.get("dhw_target", 50.0))
        self.max_solar = float(opts.get("dhw_max_solar", 60.0))
        self.tank_vol = float(opts.get("dhw_volume_liters", 200))
        self.power_kw = float(opts.get("dhw_power_kw", 2.0))
        self.deadline_hour = int(opts.get("dhw_deadline_hour", 17))

        self.scheduled_start = None

    def _kwh_needed(self, current, target):
        if current >= target:
            return 0.0
        return (self.tank_vol * (target - current) * 4.186) / 3600.0

    def calculate_action(self, current_temp):
        if current_temp is None:
            return DhwContext(DhwStatus.OFF, self.min_temp, "No Data")

        now = datetime.now()

        # 1. Deadline bepalen (Vanavond 17:00 of Morgen 17:00)
        deadline = now.replace(
            hour=self.deadline_hour, minute=0, second=0, microsecond=0
        )
        if now.hour >= self.deadline_hour:
            deadline += timedelta(days=1)

        # 2. Kritiek (Te koud)
        if current_temp < self.min_temp:
            self.scheduled_start = None
            return DhwContext(
                DhwStatus.CRITICAL,
                self.target_temp,
                "Te koud",
                self._kwh_needed(current_temp, self.target_temp),
                deadline,
            )

        # 3. Gepland Moment Bereikt?
        if self.scheduled_start and now >= self.scheduled_start:
            if current_temp < self.target_temp:
                return DhwContext(
                    DhwStatus.CRITICAL,
                    self.target_temp,
                    "Geplande Start",
                    self._kwh_needed(current_temp, self.target_temp),
                    deadline,
                )
            else:
                self.scheduled_start = None  # Al warm genoeg

        # 4. Moeten we plannen?
        if current_temp < self.target_temp:
            return DhwContext(
                DhwStatus.PLANNED,
                self.target_temp,
                "In Planning",
                self._kwh_needed(current_temp, self.target_temp),
                deadline,
            )

        return DhwContext(DhwStatus.SATISFIED, self.min_temp, "Voldaan")

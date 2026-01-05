import pandas as pd

from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from enum import Enum
from typing import Optional


class HvacMode(Enum):
    OFF = 0
    DHW = 1
    HEATING = 2
    COOLING = 3
    LEGIONELLA_PREVENTION = 4
    FROST_PROTECTION = 5


class SolarStatus(Enum):
    START = "START"
    WAIT = "WAIT"
    LOW_LIGHT = "LOW_LIGHT"
    DONE = "DONE"


@dataclass
class SolarContext:
    actual_pv: float
    load_now: float
    energy_now: float
    energy_best: float
    opportunity_cost: float
    confidence: float
    action: SolarStatus
    reason: str
    planned_start: Optional[datetime] = None
    current_bias: float = 1.0


@dataclass
class Context:
    now: datetime

    hvac_mode: int | None = None

    current_pv: float = 0.0
    current_load: float = 0.0
    stable_pv: float = 0.0
    stable_load: float = 0.0

    forecast: SolarContext | None = None
    forecast_df: pd.DataFrame | None = None

    pv_buffer: deque = field(default_factory=lambda: deque(maxlen=15))
    load_buffer: deque = field(default_factory=lambda: deque(maxlen=15))

    current_slot_start: datetime | None = None
    slot_samples: list[float] = field(default_factory=list)

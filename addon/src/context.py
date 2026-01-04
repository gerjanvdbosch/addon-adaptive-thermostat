from dataclasses import dataclass
from datetime import datetime
from collections import deque


@dataclass
class Context:
    now: datetime

    pv_power: float
    current_temp: float

    dhw_is_running: bool = False

    # Solar buffers
    pv_buffer = deque(maxlen=5)
    load_buffer = deque(maxlen=5)
    current_slot_start = None
    slot_samples = []

    forecast_df = None

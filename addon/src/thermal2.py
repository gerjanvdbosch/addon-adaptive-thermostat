import logging
from datetime import datetime
from dataclasses import dataclass

# Imports
from db import save_heating_cycle, save_cooling_cycle, fetch_physics_stats
from utils import safe_float

logger = logging.getLogger(__name__)


@dataclass
class PhysicsParams:
    cooling_rate: float = 0.2  # Lekfactor (Graden/uur/deltaT)
    heating_rate: float = 0.8  # Vermogen (Graden/uur)
    last_updated: datetime = None


class ThermalPhysics:
    def __init__(self, ha_client, opts):
        self.ha = ha_client
        self.params = PhysicsParams()

        # State Tracking
        self.last_state = "off"
        self.cycle_start_ts = datetime.now()
        self.start_temp = None
        self.temp_samples = []
        self.outside_samples = []
        self.supply_samples = []

        # Laad initiële waarden uit DB
        self.update_stats()

    def update_stats(self):
        """Leest DB en update de physics parameters voor de planner."""
        heat, cool = fetch_physics_stats(days=90)
        self.params.heating_rate = float(heat)
        # Cooling rate in DB is genormaliseerd.
        # Voor planner gebruiken we vaak een vaste factor, of we moeten in planner vermenigvuldigen met delta_t.
        # Hier slaan we de 'lekfactor' op.
        self.params.cooling_rate = float(cool)
        self.params.last_updated = datetime.now()
        logger.info(
            f"Physics Updated: Heat={heat:.2f}°C/u, Cool={cool:.4f} (Lekfactor)"
        )

    def run_cycle(self, features, hvac_action):
        """
        Wordt elke minuut aangeroepen. Detecteert start/stop van verwarmen.
        """
        now = datetime.now()
        cur_temp = safe_float(features.get("current_temp"))
        out_temp = safe_float(features.get("outside_temp"))
        supply = safe_float(features.get("supply_temp", 30))

        if cur_temp is None:
            return

        # State Change Detectie
        if hvac_action != self.last_state:
            # Er is zojuist geschakeld. Sluit de VORIGE cyclus af.
            duration = (now - self.cycle_start_ts).total_seconds() / 60.0

            if self.start_temp is not None and duration > 15:
                # Bereken gemiddelden
                avg_out = (
                    sum(self.outside_samples) / len(self.outside_samples)
                    if self.outside_samples
                    else out_temp
                )
                avg_sup = (
                    sum(self.supply_samples) / len(self.supply_samples)
                    if self.supply_samples
                    else supply
                )

                # Was dit Verwarmen?
                if self.last_state == "heating":
                    save_heating_cycle(
                        self.start_temp, cur_temp, duration, avg_out, avg_sup
                    )
                    logger.info(
                        f"Cycle: Verwarmd {self.start_temp}->{cur_temp} in {int(duration)}m"
                    )

                # Was dit Afkoelen (Idle)?
                elif self.last_state == "off" or self.last_state == "idle":
                    save_cooling_cycle(self.start_temp, cur_temp, duration, avg_out)
                    logger.info(
                        f"Cycle: Afgekoeld {self.start_temp}->{cur_temp} in {int(duration)}m"
                    )

            # Start NIEUWE cyclus
            self.last_state = hvac_action
            self.cycle_start_ts = now
            self.start_temp = cur_temp
            self.temp_samples = []
            self.outside_samples = []
            self.supply_samples = []

        # Data verzamelen tijdens cyclus
        self.temp_samples.append(cur_temp)
        if out_temp:
            self.outside_samples.append(out_temp)
        if supply:
            self.supply_samples.append(supply)

        # Periodiek (bv elk uur) stats verversen uit DB
        if (
            self.params.last_updated
            and (now - self.params.last_updated).total_seconds() > 3600
        ):
            self.update_stats()

    # --- HELPER FUNCTIES VOOR DE PLANNER ---

    def time_to_heat(self, start_temp, target_temp):
        if start_temp >= target_temp:
            return 0.0
        return (target_temp - start_temp) / self.params.heating_rate

    def time_to_cool(self, start_temp, target_temp, outside_temp):
        if start_temp <= target_temp:
            return 0.0

        curr = start_temp
        hours = 0.0
        # Simuleer afkoeling (Temp verlies = rate * delta_t * tijd)
        while curr > target_temp and hours < 24:
            delta_t = curr - outside_temp
            loss = self.params.cooling_rate * delta_t * 0.25  # kwartier
            if loss <= 0:
                break
            curr -= loss
            hours += 0.25
        return hours

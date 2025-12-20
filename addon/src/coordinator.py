import logging
import time
from datetime import datetime, timedelta, timezone
from presence import PresenceAI
from thermal import ThermalAI
from solar import SolarAI
from thermostat import ThermostatAI
from ha_client import HAClient
from collector import Collector
from utils import safe_float

logger = logging.getLogger(__name__)

class ClimateCoordinator:
    def __init__(self, opts: dict):
        self.opts = opts or {}
        self.ha = HAClient(url=opts.get("supervisor_url"), token=opts.get("supervisor_token"))

        # --- AI Agents ---
        self.collector = Collector(self.ha, opts)
        self.thermostat_ai = ThermostatAI(self.ha, self.collector, opts)
        self.solar_ai = SolarAI(self.ha, opts)
        self.presence_ai = PresenceAI(self.ha, opts)
        self.thermal_ai = ThermalAI(self.ha, opts)

        # --- Compressor Health Config ---
        # Voor vloerverwarming + WP zijn lange runs cruciaal
        self.min_run_minutes = int(self.opts.get("min_run_minutes", 60))  # Moet minimaal 1 uur draaien
        self.min_off_minutes = int(self.opts.get("min_off_minutes", 30))  # Moet minimaal 30 min rusten

        # --- State ---
        self.last_hvac_action = "idle"
        self.last_action_change_ts = datetime.now(timezone.utc) - timedelta(hours=1)
        self.is_preheating = False
        self.last_training_date = None

    def _get_hvac_action(self):
        """Haalt de huidige fysieke status van de warmtepomp op."""
        state = self.ha.get_state(self.opts.get("sensor_hvac_action", "sensor.thermostat_hvac_action"))
        return state.get("state") if state else "idle"

    def _is_change_safe(self, target_setpoint, current_setpoint):
        """
        Checkt of een wijziging van het setpoint veilig is voor de compressor.
        """
        now = datetime.now(timezone.utc)
        current_action = self._get_hvac_action()
        duration_since_change = (now - self.last_action_change_ts).total_seconds() / 60

        # Update status verandering tijdstip
        if current_action != self.last_hvac_action:
            logger.info(f"Coordinator: Compressor status gewijzigd: {self.last_hvac_action} -> {current_action}")
            self.last_hvac_action = current_action
            self.last_action_change_ts = now
            duration_since_change = 0

        # CASE 1: Warmtepomp draait ("heating") en AI wil stoppen (setpoint omlaag)
        if current_action == "heating" and target_setpoint < current_setpoint:
            if duration_since_change < self.min_run_minutes:
                wait_time = self.min_run_minutes - duration_since_change
                logger.warning(f"Compressor Safety: Stoppen geblokkeerd. Draait pas {duration_since_change:.0f} min (Min: {self.min_run_minutes}, wacht nog {wait_time:.0f} min)")
                return False

        # CASE 2: Warmtepomp rust ("idle"/"off") en AI wil starten (setpoint omhoog)
        if current_action != "heating" and target_setpoint > current_setpoint:
            if duration_since_change < self.min_off_minutes:
                wait_time = self.min_off_minutes - duration_since_change
                logger.warning(f"Compressor Safety: Starten geblokkeerd. Rust pas {duration_since_change:.0f} min (Min: {self.min_off_minutes}, wacht nog {wait_time:.0f} min)")
                return False

        return True

    def run_forever(self):
        logger.info("Coordinator: Systeem gestart met Compressor Protection.")
        while True:
            try:
                self._tick()
            except Exception:
                logger.exception("Coordinator: Error")
            time.sleep(60)

    def _tick(self):
        # ... (Andere AI run_cycles blijven gelijk) ...
        self.solar_ai.run_cycle()
        self.thermal_ai.run_cycle()
        self.presence_ai.log_current_state()

        # --- Beslis Logica ---
        # In plaats van direct self.ha.set_setpoint() aan te roepen,
        # leiden we nu alle beslissingen door een centrale 'veiligheids-check'.

        is_home = self._check_is_home()
        current_sp = safe_float(self.ha.get_shadow_setpoint())

        if is_home:
            # ThermostatAI berekent het ideale setpoint
            # We passen de ThermostatAI aan zodat deze alleen setpoints 'voorstelt'
            # Maar we sturen ze pas door na de veiligheidscheck.
            self._manage_home_comfort(current_sp)
        else:
            self._handle_away_logic(current_sp)

    def _set_setpoint_safe(self, target_sp):
        """Hulpfunctie die alleen het setpoint zet als het veilig is."""
        current_sp = safe_float(self.ha.get_shadow_setpoint())

        if abs(target_sp - current_sp) < 0.1:
            return # Geen wijziging nodig

        if self._is_change_safe(target_sp, current_sp):
            logger.info(f"Coordinator: Veiligheidscheck OK. Setpoint naar {target_sp}")
            self.ha.set_setpoint(target_sp)
            self.thermostat_ai.last_known_setpoint = target_sp
        else:
            # We doen niets, de veiligheidscheck heeft al gelogd waarom
            pass

    def _handle_away_logic(self, current_sp):
        # 1. Hoe lang duurt opwarmen?
        comfort_temp = float(self.opts.get("comfort_temp", 20.5))
        eco_temp = float(self.opts.get("eco_temp", 17.0))

        minutes_needed = self.thermal_ai.predict_heating_time(comfort_temp) or 180
        should_preheat, prob = self.presence_ai.should_preheat(dynamic_minutes=minutes_needed)

        if should_preheat:
            self.is_preheating = True
            self._set_setpoint_safe(comfort_temp)
        else:
            self.is_preheating = False
            self._set_setpoint_safe(eco_temp)

    def _manage_home_comfort(self, current_sp):
        """
        Intervenieer in de ThermostatAI run_cycle om veiligheid te garanderen.
        """
        # We laten ThermostatAI zijn werk doen, maar vangen de set_setpoint op.
        # Hiervoor moeten we ThermostatAI een klein beetje aanpassen
        # zodat hij niet zelf HA aanstuurt, of we overschrijven de methode.

        # Voor deze uitleg: we gebruiken de aanbeveling van ThermostatAI
        feats = self.collector.features_from_raw(self.collector.read_sensors())
        delta = self.thermostat_ai._predict_delta(feats)
        target = current_sp + delta

        # Alleen wijzigen bij voldoende verschil
        if abs(target - current_sp) >= 0.25:
            self._set_setpoint_safe(target)
import logging
from datetime import datetime, timedelta, timezone

# Project imports
from utils import safe_float

# AI Modules (zorg dat deze imports kloppen met je bestandsstructuur)
from thermostat import ThermostatAI
from solar import SolarAI
from presence import PresenceAI
from thermal import ThermalAI

logger = logging.getLogger(__name__)


class ClimateCoordinator:
    """
    De ClimateCoordinator is de centrale manager.
    Aangepast om aangeroepen te worden door een externe scheduler.
    """

    def __init__(self, ha_client, collector, opts: dict):
        self.opts = opts or {}
        # Dependency Injection: Gebruik de reeds bestaande connecties
        self.ha = ha_client
        self.collector = collector

        logger.info("Coordinator: Initializing AI Agents...")

        # AI Agents initialiseren
        self.thermostat_ai = ThermostatAI(self.ha, self.collector, opts)
        self.solar_ai = SolarAI(self.ha, opts)
        self.presence_ai = PresenceAI(self.ha, opts)
        self.thermal_ai = ThermalAI(self.ha, opts)

        # Settings
        self.comfort_temp = float(self.opts.get("comfort_temp", 20.0))
        self.eco_temp = float(self.opts.get("eco_temp", 19.0))
        self.min_change_threshold = float(self.opts.get("min_change_threshold", 0.5))

        # Compressor Protection
        self.min_run_minutes = int(self.opts.get("min_run_minutes", 60))
        self.min_off_minutes = int(self.opts.get("min_off_minutes", 30))

        # Runtime State
        self.last_hvac_action = "idle"
        self.last_action_change_ts = datetime.now(timezone.utc) - timedelta(hours=1)
        self.is_preheating = False

        # Initialiseren van de status
        self._init_hvac_state()

    def _init_hvac_state(self):
        """Haal initiële status op bij start."""
        try:
            self.last_hvac_action = self._get_hvac_action()
        except Exception:
            pass

    # ==============================================================================
    # PUBLIC METHODS (Voor de Scheduler)
    # ==============================================================================

    def tick(self):
        """
        Wordt aangeroepen door de scheduler (bijv. elke 60 sec).
        """
        try:
            self._run_logic()
        except Exception:
            logger.exception("Coordinator: Error during tick logic")

    def solar_tick(self):
        """
        Wordt aangeroepen door de scheduler (bijv. elke 15 sec).
        """
        try:
            self.solar_ai.run_cycle()
        except Exception:
            logger.exception("Coordinator: Error during solar tick")

    def perform_nightly_training(self):
        """
        Wordt aangeroepen door de scheduler (bijv. om 03:00).
        """
        logger.info("Coordinator: Starting nightly AI training...")
        try:
            self.presence_ai.train()
            self.thermal_ai.train()
            self.solar_ai.train()
            self.thermostat_ai.train()
            logger.info("Coordinator: All models updated successfully.")
        except Exception:
            logger.exception("Coordinator: Training session failed")

    # ==============================================================================
    # INTERNE LOGICA
    # ==============================================================================

    def _run_logic(self):
        # 1. Data ophalen (CENTRAAL)
        # We halen de data niet hier op, maar gaan ervan uit dat de collector
        # via zijn eigen job recent data heeft opgehaald, of we roepen het hier aan.
        # Gezien de structuur roepen we het hier aan, maar de collector cachet het wellicht.
        try:
            # Check of we verse data nodig hebben
            raw_data = self.collector.read_sensors()
            current_sp = safe_float(self.ha.get_setpoint())
        except Exception as e:
            logger.error(f"Coordinator: Sensor read failed: {e}")
            return

        if current_sp is None:
            return

        # 2. AI Agents Updaten & Checken op User Override
        override_detected = self.thermostat_ai.update_learning_state(
            raw_data, current_sp
        )

        # Andere AI's updaten
        self.thermal_ai.run_cycle()
        self.presence_ai.log_current_state()

        if override_detected:
            return

        # 3. Beslis-logica: Ben ik thuis of niet?
        is_home = self._check_is_home()

        if is_home:
            self._manage_home_comfort(raw_data, current_sp)
        else:
            self._handle_away_logic(current_sp)

    def _check_is_home(self):
        state = self.ha.get_state("zone.home")
        if not state:
            return False
        val = state
        if str(val).isdigit():
            return int(val) > 0
        return str(val).lower() in ["home", "on", "occupied"]

    def _manage_home_comfort(self, raw_data, current_sp):
        if self.is_preheating:
            logger.info(
                "Coordinator: User arrived. Pre-heating finished -> Handover to AI."
            )
            self.is_preheating = False

        target_sp = self.thermostat_ai.get_recommended_setpoint(raw_data, current_sp)

        if abs(target_sp - current_sp) >= self.min_change_threshold:
            logger.info(f"AI Advies: Aanpassen van {current_sp} naar {target_sp:.2f}")
            self._set_setpoint_safe(target_sp)

    def _handle_away_logic(self, current_sp):
        minutes_needed = self.thermal_ai.predict_heating_time(self.comfort_temp) or 180
        should_preheat, prob = self.presence_ai.should_preheat(
            dynamic_minutes=minutes_needed
        )

        target = self.eco_temp

        if should_preheat:
            if not self.is_preheating:
                logger.info(
                    f"Coordinator: Dynamic Pre-heat trigger! Kans {prob:.2f} over {minutes_needed:.0f} min."
                )
            self.is_preheating = True
            target = self.comfort_temp
        else:
            self.is_preheating = False

        self._set_setpoint_safe(target)

    def _set_setpoint_safe(self, target_sp):
        current_sp = safe_float(self.ha.get_setpoint())
        if current_sp is None or abs(target_sp - current_sp) < 0.1:
            return

        if self._is_change_safe(target_sp, current_sp):
            logger.info(
                f"Coordinator: Veiligheidscheck OK. Setpoint naar {target_sp:.2f}"
            )
            self.ha.set_setpoint(target_sp)
            self.thermostat_ai.notify_system_change(target_sp)
        else:
            logger.debug("Coordinator: Wijziging uitgesteld door compressor protectie.")

    def _get_hvac_action(self):
        entity = self.opts.get("sensor_hvac_action", "sensor.thermostat_hvac_action")
        state = self.ha.get_state(entity)
        return state if state else "idle"

    def _is_change_safe(self, target_setpoint, current_setpoint):
        now = datetime.now(timezone.utc)
        current_action = self._get_hvac_action()

        if current_action != self.last_hvac_action:
            self.last_hvac_action = current_action
            self.last_action_change_ts = now

        duration_mins = (now - self.last_action_change_ts).total_seconds() / 60.0

        if current_action == "heating" and target_setpoint < current_setpoint:
            if duration_mins < self.min_run_minutes:
                return False

        if current_action != "heating" and target_setpoint > current_setpoint:
            if duration_mins < self.min_off_minutes:
                return False

        return True

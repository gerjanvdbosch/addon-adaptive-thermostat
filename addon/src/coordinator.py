import logging
from datetime import datetime, timedelta

# Project imports
from utils import safe_float, safe_bool

# AI Modules
from thermostat import ThermostatAI
from solar import SolarAI
from presence import PresenceAI
from thermal import ThermalAI
from collector import Collector

logger = logging.getLogger(__name__)


class ClimateCoordinator:
    """
    De ClimateCoordinator is de centrale manager van het systeem.
    Beheert de interactie tussen de verschillende AI agents en Home Assistant.
    """

    def __init__(self, ha_client, collector: Collector, opts: dict):
        self.opts = opts or {}
        # Dependency Injection: Gebruik de reeds bestaande connecties
        self.ha = ha_client
        self.collector = collector

        logger.info("Coordinator: Initializing AI Agents...")

        # AI Agents initialiseren
        self.thermostat_ai = ThermostatAI(self.ha, self.collector, opts)
        self.solar_ai = SolarAI(self.ha, opts)
        self.presence_ai = PresenceAI(opts)
        self.thermal_ai = ThermalAI(self.ha, opts)

        # Settings
        self.home_temp = float(self.opts.get("home_temp", 20.0))
        self.away_temp = float(self.opts.get("away_temp", 19.0))
        self.min_change_threshold = float(self.opts.get("min_change_threshold", 0.5))

        # Compressor Protection
        self.min_run_minutes = int(self.opts.get("min_run_minutes", 60))
        self.min_off_minutes = int(self.opts.get("min_off_minutes", 30))

        # Runtime State
        self.last_hvac_mode = "off"
        self.last_action_change_ts = datetime.now() - timedelta(hours=1)
        self.is_preheating = False

    # ==============================================================================
    # PUBLIC METHODS (Voor de Scheduler)
    # ==============================================================================

    def tick(self):
        """Wordt elke minuut aangeroepen door de main loop."""
        try:
            self._run_logic()
        except Exception:
            logger.exception("Coordinator: Error during tick logic")

    def solar_tick(self):
        """Wordt elke 15-30 seconden aangeroepen voor zonne-energie beheer."""
        try:
            self.solar_ai.run_cycle()
        except Exception:
            logger.exception("Coordinator: Error during solar tick")

    def perform_nightly_training(self):
        """Wordt 's nachts aangeroepen om alle modellen te verversen met nieuwe data."""
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
        # 1. Data ophalen
        try:
            raw_data = self.collector.read_sensors()
            features = self.collector.features_from_raw(raw_data)
            current_sp = features.get("current_setpoint")
            hvac_mode = self._get_hvac_mode(raw_data)
        except Exception as e:
            logger.error(f"Coordinator: Sensor read failed: {e}")
            return

        if current_sp is None:
            logger.error("Coordinator: Current setpoint is None, aborting tick.")
            return

        # 2. AI Agents Updaten & Checken op User Override
        override_detected = self.thermostat_ai.update_learning_state(
            raw_data, current_sp
        )

        # Update ThermalAI voor cyclus monitoring
        self.thermal_ai.run_cycle(features, hvac_mode)

        # Update PresenceAI historie
        self.presence_ai.log_current_state(features)

        if override_detected:
            logger.info(
                "Coordinator: User override actief, AI-aanpassingen gepauzeerd."
            )
            return

        # 3. Beslis-logica: Thuis vs Afwezig
        is_home = safe_bool(features.get("home_presence", 0.0))

        if is_home:
            self._manage_home_comfort(features, current_sp, hvac_mode)
        else:
            self._handle_away_logic(current_sp, features, hvac_mode)

    def _manage_home_comfort(self, features, current_sp, hvac_mode):
        """
        Logica voor als de bewoners thuis zijn.
        Nu met bescherming tegen het onderbreken van lopende WP-runs.
        """
        if self.is_preheating:
            logger.info("Coordinator: Gebruiker is thuis. Voorverwarmen voltooid.")
            self.is_preheating = False

        # Vraag de AI om het ideale setpoint
        target_sp = self.thermostat_ai.get_recommended_setpoint(features, current_sp)

        if target_sp is None:
            return

        # Als de WP aan het verwarmen is ('heating') en de AI wil de temp verlagen
        if hvac_mode == "heating" and target_sp < current_sp:
            logger.info(
                f"Coordinator: Systeem is nog bezig. "
                f"Setpoint {current_sp:.1f}C behouden om cyclus niet te onderbreken."
            )
            return

        # Alleen aanpassen als het verschil groot genoeg is (tegen pendelen)
        if abs(target_sp - current_sp) >= self.min_change_threshold:
            logger.info(
                f"Coordinator: AI adviseert wijziging: {current_sp} -> {target_sp:.2f}"
            )
            self._set_setpoint_safe(target_sp, hvac_mode)
        else:
            logger.info(
                f"Coordinator: AI advies van {current_sp} naar {target_sp:.2f} onder drempel ({self.min_change_threshold}). Geen actie."
            )

    def _handle_away_logic(self, current_sp, features, hvac_mode):
        """
        Logica voor als de bewoners weg zijn.
        Bevat bescherming om lopende warmtepomp-cycli niet te onderbreken.
        """
        # 1. Check of we moeten voorverwarmen voor een verwachte terugkomst
        minutes_needed = (
            self.thermal_ai.predict_heating_time(self.home_temp, features) or 180
        )
        should_preheat, prob = self.presence_ai.should_preheat(
            dynamic_minutes=minutes_needed
        )

        if should_preheat:
            if not self.is_preheating:
                logger.info(
                    f"Coordinator: Dynamic Pre-heat trigger! Kans {prob:.2f} over {minutes_needed:.0f} min."
                )
            self.is_preheating = True
            self._set_setpoint_safe(self.home_temp, hvac_mode)
            return

        self.is_preheating = False

        # 2. "Afmaak-logica": Als de WP nu aan het verwarmen is, verlagen we het setpoint NIET.
        # We wachten tot de kamer op temperatuur is en de WP uit zichzelf stopt.
        if hvac_mode == "heating" and current_sp > self.away_temp:
            logger.info(
                f"Coordinator: Afwezig, Systeem is nog actief. "
                f"Setpoint {current_sp:.1f}C behouden tot cyclus eindigt."
            )
            return

        # 3. Als de WP niet (meer) verwarmt, mag hij naar de afwezigheidsstand
        if abs(current_sp - self.away_temp) > 0.1:
            logger.info(
                f"Coordinator: Systeem is uit. Setpoint naar {self.away_temp}C."
            )
            self._set_setpoint_safe(self.away_temp, hvac_mode)

    def _set_setpoint_safe(self, target_sp, current_action):
        """Controleert veiligheidstijden voordat HA wordt aangestuurd."""
        # Haal meest recente setpoint op om race conditions te voorkomen
        current_sp = safe_float(self.ha.get_setpoint())
        if current_sp is None or abs(target_sp - current_sp) < 0.1:
            return

        if self._is_change_safe(target_sp, current_sp, current_action):
            logger.info(
                f"Coordinator: Veiligheidscheck OK. HA aansturen -> {target_sp:.1f}"
            )
            self.ha.set_setpoint(target_sp)
            # Informeer de AI dat dit een systeem-actie was (voorkomt User Override detectie)
            self.thermostat_ai.notify_system_change(target_sp)
        else:
            logger.debug(
                "Coordinator: Wijziging uitgesteld door compressor-bescherming."
            )

    def _get_hvac_mode(self, raw_data):
        """Vertaalt de status van HA naar een interne hvac_mode."""
        return {
            "Uit": "off",
            "Verwarmen": "heating",
            "SWW": "hot_water",
            "Koelen": "cooling",
            "Legionellapreventie": "legionella_run",
            "Vorstbescherming": "frost_protection",
        }.get(raw_data.get("hvac_mode"), "off")

    def _is_change_safe(self, target_setpoint, current_setpoint, current_action):
        """
        Compressor-bescherming:
        Voorkomt dat de warmtepomp te snel achter elkaar aan- of uitgeschakeld wordt.
        """
        now = datetime.now()

        # Als de status ('heating' vs 'off') is veranderd t.o.v. de vorige check
        if current_action != self.last_hvac_mode:
            self.last_hvac_mode = current_action
            self.last_action_change_ts = now

        duration_mins = (now - self.last_action_change_ts).total_seconds() / 60.0

        if current_action == "heating" and target_setpoint < current_setpoint:
            if duration_mins < self.min_run_minutes:
                logger.info(
                    f"Coordinator: Systeem draait pas {duration_mins:.1f} min (minimaal: {self.min_run_minutes})."
                )
                return False

        if current_action != "heating" and target_setpoint > current_setpoint:
            if duration_mins < self.min_off_minutes:
                logger.info(
                    f"Coordinator: Systeem is pas {duration_mins:.1f} min uit (minimaal: {self.min_off_minutes})."
                )
                return False

        return True

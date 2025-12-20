import logging
import time
from datetime import datetime, timedelta
import pandas as pd # Nodig voor de presence fix

# Importeer al je AI experts
from thermostat import ThermostatAI
from solar import SolarAI
from presence import PresenceAI
from thermal import ThermalAI
from ha_client import HAClient
# Zorg dat Collector geimporteerd is
from collector import Collector
from utils import safe_float

logger = logging.getLogger(__name__)

class ClimateCoordinator:
    def __init__(self, opts: dict):
        self.opts = opts or {}
        self.ha = HAClient(url=opts.get("supervisor_url"), token=opts.get("supervisor_token"))

        logger.info("Coordinator: Initializing AI Agents...")

        # 1. Correcte Initialisatie
        self.collector = Collector(self.ha, opts)
        self.thermostat_ai = ThermostatAI(self.ha, self.collector, opts)
        self.solar_ai = SolarAI(self.ha, opts)
        self.presence_ai = PresenceAI(self.ha, opts)
        self.thermal_ai = ThermalAI(self.ha, opts)

        # 2. Instellingen
        self.comfort_temp = float(self.opts.get("comfort_temp", 20.0))
        self.eco_temp = float(self.opts.get("eco_temp", 15.0))
        self.preheat_threshold = float(self.opts.get("presence_threshold", 0.75))

        # State
        self.is_preheating = False
        self.last_training_date = None # Voor robuustere scheduling

    def run_forever(self):
        logger.info("Coordinator: System started. Taking control.")
        while True:
            try:
                self._tick()
            except Exception:
                logger.exception("Coordinator: Critical error in main loop")
            time.sleep(60)

    def _tick(self):
        now = datetime.now()

        # --- A. Updates ---
        self.solar_ai.run_cycle()
        self.thermal_ai.track_cycles()
        self.presence_ai.log_current_state()

        # --- B. Nightly Training (Robuust) ---
        # Draai als het na 4u is, en we vandaag nog niet getraind hebben
        if now.hour >= 4 and self.last_training_date != now.date():
            self._perform_nightly_training()
            self.last_training_date = now.date()

        # --- C. Situatie ---
        is_physically_home = False
        state = self.ha.get_state("zone.home")
        if state and state.get("state") == "home":
             is_physically_home = True

        # Soms is zone.home een getal (aantal personen)
        if state and str(state.get("state")).isdigit() and int(state.get("state")) > 0:
             is_physically_home = True

        # --- D. Beslissing ---
        if is_physically_home:
            if self.is_preheating:
                logger.info("Coordinator: User arrived! Pre-heating finished.")
                self.is_preheating = False

            # ThermostaatAI regelt het nu (comfort stabilisatie)
            self.thermostat_ai.run_cycle()

        else:
            self._handle_away_logic()

    def _handle_away_logic(self):
        # 1. Hoe lang duurt opwarmen?
        minutes_needed = self.thermal_ai.predict_heating_time(target_temp=self.comfort_temp)
        if minutes_needed is None:
            minutes_needed = 60

        minutes_needed += 15 # Buffer

        # 2. Toekomst check
        future_arrival_prob = self._get_presence_probability(minutes_offset=minutes_needed)

        current_setpoint = safe_float(self.ha.get_shadow_setpoint()) or self.eco_temp

        # 3. Logica
        if future_arrival_prob >= self.preheat_threshold:
            # START PRE-HEAT
            if not self.is_preheating and current_setpoint < (self.comfort_temp - 0.5):
                logger.info(f"Coordinator: PRE-HEAT START (Exp arrival in {minutes_needed:.0f}m, Prob {future_arrival_prob:.2f})")
                self.ha.set_setpoint(self.comfort_temp)
                self.is_preheating = True

                # Voorkom dat ThermostatAI dit leert als user-actie
                self.thermostat_ai.last_known_setpoint = self.comfort_temp

        else:
            # STOP PRE-HEAT / ECO MODE
            if self.is_preheating:
                # Hysterese: stop alleen als kans flink zakt (bv onder 55% als threshold 75% is)
                if future_arrival_prob < (self.preheat_threshold - 0.2):
                    logger.info("Coordinator: Pre-heat aborted (Prob dropped). Back to ECO.")
                    self.ha.set_setpoint(self.eco_temp)
                    self.is_preheating = False
                    self.thermostat_ai.last_known_setpoint = self.eco_temp

            elif current_setpoint > (self.eco_temp + 0.5):
                logger.info(f"Coordinator: House empty. Setting ECO ({self.eco_temp}).")
                self.ha.set_setpoint(self.eco_temp)
                self.is_preheating = False
                self.thermostat_ai.last_known_setpoint = self.eco_temp

    def _get_presence_probability(self, minutes_offset):
        """Geeft kans (0.0-1.0) dat iemand thuis is over X minuten."""
        if not self.presence_ai.is_fitted or not self.presence_ai.model:
            return 0.0

        # FIX: Gebruik de parameter minutes_offset
        future_ts = datetime.now() + timedelta(minutes=minutes_offset)

        try:
            # We moeten features maken.
            # Optie A: Roep interne methode aan (snelst, vereist dat _create_features bestaat)
            df_future = pd.DataFrame([{"timestamp": future_ts}])
            X_future = self.presence_ai._create_features(df_future)

            # predict_proba geeft [[kans_0, kans_1], ...]
            probs = self.presence_ai.model.predict_proba(X_future)[0]
            return probs[1] # Kans op WEL thuis
        except Exception as e:
            logger.error(f"Coordinator: Error predicting presence: {e}")
            return 0.0

    def _perform_nightly_training(self):
        logger.info("Coordinator: Nightly Training...")
        self.presence_ai.train()
        self.thermal_ai.train()
        self.solar_ai.train()
        self.thermostat_ai.train(force=True)
        logger.info("Coordinator: Training Done.")
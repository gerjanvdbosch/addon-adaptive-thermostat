import logging
import time
from datetime import datetime, timedelta

# Importeer al je AI experts
from thermostat import ThermostatAI
from solar import SolarAI
from presence import PresenceAI
from thermal import ThermalAI
from ha_client import HAClient
from utils import safe_float

logger = logging.getLogger(__name__)

class ClimateCoordinator:
    """
    De 'CEO' van het systeem.
    Knoopt alle AI-modellen aan elkaar om zelfstandig te beslissen
    wanneer de verwarming aan moet, gebaseerd op fysica (Thermal),
    gewoontes (Presence) en comfort (Thermostat).
    """

    def __init__(self, opts: dict):
        self.opts = opts or {}
        self.ha = HAClient(url=opts.get("supervisor_url"), token=opts.get("supervisor_token"))

        # 1. Initialiseer de AI Experts
        logger.info("Coordinator: Initializing AI Agents...")
        self.thermostat_ai = ThermostatAI(self.ha, None, opts) # Collector zit intern in ThermostatAI in de eerdere versie?
        # Correctie: ThermostatAI verwacht collector. We maken die hier aan als hij niet intern zit.
        # Voor de netheid ga ik ervan uit dat je collector meegeeft of dat ThermostatAI hem zelf maakt.
        # Hieronder de meest waarschijnlijke setup op basis van vorige code:
        from collector import Collector
        self.collector = Collector(self.ha, opts)

        self.thermostat_ai = ThermostatAI(self.ha, self.collector, opts)
        self.solar_ai = SolarAI(self.ha, opts)
        self.presence_ai = PresenceAI(self.ha, opts)
        self.thermal_ai = ThermalAI(self.ha, opts)

        # 2. Instellingen
        self.comfort_temp = float(self.opts.get("comfort_temp", 20.0))
        self.eco_temp = float(self.opts.get("eco_temp", 15.0))
        self.preheat_threshold = float(self.opts.get("presence_threshold", 0.75)) # 75% zekerheid

        # State
        self.last_tick = None
        self.is_preheating = False

    def run_forever(self):
        """De Main Loop die je in je main.py aanroept."""
        logger.info("Coordinator: System started. Taking control.")

        while True:
            try:
                self._tick()
            except Exception:
                logger.exception("Coordinator: Critical error in main loop")

            # Wacht 60 seconden voor de volgende 'hartslag'
            time.sleep(60)

    def _tick(self):
        now = datetime.now()

        # ======================================================================
        # 1. DATA VERZAMELEN & MODELLEN UPDATEN
        # ======================================================================

        # Solar (SWW) draait volledig onafhankelijk
        self.solar_ai.run_cycle()

        # Thermal leert van elke stook-cyclus
        self.thermal_ai.track_cycles()

        # Presence logt of je er nu bent (voor later leren)
        self.presence_ai.log_current_state()

        # 's Nachts trainen (bijv. om 04:00)
        if now.hour == 4 and now.minute == 0:
            self._perform_nightly_training()

        # ======================================================================
        # 2. SITUATIE BEOORDELING
        # ======================================================================

        # Check 'Harde' aanwezigheid (Sensor)
        is_physically_home = False
        state = self.ha.get_state("zone.home") # Of je geconfigureerde sensor
        if state and state.get("state") == "home":
            is_physically_home = True

        # ======================================================================
        # 3. BESLISLOGICA
        # ======================================================================

        if is_physically_home:
            # SCENARIO A: Je bent THUIS
            # --------------------------
            # We laten de ThermostatAI (de 'Finetuner') zijn werk doen.
            # Die regelt stabiliteit en kleine comfort-aanpassingen.

            if self.is_preheating:
                logger.info("Coordinator: User arrived! Pre-heating finished.")
                self.is_preheating = False

            # Draai de thermostat cycle (leert van overrides, past setpoint aan)
            self.thermostat_ai.run_cycle()

        else:
            # SCENARIO B: Je bent WEG
            # -----------------------
            # Hier neemt de Coordinator de leiding. Wij bepalen of er Eco of Pre-heat nodig is.

            self._handle_away_logic()

    def _handle_away_logic(self):
        """
        De kern van de 'Smart Pre-heat' logica.
        Knoopt ThermalAI (Fysica) aan PresenceAI (Gewoontes).
        """
        # 1. Hoe lang duurt het om naar comfort temp te gaan?
        minutes_needed = self.thermal_ai.predict_heating_time(target_temp=self.comfort_temp)

        if minutes_needed is None:
            minutes_needed = 60 # Veilige fallback als model nog niet getraind is

        # Buffer van 15 minuten erbij voor de zekerheid
        minutes_needed += 15

        # 2. Check de toekomst: Is er iemand thuis over 'minutes_needed'?
        # We gebruiken de interne logica van PresenceAI maar vragen specifiek dit tijdstip
        future_arrival_prob = self._get_presence_probability(minutes_offset=minutes_needed)

        # 3. Huidige status ophalen
        current_setpoint = safe_float(self.ha.get_shadow_setpoint()) or self.eco_temp

        # 4. BESLISSEN
        if future_arrival_prob >= self.preheat_threshold:
            # ACTIE: PRE-HEATING STARTEN

            if not self.is_preheating and current_setpoint < self.comfort_temp:
                logger.info(
                    f"Coordinator: PRE-HEAT TRIGGERED! "
                    f"Exp. Arrival in {minutes_needed:.0f}m (Prob {future_arrival_prob:.2f}). "
                    f"Heating to {self.comfort_temp}°C."
                )
                self.ha.set_setpoint(self.comfort_temp)
                self.is_preheating = True

                # Update ThermostatAI state zodat hij dit niet als 'User Override' ziet
                self.thermostat_ai.last_known_setpoint = self.comfort_temp

            elif self.is_preheating:
                logger.debug(f"Coordinator: Pre-heating active... ({minutes_needed:.0f}m remaining)")

        else:
            # ACTIE: ECO MODUS (Of blijven wachten)

            if self.is_preheating:
                # Oeps, kans is gedaald? (Bijv: je kwam toch niet). Terug naar Eco.
                # Of we zijn gewoon nog te vroeg.
                # Voor rust in systeem: als we eenmaal pre-heaten, blijven we dat meestal doen
                # tenzij de kans dramatisch zakt. Voor nu: simpel houden.
                if future_arrival_prob < (self.preheat_threshold - 0.2): # Hysterese
                    logger.info("Coordinator: Pre-heat aborted. Arrival probability dropped.")
                    self.ha.set_setpoint(self.eco_temp)
                    self.is_preheating = False

            elif current_setpoint > (self.eco_temp + 0.5):
                # Niemand thuis, geen pre-heat nodig -> Naar ECO
                logger.info(f"Coordinator: House empty. Setting ECO ({self.eco_temp}°C).")
                self.ha.set_setpoint(self.eco_temp)
                self.is_preheating = False

                # Update ThermostatAI state
                self.thermostat_ai.last_known_setpoint = self.eco_temp

    def _get_presence_probability(self, minutes_offset):
        """Helper om PresenceAI te vragen voor een specifiek moment."""
        # Omdat PresenceAI.should_preheat() misschien hardcoded is,
        # roepen we hier direct het model aan (of gebruiken we een helper in PresenceAI als je die update).
        # Dit is een 'in-place' implementatie die features maakt voor het doel-tijdstip.

        if not self.presence_ai.is_fitted or not self.presence_ai.model:
            return 0.0 # Geen model = geen gok

        import pandas as pd
        future_ts = datetime.now() + timedelta(minutes=minutes_needed)

        # Gebruik de feature logic van presence_ai
        # (Dit vereist dat _create_features 'publiek' of bereikbaar is, of we dupliceren het even voor de zekerheid)
        # Beter: Voeg een methode 'predict_proba_at(ts)' toe aan PresenceAI.
        # Hieronder de 'vuile' manier door direct de interne methode te gebruiken:

        try:
            df_future = pd.DataFrame([{"timestamp": future_ts}])
            X_future = self.presence_ai._create_features(df_future)
            probs = self.presence_ai.model.predict_proba(X_future)[0]
            return probs[1] # Kans op '1' (Thuis)
        except:
            return 0.0

    def _perform_nightly_training(self):
        """Traint alle modellen 's nachts."""
        logger.info("Coordinator: Starting Nightly Training Sequence...")

        # 1. Presence (Patronen leren)
        self.presence_ai.train()

        # 2. Thermal (Isolatie/Stookgedrag leren)
        self.thermal_ai.train()

        # 3. Solar (Zon voorspelling)
        self.solar_ai.train()

        # 4. Thermostat (Comfort delta's)
        # Deze traint vaak al direct na user-input, maar een nightly refresh kan geen kwaad
        self.thermostat_ai.train(force=True)

        logger.info("Coordinator: Nightly Training Completed.")
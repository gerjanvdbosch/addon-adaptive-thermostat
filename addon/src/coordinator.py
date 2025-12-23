import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

# Project imports
from utils import safe_float, safe_bool
from thermostat import ThermostatAI
from solar import SolarAI
from presence import PresenceAI
from thermal import ThermalAI
from collector import Collector

logger = logging.getLogger(__name__)


class HouseState(Enum):
    COMFORT = "Comfort"
    ECO = "Eco"
    PREHEAT = "Voorverwarmen"
    SOLAR_BOOST = "Zonnebuffer"
    UNKNOWN = "Onbekend"


@dataclass
class ClimateContext:
    """Bevat alle 'feiten' van dit moment."""

    current_temp: float
    current_setpoint: float
    is_compressor_active: bool
    hvac_mode: str
    is_home: bool
    ai_recommendation: float
    minutes_to_comfort: int
    preheat_needed: bool
    preheat_prob: float
    solar_excess: bool


class ClimateCoordinator:
    def __init__(self, ha_client, collector: Collector, opts: dict):
        self.ha = ha_client
        self.collector = collector
        self.opts = opts or {}

        # AI Agents
        self.thermostat_ai = ThermostatAI(self.ha, self.collector, opts)
        self.solar_ai = SolarAI(self.ha, opts)
        self.presence_ai = PresenceAI(opts)
        self.thermal_ai = ThermalAI(self.ha, opts)

        # Configuraties
        self.settings = {
            "home_fallback": float(self.opts.get("home_temp", 20.0)),
            "away_min": float(self.opts.get("away_temp", 19.0)),
            "solar_boost_delta": float(self.opts.get("solar_boost_delta", 1.0)),
            "comfort_hysteresis": float(self.opts.get("comfort_hysteresis", 0.5)),
            "min_off_min": int(self.opts.get("min_off_minutes", 30)),
            "deadband": float(self.opts.get("min_change_threshold", 0.5)),
        }

        # Veiligheid Timer
        self.last_switch_time = datetime.now() - timedelta(hours=1)

        logger.info("Coordinator: Klimaatsysteem gestart.")

    # ==============================================================================
    # MAIN LOOP
    # ==============================================================================

    def tick(self):
        try:
            # 1. Data Verzamelen
            raw = self.collector.read_sensors()
            features = self.collector.features_from_raw(raw)
            cur_sp = features.get("current_setpoint")

            if cur_sp is None:
                return

            # Bepaal status en activiteit
            hvac_mode = self._get_hvac_mode(raw)

            # 2. AI Modellen Updaten (Altijd!)
            # We geven de specifieke mode mee zodat AI leert van SWW vs Heating
            self.thermal_ai.run_cycle(features, hvac_mode)
            self.presence_ai.log_current_state(features)

            # 3. Context Bouwen & Override Check
            context = self._build_context(raw, features, cur_sp, hvac_mode)

            if self.thermostat_ai.update_learning_state(raw, cur_sp):
                logger.info(f"Coordinator: Gebruikers-override actief (Set: {cur_sp})")
                return

            # 4. Status Bepalen
            target_state = self._determine_house_state(context)

            # 5. Doel Temperatuur Berekenen (Inclusief hysteresis & probability!)
            target_temp = self._calculate_target_for_state(target_state, context)

            # 6. Uitvoeren
            self._execute_safe_transition(target_temp, context, target_state)

        except Exception:
            logger.exception("Coordinator: Fout in logic")

    def solar_tick(self):
        try:
            self.solar_ai.run_cycle()
        except Exception:
            logger.exception("Coordinator: Fout in solar logic")

    def perform_nightly_training(self):
        logger.info("Coordinator: Start nachtelijke AI training...")
        for agent in [
            self.presence_ai,
            self.thermal_ai,
            self.solar_ai,
            self.thermostat_ai,
        ]:
            try:
                agent.train()
            except Exception as e:
                logger.error(f"Training mislukt: {e}")
        logger.info("Coordinator: Training voltooid.")

    # ==============================================================================
    # INTERNE LOGICA
    # ==============================================================================

    def _build_context(self, raw, features, cur_sp, hvac_mode) -> ClimateContext:
        is_active = self._is_compressor_active(hvac_mode)
        is_home = safe_bool(features.get("home_presence", 0))

        # AI Vragen
        ai_rec = self.thermostat_ai.get_recommended_setpoint(features, cur_sp)

        # Bereken altijd de opwarmtijd
        heat_mins = (
            self.thermal_ai.predict_heating_time(
                self.settings["home_fallback"], features
            )
            or 180
        )

        # Haal probability op
        if not is_home:
            should_preheat, prob = self.presence_ai.should_preheat(
                dynamic_minutes=heat_mins
            )
        else:
            should_preheat = False
            prob = 0.0

        # Solar Status (Placeholder)
        solar_excess = False

        context = ClimateContext(
            current_temp=safe_float(features.get("current_temp", 20)),
            current_setpoint=cur_sp,
            is_compressor_active=is_active,
            hvac_mode=hvac_mode,
            is_home=is_home,
            ai_recommendation=ai_rec,
            minutes_to_comfort=heat_mins,
            preheat_needed=should_preheat,
            preheat_prob=prob,
            solar_excess=solar_excess,
        )
        return context

    def _determine_house_state(self, context: ClimateContext) -> HouseState:
        if context.is_home:
            return HouseState.COMFORT
        if context.preheat_needed:
            return HouseState.PREHEAT
        if context.solar_excess:
            return HouseState.SOLAR_BOOST
        return HouseState.ECO

    def _calculate_target_for_state(
        self, state: HouseState, context: ClimateContext
    ) -> float:
        """
        Berekent doel temperatuur.
        """

        # --- COMFORT (THUIS) ---
        if state == HouseState.COMFORT:
            base_temp = (
                context.ai_recommendation
                if context.ai_recommendation
                else self.settings["home_fallback"]
            )

            # Solar boost
            if context.solar_excess:
                return base_temp + self.settings["solar_boost_delta"]

            # HYSTERESIS LOGICA:
            # Als de compressor UIT staat, houden we hem uit tot de drempel bereikt is.
            start_threshold = base_temp - self.settings["comfort_hysteresis"]

            if not context.is_compressor_active:
                if context.current_temp is None:
                    logger.warning(
                        "Coordinator: Geen huidige temperatuur bekend, hysteresis overgeslagen."
                    )
                    return base_temp

                if context.current_temp > start_threshold:
                    if context.current_setpoint < context.current_temp:
                        logger.info(
                            f"Coordinator: Huidig {context.current_temp} > Startgrens {start_threshold:.1f}. Huidige setpoint {context.current_setpoint:.1f} behouden."
                        )
                        return context.current_setpoint

                    logger.info(
                        f"Coordinator: Huidig {context.current_temp} > Startgrens {start_threshold:.1f}. Systeem blijft UIT."
                    )

                    return start_threshold

            # Als hij WEL draait (of temp te laag), gaan we naar doel.
            return base_temp

        # --- PREHEAT (SLIM MET PROBABILITY) ---
        elif state == HouseState.PREHEAT:
            base_temp = (
                context.ai_recommendation
                if context.ai_recommendation
                else self.settings["home_fallback"]
            )

            # Als de kans > 80% is -> Volle bak stoken
            if context.preheat_prob >= 0.8:
                return base_temp

            # Als de kans tussen threshold (bv 40%) en 80% is -> Voorzichtig voorverwarmen (-0.5 graad)
            # Dit voorkomt onnodig stoken als je toch niet thuiskomt, maar zorgt dat de vloer niet ijskoud is.
            logger.info(
                f"Coordinator: Pre-heat soft start (Kans {context.preheat_prob:.2f}). Doel iets verlaagd."
            )
            return base_temp - 0.5

        # --- SOLAR BUFFER (WEG) ---
        elif state == HouseState.SOLAR_BOOST:
            return self.settings["away_min"] + 2.0

        # --- ECO (WEG) ---
        elif state == HouseState.ECO:
            return self.settings["away_min"]

        return self.settings["away_min"]

    def _execute_safe_transition(
        self, target_temp, context: ClimateContext, state: HouseState
    ):
        current_temp = context.current_setpoint

        # Deadband check
        if abs(target_temp - current_temp) < self.settings["deadband"]:
            return

        # Safety Check
        action = "heating" if target_temp > current_temp else "off"

        # We geven het hele context object mee voor de active check
        is_safe, reason_log = self._is_hardware_safe_with_reason(
            action, context, target_temp
        )

        if not is_safe:
            logger.info(
                f"Coordinator: Wacht [{state.value}] Doel {target_temp:.1f}C geweigerd: {reason_log}"
            )
            return

        # Uitvoeren
        logger.info(
            f"Coordinator: Actie [{state.value}]: Setpoint aanpassen {current_temp} -> {target_temp:.1f}C"
        )
        self.ha.set_setpoint(target_temp)
        self.thermostat_ai.notify_system_change(target_temp)

        self.last_switch_time = datetime.now()

    def _is_hardware_safe_with_reason(
        self, intended_action, context: ClimateContext, target_sp
    ):
        """
        Controleert of we mogen schakelen.
        Gebruikt context.is_compressor_active om te zien of hij draait.
        """
        now = datetime.now()
        mins_since_switch = (now - self.last_switch_time).total_seconds() / 60.0

        # REGEL 1: Starten na rust
        # Als we willen starten ('heating') en hij is nu NIET actief:
        if intended_action == "heating" and not context.is_compressor_active:
            if mins_since_switch < self.settings["min_off_min"]:
                wait_time = int(self.settings["min_off_min"] - mins_since_switch)
                return (
                    False,
                    f"Compressor staat pas {mins_since_switch:.0f}m uit (min {self.settings['min_off_min']}m). Nog {wait_time}m wachten.",
                )

        # REGEL 2: Cyclus Afmaken (Cruciaal)
        # Als hij actief is (Verwarmen OF SWW), en we willen omlaag: Niet doen.
        if context.is_compressor_active and target_sp < context.current_setpoint:
            # Kleine wijzigingen naar beneden blokkeren we.
            if (context.current_setpoint - target_sp) < 2.0:
                return (
                    False,
                    f"Systeem is actief ({context.hvac_mode}). We breken de cyclus niet af.",
                )

        return True, "OK"

    # ==============================================================================
    # SUPPORT
    # ==============================================================================

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

    def _is_compressor_active(self, hvac_mode):
        return hvac_mode in ["heating", "hot_water", "legionella_run", "cooling"]

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

# Project imports
from utils import safe_float, safe_bool, round_half
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


@dataclass
class ClimateContext:
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
            "min_safety_temp": float(self.opts.get("min_setpoint", 15.0)),
            "comfort_hysteresis": float(self.opts.get("comfort_hysteresis", 0.5)),
            "min_off_min": int(self.opts.get("min_off_minutes", 30)),
            "deadband": float(self.opts.get("min_change_threshold", 0.5)),
        }

        self.last_switch_time = datetime.now() - timedelta(hours=24)

        logger.info("Coordinator: Klimaatsysteem gestart.")

    def tick(self):
        try:
            # 1. Data Verzamelen
            raw = self.collector.read_sensors()
            features = self.collector.features_from_raw(raw)
            cur_sp = round_half(features.get("current_setpoint"))

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
                logger.info(f"Coordinator: Gebruikers-override actief ({cur_sp}).")
                self.last_switch_time = datetime.now()
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
        is_home = safe_bool(features.get("home_presence", 0))
        ai_rec = self.thermostat_ai.get_recommended_setpoint(features, cur_sp)

        # Gebruik ai_rec als basis voor de thermal check
        heat_mins = self.thermal_ai.predict_heating_time(ai_rec, features) or 180

        should_preheat, prob = (False, 0.0)
        if not is_home:
            should_preheat, prob = self.presence_ai.should_preheat(
                dynamic_minutes=heat_mins
            )

        # Solar Status (Placeholder)
        solar_excess = False

        return ClimateContext(
            current_temp=safe_float(features.get("current_temp")),
            current_setpoint=cur_sp,
            is_compressor_active=self._is_compressor_active(hvac_mode),
            hvac_mode=hvac_mode,
            is_home=is_home,
            ai_recommendation=ai_rec,
            minutes_to_comfort=heat_mins,
            preheat_needed=should_preheat,
            preheat_prob=prob,
            solar_excess=solar_excess,
        )

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
        """Berekent wat de AI zou willen doen (Strategie)."""
        base_temp = (
            context.ai_recommendation
            if context.ai_recommendation
            else self.settings["min_safety_temp"]
        )

        # --- COMFORT (THUIS) ---
        if state == HouseState.COMFORT:
            if context.solar_excess:
                return base_temp + float(self.opts.get("solar_boost_delta", 1.0))

            # Hysteresis bij opwarmen
            start_threshold = base_temp - self.settings["comfort_hysteresis"]
            if not context.is_compressor_active:
                if (
                    context.current_temp is not None
                    and context.current_temp > start_threshold
                ):
                    if context.current_setpoint < context.current_temp:
                        logger.info(
                            f"Coordinator: Hysteresis actief. Huidig {context.current_temp:.1f}C > Start {start_threshold:.1f}C. Behoud setpoint {context.current_setpoint}C."
                        )
                        return context.current_setpoint
                    logger.info(
                        f"Coordinator: Hysteresis actief. Huidig {context.current_temp:.1f}C > Start {start_threshold:.1f}C. Zet naar start {start_threshold:.1f}C."
                    )
                    return start_threshold
            logger.info(f"Coordinator: Comfort modus. Doel {base_temp:.1f}C.")
            return base_temp

        elif state == HouseState.PREHEAT:
            logger.info(
                f"Coordinator: Voorverwarmen modus. Doel {base_temp:.1f}C. Kans: {context.preheat_prob:.2f}"
            )
            return base_temp if context.preheat_prob >= 0.8 else base_temp - 0.5

        # Eco stand gebruikt ook AI advies
        return base_temp

    def _execute_safe_transition(
        self, target_temp, context: ClimateContext, state: HouseState
    ):
        current_sp = context.current_setpoint
        if abs(target_temp - current_sp) < self.settings["deadband"]:
            logger.info(
                f"Coordinator: Geen actie [{state.value}] Doel {target_temp:.1f}C binnen deadband van {self.settings['deadband']}C."
            )
            return

        intended_action = "heating" if target_temp > current_sp else "off"
        is_safe, reason = self._is_hardware_safe_with_reason(
            intended_action, context, target_temp, state
        )

        if not is_safe:
            logger.info(
                f"Coordinator: Wacht [{state.value}] Doel {target_temp:.1f}C geweigerd: {reason}"
            )
            return

        logger.info(
            f"Coordinator: [{state.value}] Setpoint aanpassen {current_sp} -> {target_temp:.1f}C (Reden: {reason})"
        )
        self.ha.set_setpoint(target_temp)
        self.thermostat_ai.notify_system_change(target_temp)
        self.last_switch_time = datetime.now()

    def _is_hardware_safe_with_reason(
        self, intended_action, context: ClimateContext, target_sp, state: HouseState
    ):
        now = datetime.now()
        mins_since_change = (now - self.last_switch_time).total_seconds() / 60.0
        ai_cooldown_mins = float(self.opts.get("cooldown_hours", 2)) * 60

        # REGEL 1: Cyclus Afmaken (Anti-pendel)
        # Als de warmtepomp draait, niet zomaar stoppen voor kleine AI wijzigingen.
        if context.is_compressor_active and target_sp < context.current_setpoint:
            if state != HouseState.ECO and (context.current_setpoint - target_sp) < 2.0:
                return False, f"Cyclus bezig ({context.hvac_mode})"

        # REGEL 2: AI Cooldown (STRICT)
        # Als er recent handmatig of door AI is geschakeld, blokkeer AI acties.
        if mins_since_change < ai_cooldown_mins:
            # NOODREM: Is het in huis écht kouder dan de startdrempel?
            # Dan mag comfort winnen van de cooldown.
            comfort_threshold = target_sp - self.settings["comfort_hysteresis"]

            if (
                context.current_temp is not None
                and context.current_temp < comfort_threshold
            ):
                return True, "Comfort prioriteit (Hysteresis grens)"

            # Anders: Blokkeer de AI. Jouw handmatige setting (of de vorige AI stand) blijft staan.
            return (
                False,
                f"In cooldown na wijziging ({int(ai_cooldown_mins - mins_since_change)}m resterend)",
            )

        # REGEL 3: Besparing
        # (Alleen bereikbaar als cooldown voorbij is)
        if target_sp < (context.current_setpoint - 0.05):
            return True, "Energiebesparing"

        # REGEL 4: Hardware rusttijd (Compressor bescherming na UIT gaan)
        if intended_action == "heating" and not context.is_compressor_active:
            if mins_since_change < self.settings["min_off_min"]:
                return False, "Compressor cooldown"

        return True, "AI Advies"

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

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

# Project imports
from utils import safe_float, safe_bool, round_half
from thermostat import ThermostatAI
from solar import Solar
from presence import PresenceAI
from thermal import ThermalAI
from collector import Collector
from dhw import DhwAI

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
        self.solar = SolarAI(self.ha, opts)
        self.presence_ai = PresenceAI(opts)
        self.thermal_ai = ThermalAI(self.ha, opts)
        self.dhw_ai = DhwAI(self.ha, opts)

        # Configuraties
        self.settings = {
            "min_safety_temp": float(self.opts.get("min_setpoint")),
            "max_safety_temp": float(self.opts.get("max_setpoint")),
            "comfort_hysteresis": float(self.opts.get("comfort_hysteresis", 1.0)),
            "min_off_min": int(self.opts.get("min_off_minutes", 30)),
            "deadband": float(self.opts.get("min_change_threshold", 0.5)),
        }

        self.p1_power = self.opts.get("sensor_p1", "sensor.p1_meter_power")

        # Drempels (in kW)
        self.boost_delta = float(self.opts.get("solar_boost_delta", 0.5))
        self.boost_on_kw = float(self.opts.get("solar_boost_on_kw", 1.0))
        self.boost_off_kw = float(self.opts.get("solar_boost_off_kw", 0.1))

        # Hoe lang mag de zon weg zijn voordat we stoppen?
        self.boost_off_delay_mins = int(
            self.opts.get("solar_boost_off_delay_minutes", 30)
        )

        # State tracking
        self.last_switch_time = datetime.now() - timedelta(hours=24)
        self.is_boosting = False

        # Tracking voor de wolken buffer
        self.boost_low_power_start_ts = None

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
            self.presence_ai.run_cycle(features)
            # self.dhw_ai.run_cycle(features, hvac_mode)

            # 3. Context Bouwen & Override Check
            context = self._build_context(raw, features, cur_sp, hvac_mode)

            if self.thermostat_ai.update_learning_state(raw, cur_sp):
                logger.info(f"Coordinator: Gebruikers-override actief ({cur_sp}).")
                self.last_switch_time = datetime.now()
                self.is_boosting = False
                self.boost_low_power_start_ts = None
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
            self.solar.run_cycle()
        except Exception:
            logger.exception("Coordinator: Fout in solar logic")

        try:
            self.solar2.run_cycle()
        except Exception as e:
            logger.exception(f"Coordinator: Fout in solar2: {e}")

    def perform_nightly_training(self):
        logger.info("Coordinator: Start nachtelijke AI training...")
        for agent in [
            self.presence_ai,
            self.thermal_ai,
            self.solar,
            self.solar2,
            self.thermostat_ai,
            self.dhw_ai,
        ]:
            try:
                agent.train()
            except Exception as e:
                logger.error(f"Training mislukt: {e}")
        logger.info("Coordinator: Training voltooid.")

    # ==============================================================================
    # INTERNE LOGICA
    # ==============================================================================

    def _check_p1_excess(self):
        """
        Bepaalt op basis van de P1 meter of er OVERTOLLIGE zonne-energie is.
        Bevat een 'Wolken-buffer' om te voorkomen dat hij direct uitvalt.
        """
        return False

        p1_state = self.ha.get_state(self.p1_power)
        # Conversie naar kW en injectie positief maken
        val = float(p1_state)
        if abs(val) > 100:
            val = val / 1000.0  # Watt naar kW correctie
        injection = -val

        now = datetime.now()

        # SCENARIO 1: We zijn aan het boosten
        if self.is_boosting:
            # Is er nog genoeg zon?
            if injection >= self.boost_off_kw:
                # Ja, zon is er (weer). Reset eventuele wolken-timer.
                if self.boost_low_power_start_ts is not None:
                    logger.info(
                        f"Coordinator: Zon terug ({injection:.2f}kW). Wolken-timer gereset."
                    )
                    self.boost_low_power_start_ts = None
                return True  # Blijf boosten

            # Nee, vermogen is gezakt (Wolk of verbruik)
            else:
                if self.boost_low_power_start_ts is None:
                    # Dit is de eerste keer dat het zakt. Start de timer.
                    self.boost_low_power_start_ts = now
                    logger.info(
                        f"Coordinator: Vermogen gezakt ({injection:.2f}kW). Wolken-buffer gestart..."
                    )
                    return True  # Blijf nog even boosten (buffer)

                # Timer loopt al. Hoe lang?
                diff_min = (now - self.boost_low_power_start_ts).total_seconds() / 60.0

                if diff_min < self.boost_off_delay_mins:
                    # We zitten nog in de buffer tijd
                    return True
                else:
                    # Buffer verlopen. Het is echt bewolkt. STOPPEN.
                    logger.info(
                        f"Coordinator: Buffer verlopen ({diff_min:.1f}m). Stop boost."
                    )
                    self.is_boosting = False
                    self.boost_low_power_start_ts = None
                    return False

        # SCENARIO 2: We zijn NIET aan het boosten
        else:
            # Starten vereist veel vermogen (geen buffer nodig bij start, wel hysteresis)
            if injection > self.boost_on_kw:
                logger.info(
                    f"Coordinator: Overschot gedetecteerd: {injection:.2f}kW. Start boost."
                )
                self.is_boosting = True
                self.boost_low_power_start_ts = None
                return True

            return False

    def _build_context(self, raw, features, cur_sp, hvac_mode) -> ClimateContext:
        is_home = safe_bool(features.get("home_presence", 0))
        ai_rec = round_half(
            self.thermostat_ai.get_recommended_setpoint(features, cur_sp)
        )

        # Gebruik ai_rec als basis voor de thermal check
        heat_mins = self.thermal_ai.predict_heating_time(ai_rec, features) or 180

        should_preheat, prob = (False, 0.0)
        if not is_home:
            should_preheat, prob = self.presence_ai.should_preheat(
                dynamic_minutes=heat_mins
            )

        solar_excess = self._check_p1_excess()

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
        if context.solar_excess:
            if (
                context.current_temp is None
                or context.current_temp <= self.settings["max_safety_temp"]
            ):  # Max temp beveiliging
                return HouseState.SOLAR_BOOST

        if context.is_home:
            return HouseState.COMFORT
        if context.preheat_needed:
            return HouseState.PREHEAT
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

        if state == HouseState.SOLAR_BOOST:
            return base_temp + self.boost_delta

        if state == HouseState.COMFORT:
            return base_temp

        if state == HouseState.PREHEAT:
            return base_temp if context.preheat_prob >= 0.8 else base_temp - 0.5

        return base_temp

    def _execute_safe_transition(
        self, target_temp, context: ClimateContext, state: HouseState
    ):
        current_sp = context.current_setpoint

        if abs(target_temp - current_sp) < self.settings["deadband"]:
            if state in [HouseState.SOLAR_BOOST, HouseState.PREHEAT]:
                if not self.thermostat_ai.learning_blocked:
                    self.thermostat_ai.notify_system_change(
                        current_sp, block_learning=True
                    )
            return

        intended_action = "heating" if target_temp > current_sp else "off"
        is_safe, reason = self._is_hardware_safe_with_reason(
            intended_action, context, target_temp, state
        )

        if not is_safe:
            logger.info(
                f"Coordinator: Wacht [{state.value}] Doel {target_temp:.1f}°C geweigerd: {reason}"
            )
            return

        logger.info(
            f"Coordinator: [{state.value}] Setpoint aanpassen {current_sp} -> {target_temp:.1f}°C (Reden: {reason})"
        )
        self.ha.set_setpoint(target_temp)

        should_block_learning = state in [HouseState.SOLAR_BOOST, HouseState.PREHEAT]
        self.thermostat_ai.notify_system_change(
            target_temp, block_learning=should_block_learning
        )
        self.last_switch_time = datetime.now()

    def _is_hardware_safe_with_reason(
        self, intended_action, context: ClimateContext, target_sp, state: HouseState
    ):
        now = datetime.now()
        mins_since_change = (now - self.last_switch_time).total_seconds() / 60.0
        ai_cooldown_mins = float(self.opts.get("cooldown_hours", 2)) * 60

        # --- PRIORITEIT 1: HARDWARE BESCHERMING (Compressor Min-Off) ---
        # Dit wint altijd. Je mag de warmtepomp niet te snel weer aanzetten.
        if intended_action == "heating" and not context.is_compressor_active:
            if mins_since_change < self.settings["min_off_min"]:
                remaining = int(self.settings["min_off_min"] - mins_since_change)
                return False, f"Compressor cooldown ({remaining}m)"

        # --- PRIORITEIT 2: SOLAR BOOST (Gratis Energie) ---
        # Als we zonne-energie hebben, mag dat de cooldown doorbreken.
        # (Behalve de hardware protectie hierboven, die is al gecheckt).
        if state == HouseState.SOLAR_BOOST and intended_action == "heating":
            return True, "Zonneboost start"

        # --- PRIORITEIT 3: ANTI-PENDEL (Tijdens bedrijf) ---
        # Als hij draait, mag hij niet zomaar uit, tenzij we naar ECO gaan.
        if context.is_compressor_active and target_sp < context.current_setpoint:
            if state != HouseState.ECO and (context.current_setpoint - target_sp) < 2.0:
                return False, f"Cyclus bezig ({context.hvac_mode})"

        # --- PRIORITEIT 4: DE GEBRUIKERS COOLDOWN ---
        if mins_since_change < ai_cooldown_mins:

            # UITZONDERING A: NOODGEVAL (Te Koud)
            # Als het binnen kouder is dan Target - Hysteresis, MOET hij aan.
            comfort_threshold = target_sp - self.settings["comfort_hysteresis"]
            if (
                context.current_temp is not None
                and context.current_temp < comfort_threshold
            ):
                # Alleen toestaan als we omhoog gaan (verwarmen)
                if target_sp > context.current_setpoint:
                    return True, "Comfort prioriteit (Te koud)"

            # UITZONDERING B: MODUS WISSEL NAAR ECO?
            # Optioneel: Als je het huis verlaat (Comfort -> Eco), wil je misschien
            # dat hij WEL direct omlaag gaat, ondanks dat je net aan de knop draaide.
            # Als je dat wilt, uncomment de volgende regels:
            # if state == HouseState.ECO and context.hvac_mode != 'off':
            #    return True, "Vertrek gedetecteerd (Naar Eco)"

            # In alle andere gevallen: BLOKKEREN.
            remaining = int(ai_cooldown_mins - mins_since_change)
            return False, f"In AI cooldown ({remaining}m)"

        # Als we hier komen is de cooldown voorbij en de hardware veilig.
        return True, "AI Advies"

    def _get_hvac_mode(self, raw_data):
        """Vertaalt de status van HA naar een interne hvac_mode."""
        return {
            "Uit": "off",
            "Verwarmen": "heating",
            "SWW": "dhw",
            "Koelen": "cooling",
            "Legionellapreventie": "legionella",
            "Vorstbescherming": "frost",
        }.get(raw_data.get("hvac_mode"), "off")

    def _is_compressor_active(self, hvac_mode):
        return hvac_mode in ["heating", "dhw", "legionella", "cooling"]

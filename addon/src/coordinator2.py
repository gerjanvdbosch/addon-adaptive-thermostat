import logging
from collector import Collector
from solar import Solar
from dhw import DhwAI, DhwStatus
from thermostat import ThermostatAI
from planner import EnergyPlanner
from utils import safe_float, round_half

logger = logging.getLogger(__name__)


class ClimateCoordinator:
    def __init__(self, ha_client, collector: Collector, opts: dict):
        self.ha = ha_client
        self.collector = collector
        self.opts = opts

        # Agents
        self.solar = Solar(ha_client, opts)
        self.dhw_ai = DhwAI(ha_client, opts)
        self.thermostat_ai = ThermostatAI(ha_client, collector, opts)

        # Planner (Gebruikt physics van thermostaat en specs van boiler)
        self.planner = EnergyPlanner(
            self.thermostat_ai.physics, dhw_specs={"power": self.dhw_ai.power_kw}
        )

    def tick(self):
        try:
            # 1. UPDATE DATA
            raw = self.collector.read_sensors()
            features = self.collector.features_from_raw(raw)
            self.solar.run_cycle()  # Forecasts update

            cur_dhw = safe_float(raw.get("dhw_temp"))
            cur_house = safe_float(raw.get("current_temp"))
            cur_sp = round_half(features.get("current_setpoint"))
            cur_load = safe_float(raw.get("current_power_consumption", 0.5))

            # 2. PLANNING FASE (Strategie)
            dhw_decision = self.dhw_ai.calculate_action(cur_dhw)

            # Moeten we een tijdslot zoeken?
            if (
                dhw_decision.status == DhwStatus.PLANNED
                and self.dhw_ai.scheduled_start is None
            ):
                df = self.solar.cached_forecast
                if df is not None:
                    best_time = self.planner.find_best_dhw_slot(
                        forecast_df=df,
                        kwh_needed=dhw_decision.energy_needed,
                        deadline=dhw_decision.deadline,
                        current_load=cur_load,
                        cur_house_temp=cur_house,
                        min_house_temp=20.0,  # Min temp huis om heating penalty te berekenen
                    )
                    if best_time:
                        self.dhw_ai.scheduled_start = best_time
                        logger.info(f"PLANNER: DHW ingepland op {best_time}")

            # 3. UITVOERINGS FASE (Prioriteit)

            # --- PRIO 1: DHW DRAAIEN ---
            # Als status CRITICAL is (door kou of planning bereikt)
            if dhw_decision.status == DhwStatus.CRITICAL:
                logger.info(f"ACTIE: Start DHW ({dhw_decision.reason})")
                self.ha.set_dhw_setpoint(dhw_decision.target_temp)

                # PAUZEER VERWARMING (Hardware Resource Lock)
                # We hoeven setpoint niet te verlagen, maar blokkeren wel het leren
                # omdat de verwarming effectief uitvalt door de driewegklep.
                self.thermostat_ai.notify_system_change(cur_sp, block_learning=True)
                return  # STOP TICK

            # Als we wachten op planning: Zet boiler laag
            if dhw_decision.status == DhwStatus.PLANNED:
                self.ha.set_dhw_setpoint(self.dhw_ai.min_temp)

            # --- PRIO 2: VERWARMING (Met Snooze) ---
            # Vraag advies (Fysica/ML)
            target_temp, reason = self.thermostat_ai.get_recommendation(
                features, cur_sp
            )

            # Check Snooze (Zon op komst?)
            snooze_temp = self.thermostat_ai.check_solar_deferral(
                cur_house, cur_sp, self.solar.cached_forecast
            )
            if snooze_temp:
                target_temp = snooze_temp
                reason = "Wacht op Zon"

            # --- PRIO 3: SOLAR BOOST (Bufferen) ---
            # Alleen als:
            # 1. DHW tevreden is
            # 2. Verwarming tevreden is (current >= target)
            # 3. Er zonne-overschot is
            solar_ctx = getattr(self.solar, "last_decision_ctx", None)

            if (
                dhw_decision.status == DhwStatus.SATISFIED
                and cur_house >= target_temp
                and solar_ctx
                and solar_ctx.action.value == "START"
            ):

                # Boost condities voldaan!
                logger.info("ACTIE: Solar Boost (Alles is op temp, energie over)")
                # Boost verwarming
                target_temp += 1.0
                # Boost DHW
                self.ha.set_dhw_setpoint(self.dhw_ai.max_solar)
                reason = "Solar Boost"

            else:
                # Geen boost, dus boiler naar ruststand
                if dhw_decision.status == DhwStatus.SATISFIED:
                    self.ha.set_dhw_setpoint(self.dhw_ai.min_temp)

            # Voer verwarming uit
            if abs(target_temp - cur_sp) > 0.1:
                logger.info(f"ACTIE: Verwarming -> {target_temp} ({reason})")
                self.ha.set_setpoint(target_temp)
                # Leer niet van trucks (Snooze, Boost, Nacht)
                self.thermostat_ai.notify_system_change(
                    target_temp, block_learning=(reason != "Comfort (Manual/ML)")
                )

        except Exception:
            logger.exception("Coordinator Tick Error")

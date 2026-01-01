import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EnergyPlanner:
    def __init__(self, physics, dhw_specs: dict):
        self.physics = physics
        self.dhw_power = dhw_specs.get("power", 2.0)  # kW

    def find_best_dhw_slot(
        self,
        forecast_df: pd.DataFrame,
        kwh_needed: float,
        deadline: datetime,
        current_load: float,
        cur_house_temp: float,
        min_house_temp: float,
    ):
        """
        Zoekt het beste blokje zonne-energie VOOR de deadline.
        Houdt rekening met:
        - Verwacht huisverbruik (Load)
        - Verwachte verwarmingsvraag (Penalty)
        """
        if forecast_df is None or forecast_df.empty:
            return None

        # 1. Benodigde Tijd
        quarters_needed = max(1, int(np.ceil((kwh_needed / self.dhw_power) * 4)))

        # 2. Filter data tot deadline
        # We moeten KLAAR zijn op de deadline
        latest_start = deadline - timedelta(minutes=quarters_needed * 15)
        df = forecast_df[forecast_df["timestamp"] <= latest_start].copy()

        if len(df) < quarters_needed:
            return None  # Deadline gemist

        solar = df["power_corrected"].to_numpy()
        # Fallback load als er geen 'projected' kolom is
        load = (
            df["projected_load"].to_numpy()
            if "projected_load" in df
            else np.full(len(df), current_load)
        )

        # 3. Bereken Heating Penalty (Simulatie)
        # Als we DHW doen, koelt het huis af. Moet de verwarming daarna harder werken?
        heating_penalty = np.zeros(len(df))
        sim_temp = cur_house_temp
        out_temps = (
            df["temperature"].to_numpy()
            if "temperature" in df
            else np.full(len(df), 10.0)
        )

        for i in range(len(df)):
            loss = self.physics.params.cooling_rate * (sim_temp - out_temps[i]) * 0.25
            sim_temp -= loss
            if sim_temp < min_house_temp:
                heating_penalty[i] = 1.0  # Stel: 1kW nodig om op temp te blijven
                sim_temp += self.physics.params.heating_rate * 0.25

        # 4. Netto Beschikbaar = Zon - (Huis + Verwarming)
        net_available = solar - (load + heating_penalty)

        # 5. Zoek beste blok (Rolling Window)
        scores = np.full(len(df), -np.inf)
        for i in range(len(df) - quarters_needed):
            # Score = Som van bruikbare zonne-energie in dit blok
            block_score = 0
            for j in range(quarters_needed):
                # Maximaal wat de boiler aankan
                usable = min(max(0, net_available[i + j]), self.dhw_power)
                block_score += usable
            scores[i] = block_score

        best_idx = int(np.argmax(scores))

        # Fallback: Als er GEEN zon is (-inf), plan zo laat mogelijk (Just-In-Time)
        if scores[best_idx] == -np.inf:
            logger.info("Planner: Geen zon, Just-In-Time planning.")
            return df["timestamp"].iloc[-1]  # Laatst mogelijke startmoment

        return df["timestamp"].iloc[best_idx]

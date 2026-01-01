# Adaptive Thermostat

AI-driven climate control optimized for modulating heat pumps and modern, highly insulated homes.

> Experimental — use in test environments only.

## Core Features

*   **Self-Learning:** Recognizes user patterns to proactively adjust temperature settings.
*   **Thermal Intelligence:** Predicts exact heat-up times based on building physics, solar gain (Watts), and heat pump flow temperature.
*   **Solar Optimization:** Automatically schedules energy-intensive tasks, such as domestic hot water (DHW), for the day's peak solar output.
*   **Smart Presence:** Dynamically pre-heats the home based on predicted arrival times and real-time thermal performance.
*   **Compressor Protection:** Prevents equipment wear and short-cycling by enforcing minimum runtimes and always allowing active cycles to complete efficiently.

## Architecture

| Module | Responsibility |
| :--- | :--- |
| **ThermostatAI** | Learns comfort preferences and daily routines. |
| **ThermalAI** | Models building physics to predict heating duration. |
| **Solar** | Optimizes consumption based on solar forecasts. |
| **PresenceAI** | Predicts arrival times for proactive heating. |
| **Coordinator** | Central manager that safely executes actions. |

## Monitoring & API
Includes a built-in **FastAPI** server to:
*   View training history and datasets.
*   Manually trigger model retraining.
*   Manage and clean up stored data points.

## Installation
[![Open your Home Assistant instance and show the add add-on repository dialog with a specific repository URL pre-filled.](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fgerjanvdbosch%2Faddon-adaptive-thermostat)

---
*Maximum comfort through intelligence, maximum savings through data.*
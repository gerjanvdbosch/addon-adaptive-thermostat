from dataclasses import dataclass


@dataclass
class Config:
    temp_night: float
    temp_morning: float
    temp_day: float
    temp_evening: float

    system_max_kw: float
    dhw_duration_hours: float
    avg_baseload_kw: float

    webapi_host: str
    webapi_port: int

    @staticmethod
    def load():
        return Config(
            temp_night=19.0,
            temp_morning=19.5,
            temp_day=19.0,
            temp_evening=20.0,
            system_max_kw=4.0,
            dhw_duration_hours=1.0,
            avg_baseload_kw=0.2,
            webapi_host="0.0.0.0",
            webapi_port=8000,
        )

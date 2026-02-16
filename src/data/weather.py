"""OpenWeatherMap API client for match-day weather data.

Fetches temperature, humidity, wind, and dew point for T20 venues.
Critical for T20 cricket: dew in evening matches makes bowling harder in 2nd innings.

Usage:
    from src.data.weather import WeatherClient

    client = WeatherClient()
    weather = client.get_weather("Wankhede Stadium, Mumbai", "2026-02-15")
"""

import requests
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from src.config import settings, CricketConstants
from src.utils.logger import logger


@dataclass
class MatchWeather:
    """Weather data for a single match."""

    venue: str
    date: str
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    dew_point_c: Optional[float] = None
    rain_probability: Optional[float] = None
    description: Optional[str] = None
    is_day_night: Optional[bool] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "venue": self.venue,
            "date": self.date,
            "temperature_c": self.temperature_c,
            "humidity_pct": self.humidity_pct,
            "wind_speed_kmh": self.wind_speed_kmh,
            "dew_point_c": self.dew_point_c,
            "rain_probability": self.rain_probability,
        }


class WeatherClient:
    """Client for OpenWeatherMap API."""

    BASE_URL = "https://api.openweathermap.org/data/2.5"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.api_keys.openweather
        self.coords = CricketConstants().VENUE_COORDS
        self._cache: dict[str, MatchWeather] = {}

    @property
    def is_configured(self) -> bool:
        """Check if API key is set."""
        return bool(self.api_key) and self.api_key != "your_key_here"

    def get_weather(self, venue: str, match_date: str) -> MatchWeather:
        """Get weather data for a venue on a specific date.

        Args:
            venue: Venue name (must be in VENUE_COORDS).
            match_date: Date string in YYYY-MM-DD format.

        Returns:
            MatchWeather dataclass with weather data.
            Returns empty MatchWeather if API is not configured or venue unknown.
        """
        cache_key = f"{venue}_{match_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Fallback if no API key or unknown venue
        if not self.is_configured:
            logger.debug("Weather API not configured — returning empty weather data")
            return MatchWeather(venue=venue, date=match_date)

        # Find matching venue coordinates (fuzzy match)
        coords = self._find_venue_coords(venue)
        if coords is None:
            logger.warning(f"Unknown venue for weather: '{venue}'")
            return MatchWeather(venue=venue, date=match_date)

        lat, lon = coords

        try:
            weather = self._fetch_forecast(lat, lon, venue, match_date)
            self._cache[cache_key] = weather
            return weather
        except Exception as e:
            logger.error(f"Weather API failed for {venue}: {e}")
            return MatchWeather(venue=venue, date=match_date)

    def _find_venue_coords(self, venue: str) -> Optional[tuple[float, float]]:
        """Fuzzy match venue name to known coordinates."""
        venue_lower = venue.lower()
        for known_venue, coords in self.coords.items():
            if known_venue.lower() in venue_lower or venue_lower in known_venue.lower():
                return coords
        # Try partial keyword matching
        for known_venue, coords in self.coords.items():
            keywords = known_venue.lower().split(",")[0].split()
            if any(kw in venue_lower for kw in keywords if len(kw) > 3):
                return coords
        return None

    def _fetch_forecast(
        self, lat: float, lon: float, venue: str, match_date: str
    ) -> MatchWeather:
        """Fetch weather forecast from OpenWeatherMap."""
        url = f"{self.BASE_URL}/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Find the forecast closest to match time
        # T20 evening matches in India typically start at ~19:00 IST (13:30 UTC)
        target_hour = 14  # UTC approximate
        best_forecast = self._find_closest_forecast(data["list"], match_date, target_hour)

        if best_forecast is None:
            return MatchWeather(venue=venue, date=match_date)

        main = best_forecast.get("main", {})
        wind = best_forecast.get("wind", {})

        return MatchWeather(
            venue=venue,
            date=match_date,
            temperature_c=main.get("temp"),
            humidity_pct=main.get("humidity"),
            wind_speed_kmh=round(wind.get("speed", 0) * 3.6, 1),  # m/s → km/h
            dew_point_c=self._calc_dew_point(main.get("temp"), main.get("humidity")),
            rain_probability=best_forecast.get("pop", 0) * 100,
            description=best_forecast.get("weather", [{}])[0].get("description"),
        )

    def _find_closest_forecast(
        self, forecasts: list[dict], match_date: str, target_hour: int
    ) -> Optional[dict]:
        """Find the forecast entry closest to the match time."""
        target = datetime.strptime(f"{match_date} {target_hour}:00:00", "%Y-%m-%d %H:%M:%S")
        closest = None
        min_diff = float("inf")

        for f in forecasts:
            forecast_time = datetime.strptime(f["dt_txt"], "%Y-%m-%d %H:%M:%S")
            diff = abs((forecast_time - target).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest = f

        return closest

    @staticmethod
    def _calc_dew_point(temp_c: Optional[float], humidity: Optional[float]) -> Optional[float]:
        """Approximate dew point using Magnus formula.

        Dew point is CRITICAL for T20 cricket:
        - High dew point (>20°C) = wet outfield = hard to grip ball
        - This heavily favours batting in the 2nd innings of evening matches
        """
        if temp_c is None or humidity is None:
            return None
        a, b = 17.27, 237.7
        alpha = (a * temp_c) / (b + temp_c) + np.log(humidity / 100)
        dew_point = (b * alpha) / (a - alpha)
        return round(dew_point, 1)


# Need numpy for dew point calc
import numpy as np

"""Weather API utilities for fetching live weather data."""
import urllib.request
import urllib.parse
import json
import os
import time
from typing import Dict, Optional
from dotenv import load_dotenv
from src.app.logger_config import setup_logger, log_api_call, log_api_success, log_api_error

load_dotenv()

# Set up logger for this module
logger = setup_logger(__name__)

def get_current_weather_brief(
    location: str,
    api_key: str = None,
    unit_group: str = "metric",   # "metric" for °C, "us" for °F
    timeout: int = 10,
    max_retries: int = 3
) -> Optional[Dict[str, str]]:
    """
    Fetches current weather for `location` from Visual Crossing with retry logic.
    Returns a brief dict:
      {
        "temperature": "12.3",
        "unit": "°C",
        "conditions": "Partially cloudy"
      }

    Returns None if the request fails after retries.

    Args:
        location: City name or location string
        api_key: Visual Crossing API key (defaults to env var WEATHER_API_KEY)
        unit_group: "metric" for °C, "us" for °F
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts (default 3)
    """
    if not api_key:
        api_key = os.getenv("WEATHER_API_KEY")

    if not api_key:
        logger.error("Weather API key not found. Set WEATHER_API_KEY in .env file")
        return None

    if not location or not location.strip():
        logger.warning("Location is required for weather fetch")
        return None

    base = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
    qloc = urllib.parse.quote(location.strip())
    params = {
        "unitGroup": unit_group,
        "key": api_key,
        "contentType": "json",
    }
    params_encoded = urllib.parse.urlencode(params)
    url = f"{base}{qloc}?{params_encoded}"

    # Log API call (without exposing API key)
    log_api_call(logger, "Weather API", location, {"unitGroup": unit_group})

    # Retry logic
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = json.load(resp)
                log_api_success(logger, "Weather API", f"Location: {location}")
                break  # Success, exit retry loop

        except urllib.error.HTTPError as e:
            # Server returned an error code
            try:
                detail = e.read().decode()
            except Exception:
                detail = str(e)

            error_msg = f"HTTPError {e.code}: {detail}"
            log_api_error(logger, "Weather API", Exception(error_msg), attempt)

            # Don't retry on client errors (4xx)
            if 400 <= e.code < 500:
                logger.error(f"Client error {e.code}, not retrying")
                return None

            # Retry on server errors (5xx)
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff: 2s, 4s, 8s
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Max retries ({max_retries}) reached for Weather API")
                return None

        except urllib.error.URLError as e:
            # Network issue or invalid URL
            error_msg = getattr(e, 'reason', str(e))
            log_api_error(logger, "Weather API", Exception(f"URLError: {error_msg}"), attempt)

            if attempt < max_retries:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Max retries ({max_retries}) reached for Weather API")
                return None

        except Exception as e:
            log_api_error(logger, "Weather API", e, attempt)

            if attempt < max_retries:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Max retries ({max_retries}) reached for Weather API")
                return None

    # Visual Crossing usually provides "currentConditions"
    curr = data.get("currentConditions") or {}
    temp = curr.get("temp")
    # Prefer "conditions" (e.g., "Partially cloudy") and fall back to "icon" (e.g., "partly-cloudy-day")
    conditions = curr.get("conditions") or curr.get("icon") or "Unknown"

    # If temp missing in currentConditions, try to fall back to the first day's first hour (rare)
    if temp is None:
        logger.warning("Temperature not found in currentConditions, trying fallback")
        try:
            first_hour = (data.get("days") or [{}])[0].get("hours") or []
            if first_hour:
                temp = first_hour[0].get("temp")
                logger.info("Temperature found in fallback location")
        except Exception as e:
            logger.error(f"Fallback temperature retrieval failed: {e}")

    if temp is None:
        logger.error("Temperature not found in API response after all attempts")
        return None

    unit_symbol = "°C" if unit_group.lower() == "metric" else "°F"

    logger.info(f"Weather retrieved successfully: {temp}{unit_symbol}, {conditions}")

    return {
        "temperature": f"{temp}",
        "unit": unit_symbol,
        "conditions": conditions,
    }


def format_weather_for_display(weather_data: Optional[Dict[str, str]]) -> str:
    """
    Format weather data for display in UI.

    Args:
        weather_data: Dict with temperature, unit, and conditions

    Returns:
        Formatted weather string for display
    """
    if not weather_data:
        return "Weather data unavailable"

    temp = weather_data.get("temperature", "?")
    unit = weather_data.get("unit", "")
    conditions = weather_data.get("conditions", "Unknown")

    return f"🌡️ {temp}{unit}, {conditions}"


def format_weather_for_prompt(location: str, weather_data: Optional[Dict[str, str]]) -> str:
    """
    Format weather data to include in LLM prompt.

    Args:
        location: City name
        weather_data: Dict with temperature, unit, and conditions

    Returns:
        Formatted weather context for LLM prompt
    """
    if not weather_data:
        return f"\nLocation: {location}"

    temp = weather_data.get("temperature", "?")
    unit = weather_data.get("unit", "")
    conditions = weather_data.get("conditions", "Unknown")

    return f"\nLocation: {location} (Current: {temp}{unit}, {conditions})"

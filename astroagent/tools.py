"""
Astronomy tools for the AstroAgent.
Provides various capabilities for astronomical queries and calculations.
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from astropy.time import Time
from astropy.coordinates import get_sun, get_moon, get_body, EarthLocation, AltAz, SkyCoord
from astropy import units as u
import os


def get_nasa_apod(date: Optional[str] = None) -> Dict[str, Any]:
    """
    Get NASA's Astronomy Picture of the Day.
    
    Args:
        date: Date in YYYY-MM-DD format. If None, gets today's APOD.
        
    Returns:
        Dictionary with APOD information including title, explanation, URL, etc.
    """
    api_key = os.getenv("NASA_API_KEY", "DEMO_KEY")
    base_url = "https://api.nasa.gov/planetary/apod"
    
    params = {"api_key": api_key}
    if date:
        params["date"] = date
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_near_earth_objects(start_date: str, end_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Get Near Earth Objects (asteroids) approaching Earth.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format. If None, uses start_date.
        
    Returns:
        Dictionary with NEO information
    """
    api_key = os.getenv("NASA_API_KEY", "DEMO_KEY")
    base_url = "https://api.nasa.gov/neo/rest/v1/feed"
    
    if not end_date:
        end_date = start_date
    
    params = {
        "api_key": api_key,
        "start_date": start_date,
        "end_date": end_date
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Simplify the response
        neo_count = data.get("element_count", 0)
        neos = []
        
        for date, objects in data.get("near_earth_objects", {}).items():
            for obj in objects[:3]:  # Limit to 3 per date
                neos.append({
                    "name": obj.get("name"),
                    "date": date,
                    "diameter_meters": obj.get("estimated_diameter", {}).get("meters", {}),
                    "is_hazardous": obj.get("is_potentially_hazardous_asteroid"),
                    "miss_distance_km": obj.get("close_approach_data", [{}])[0].get("miss_distance", {}).get("kilometers")
                })
        
        return {
            "total_count": neo_count,
            "near_earth_objects": neos[:10]  # Limit to 10 total
        }
    except Exception as e:
        return {"error": str(e)}


def calculate_planet_position(planet_name: str, date: Optional[str] = None, 
                              latitude: float = 0.0, longitude: float = 0.0,
                              elevation: float = 0.0) -> Dict[str, Any]:
    """
    Calculate the position of a planet in the sky for a given location and time.
    
    Args:
        planet_name: Name of the planet (e.g., 'mars', 'jupiter', 'venus')
        date: Date/time in ISO format. If None, uses current time.
        latitude: Observer latitude in degrees
        longitude: Observer longitude in degrees
        elevation: Observer elevation in meters
        
    Returns:
        Dictionary with position information (altitude, azimuth, RA, Dec)
    """
    try:
        # Parse time
        if date:
            obs_time = Time(date)
        else:
            obs_time = Time.now()
        
        # Create observer location
        location = EarthLocation(lat=latitude*u.deg, lon=longitude*u.deg, height=elevation*u.m)
        
        # Get planet position
        planet = get_body(planet_name.lower(), obs_time, location)
        
        # Convert to AltAz frame
        altaz_frame = AltAz(obstime=obs_time, location=location)
        planet_altaz = planet.transform_to(altaz_frame)
        
        # Check if visible (above horizon)
        is_visible = planet_altaz.alt.deg > 0
        
        return {
            "planet": planet_name,
            "observation_time": obs_time.iso,
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "elevation_m": elevation
            },
            "position": {
                "altitude_deg": round(planet_altaz.alt.deg, 2),
                "azimuth_deg": round(planet_altaz.az.deg, 2),
                "right_ascension": planet.ra.to_string(unit=u.hour, sep=":"),
                "declination": planet.dec.to_string(unit=u.deg, sep=":")
            },
            "is_visible": is_visible
        }
    except Exception as e:
        return {"error": str(e)}


def get_moon_phase(date: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the current moon phase and illumination.
    
    Args:
        date: Date in YYYY-MM-DD format. If None, uses current date.
        
    Returns:
        Dictionary with moon phase information
    """
    try:
        if date:
            obs_time = Time(date)
        else:
            obs_time = Time.now()
        
        # Get Sun and Moon positions
        sun = get_sun(obs_time)
        moon = get_moon(obs_time)
        
        # Calculate elongation (angle between Sun and Moon)
        elongation = sun.separation(moon).deg
        
        # Calculate phase (0 = new moon, 180 = full moon)
        # Rough illumination calculation
        illumination = (1 - abs((elongation - 180) / 180)) * 100
        
        # Determine phase name
        if elongation < 45:
            phase_name = "New Moon"
        elif elongation < 90:
            phase_name = "Waxing Crescent"
        elif elongation < 135:
            phase_name = "First Quarter"
        elif elongation < 180:
            phase_name = "Waxing Gibbous"
        elif elongation < 225:
            phase_name = "Full Moon"
        elif elongation < 270:
            phase_name = "Waning Gibbous"
        elif elongation < 315:
            phase_name = "Last Quarter"
        else:
            phase_name = "Waning Crescent"
        
        return {
            "date": obs_time.iso,
            "phase_name": phase_name,
            "illumination_percent": round(illumination, 1),
            "elongation_deg": round(elongation, 2),
            "moon_ra": moon.ra.to_string(unit=u.hour, sep=":"),
            "moon_dec": moon.dec.to_string(unit=u.deg, sep=":")
        }
    except Exception as e:
        return {"error": str(e)}


def calculate_sun_position(date: Optional[str] = None,
                          latitude: float = 0.0, longitude: float = 0.0,
                          elevation: float = 0.0) -> Dict[str, Any]:
    """
    Calculate the Sun's position in the sky for a given location and time.
    
    Args:
        date: Date/time in ISO format. If None, uses current time.
        latitude: Observer latitude in degrees
        longitude: Observer longitude in degrees
        elevation: Observer elevation in meters
        
    Returns:
        Dictionary with Sun position and rise/set information
    """
    try:
        if date:
            obs_time = Time(date)
        else:
            obs_time = Time.now()
        
        location = EarthLocation(lat=latitude*u.deg, lon=longitude*u.deg, height=elevation*u.m)
        sun = get_sun(obs_time)
        
        # Convert to AltAz frame
        altaz_frame = AltAz(obstime=obs_time, location=location)
        sun_altaz = sun.transform_to(altaz_frame)
        
        return {
            "observation_time": obs_time.iso,
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "elevation_m": elevation
            },
            "position": {
                "altitude_deg": round(sun_altaz.alt.deg, 2),
                "azimuth_deg": round(sun_altaz.az.deg, 2),
                "right_ascension": sun.ra.to_string(unit=u.hour, sep=":"),
                "declination": sun.dec.to_string(unit=u.deg, sep=":")
            },
            "is_daytime": sun_altaz.alt.deg > 0
        }
    except Exception as e:
        return {"error": str(e)}


def get_constellation_info(constellation_name: str) -> Dict[str, Any]:
    """
    Get information about a constellation.
    
    Args:
        constellation_name: Name of the constellation
        
    Returns:
        Dictionary with constellation information
    """
    # Basic constellation data
    constellations = {
        "orion": {
            "name": "Orion",
            "meaning": "The Hunter",
            "brightest_star": "Rigel (Beta Orionis)",
            "notable_objects": ["Orion Nebula (M42)", "Horsehead Nebula", "Betelgeuse", "Rigel"],
            "best_viewing": "December to March",
            "hemisphere": "Both (equatorial)",
            "area_sq_deg": 594
        },
        "ursa major": {
            "name": "Ursa Major",
            "meaning": "Great Bear",
            "brightest_star": "Alioth (Epsilon Ursae Majoris)",
            "notable_objects": ["Big Dipper asterism", "M81 galaxy", "M82 galaxy"],
            "best_viewing": "March to May",
            "hemisphere": "Northern",
            "area_sq_deg": 1280
        },
        "cassiopeia": {
            "name": "Cassiopeia",
            "meaning": "The Queen",
            "brightest_star": "Schedar (Alpha Cassiopeiae)",
            "notable_objects": ["M52 open cluster", "Heart Nebula", "Soul Nebula"],
            "best_viewing": "October to December",
            "hemisphere": "Northern",
            "area_sq_deg": 598
        },
        "andromeda": {
            "name": "Andromeda",
            "meaning": "The Chained Maiden",
            "brightest_star": "Alpheratz (Alpha Andromedae)",
            "notable_objects": ["Andromeda Galaxy (M31)", "M32 galaxy", "M110 galaxy"],
            "best_viewing": "October to December",
            "hemisphere": "Northern",
            "area_sq_deg": 722
        }
    }
    
    constellation_key = constellation_name.lower()
    if constellation_key in constellations:
        return constellations[constellation_key]
    else:
        return {
            "error": f"Constellation '{constellation_name}' not found in database.",
            "available": list(constellations.keys())
        }


# Tool definitions for OpenAI function calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_nasa_apod",
            "description": "Get NASA's Astronomy Picture of the Day with explanation",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format. Leave empty for today's APOD."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_near_earth_objects",
            "description": "Get information about Near Earth Objects (asteroids) approaching Earth",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format (optional)"
                    }
                },
                "required": ["start_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_planet_position",
            "description": "Calculate the position of a planet in the sky for a specific location and time",
            "parameters": {
                "type": "object",
                "properties": {
                    "planet_name": {
                        "type": "string",
                        "description": "Name of the planet (e.g., mars, jupiter, venus, saturn, mercury, uranus, neptune)"
                    },
                    "date": {
                        "type": "string",
                        "description": "Date/time in ISO format (optional, defaults to now)"
                    },
                    "latitude": {
                        "type": "number",
                        "description": "Observer latitude in degrees (default 0.0)"
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Observer longitude in degrees (default 0.0)"
                    },
                    "elevation": {
                        "type": "number",
                        "description": "Observer elevation in meters (default 0.0)"
                    }
                },
                "required": ["planet_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_moon_phase",
            "description": "Get the current moon phase and illumination percentage",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format (optional, defaults to today)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_sun_position",
            "description": "Calculate the Sun's position in the sky for a specific location and time",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date/time in ISO format (optional, defaults to now)"
                    },
                    "latitude": {
                        "type": "number",
                        "description": "Observer latitude in degrees (default 0.0)"
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Observer longitude in degrees (default 0.0)"
                    },
                    "elevation": {
                        "type": "number",
                        "description": "Observer elevation in meters (default 0.0)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_constellation_info",
            "description": "Get information about a specific constellation",
            "parameters": {
                "type": "object",
                "properties": {
                    "constellation_name": {
                        "type": "string",
                        "description": "Name of the constellation"
                    }
                },
                "required": ["constellation_name"]
            }
        }
    }
]


# Map of tool names to functions
TOOL_FUNCTIONS = {
    "get_nasa_apod": get_nasa_apod,
    "get_near_earth_objects": get_near_earth_objects,
    "calculate_planet_position": calculate_planet_position,
    "get_moon_phase": get_moon_phase,
    "calculate_sun_position": calculate_sun_position,
    "get_constellation_info": get_constellation_info
}


def get_available_tools() -> tuple[List[Dict[str, Any]], Dict[str, callable]]:
    """
    Get all available tools and their implementations.
    
    Returns:
        Tuple of (tool_definitions, tool_functions)
    """
    return TOOL_DEFINITIONS, TOOL_FUNCTIONS
